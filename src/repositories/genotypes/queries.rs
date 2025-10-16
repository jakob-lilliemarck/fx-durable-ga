use crate::models::Genotype;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx), fields(genotypes_count = genotypes.len()))]
#[cfg(test)]
async fn new_genotypes<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotypes: Vec<Genotype>,
) -> Result<Vec<Genotype>, super::Error> {
    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.genotypes (
            id,
            generated_at,
            type_name,
            type_hash,
            genome
        ) VALUES ",
    );

    let mut first = true;
    // Stream serialization: serialize each event as needed rather than
    // pre-allocating all payloads, reducing peak memory usage
    for g in genotypes {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        query_builder
            .push("(")
            .push_bind(g.id)
            .push(", ")
            .push_bind(g.generated_at)
            .push(", ")
            .push_bind(g.type_name)
            .push(", ")
            .push_bind(g.type_hash)
            .push(", ")
            .push_bind(g.genome)
            .push(")");
    }
    query_builder.push(" RETURNING id, generated_at, type_name, type_hash, genome");

    let genotypes = query_builder
        .build_query_as::<Genotype>()
        .fetch_all(tx)
        .await?;

    Ok(genotypes)
}

#[cfg(test)]
mod new_genotypes_tests {
    use super::*;
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let genotypes = vec![Genotype::new("test", 1, vec![1, 2, 3])];
        let genotypes_clone = genotypes.clone();

        let inserted = new_genotypes(&pool, genotypes).await?;

        assert_eq!(genotypes_clone[0].id, inserted[0].id);
        assert_eq!(
            genotypes_clone[0].generated_at.trunc_subsecs(6),
            inserted[0].generated_at
        );
        assert_eq!(genotypes_clone[0].type_name, inserted[0].type_name);
        assert_eq!(genotypes_clone[0].type_hash, inserted[0].type_hash);
        assert_eq!(genotypes_clone[0].genome, inserted[0].genome);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let genotype = Genotype::new("test", 1, vec![1, 2, 3]);
        let genotype_clone = genotype.clone();

        new_genotypes(&pool, vec![genotype]).await?;
        let result = new_genotypes(&pool, vec![genotype_clone]).await;

        assert!(result.is_err());
        Ok(())
    }
}

#[instrument(level = "debug", skip(tx), fields(genotype_id = %id))]
pub(crate) async fn get_genotype<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: &Uuid,
) -> Result<Genotype, super::Error> {
    let genotype = sqlx::query_as!(
        Genotype,
        r#"
            SELECT
                id,
                generated_at,
                type_name,
                type_hash,
                genome
            FROM fx_durable_ga.genotypes
            WHERE id = $1;
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    Ok(genotype)
}

#[cfg(test)]
mod get_genotype_tests {
    use super::*;
    use chrono::{SubsecRound, Utc};

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let genotype = Genotype {
            id: Uuid::now_v7(),
            generated_at: Utc::now().trunc_subsecs(6),
            type_name: "test".to_string(),
            type_hash: 1,
            genome: vec![1, 2, 3],
        };
        let genotype_id = genotype.id;

        new_genotypes(&pool, vec![genotype]).await?;

        let selected = get_genotype(&pool, &genotype_id).await?;

        assert_eq!(genotype_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let non_existent_id = Uuid::now_v7();

        let result = get_genotype(&pool, &non_existent_id).await;

        assert!(result.is_err());
        Ok(())
    }
}

#[instrument(level = "debug", skip(tx), fields(genotype_id = ?ids))]
pub(crate) async fn get_genotypes<'tx, E: PgExecutor<'tx>>(
    tx: E,
    ids: &[Uuid],
) -> Result<Vec<Genotype>, super::Error> {
    let genotype = sqlx::query_as!(
        Genotype,
        r#"
            SELECT
                id,
                generated_at,
                type_name,
                type_hash,
                genome
            FROM fx_durable_ga.genotypes
            WHERE id = ANY($1)
            ORDER BY id;
        "#,
        ids
    )
    .fetch_all(tx)
    .await?;

    Ok(genotype)
}

const CREATE_GENERATION_IF_EMPTY_SQL: &str = r#"
    WITH generation_lock AS (
        SELECT id
        FROM fx_durable_ga.requests
        WHERE id = $1
        FOR UPDATE SKIP LOCKED
    )
    INSERT INTO fx_durable_ga.genotypes (
        id,
        generated_at,
        type_name,
        type_hash,
        genome
    )
    SELECT
        id,
        generated_at,
        type_name,
        type_hash,
        genome
    FROM (VALUES {values}) AS new_genotypes(
        id,
        generated_at,
        type_name,
        type_hash,
        genome
    )
    WHERE
        EXISTS(SELECT 1 FROM generation_lock)
        AND NOT EXISTS(
            SELECT 1 FROM fx_durable_ga.individuals
            WHERE request_id = $1 AND generation_id = $2
        )
    RETURNING
        id,
        generated_at,
        type_name,
        type_hash,
        genome;
"#;

#[instrument(level = "debug", skip(tx, genotypes), fields(request_id = %request_id, generation_id = generation_id, genotypes_count = genotypes.len()))]
pub(crate) async fn create_generation_if_empty<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
    generation_id: i32,
    genotypes: Vec<Genotype>,
) -> Result<Vec<Genotype>, super::Error> {
    if genotypes.is_empty() {
        return Ok(vec![]);
    }

    // Build the VALUES clause
    let values_clause = genotypes
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let base = i * 5 + 3; // Start after the 2 fixed parameters (request_id used twice, generation_id once)
            format!(
                "(${}, ${}, ${}, ${}, ${})",
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let sql = CREATE_GENERATION_IF_EMPTY_SQL.replace("{values}", &values_clause);

    let mut query = sqlx::query_as::<_, Genotype>(&sql)
        .bind(request_id)
        .bind(generation_id);

    // Bind all genotype values
    for g in &genotypes {
        query = query
            .bind(g.id)
            .bind(g.generated_at)
            .bind(&g.type_name)
            .bind(g.type_hash)
            .bind(&g.genome)
    }

    let inserted_genotypes = query.fetch_all(tx).await?;
    Ok(inserted_genotypes)
}

#[cfg(test)]
mod create_generation_if_empty_tests {
    use super::*;
    use crate::models::{Crossover, FitnessGoal, Mutagen, Request, Strategy};
    use crate::repositories::requests::queries::new_request;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_creates_generation_when_empty(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first - the locking mechanism requires the request to exist
        // because it uses "SELECT id FROM fx_durable_ga.requests WHERE id = $1 FOR UPDATE SKIP LOCKED"
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        // Create genotypes for generation 1
        let genotypes = vec![
            Genotype::new("test", 1, vec![1, 2, 3]),
            Genotype::new("test", 1, vec![4, 5, 6]),
        ];

        let result = create_generation_if_empty(&pool, request_id, 1, genotypes).await?;

        assert_eq!(result.len(), 2);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_returns_empty_when_generation_exists(pool: sqlx::PgPool) -> anyhow::Result<()> {
        use crate::models::Individual;
        use crate::repositories::populations::queries::add_to_population;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        // First call - should succeed
        let genotypes = vec![Genotype::new("test", 1, vec![1, 2, 3])];
        let genotype_id = genotypes[0].id;
        let first_result = create_generation_if_empty(&pool, request_id, 1, genotypes).await?;
        assert_eq!(first_result.len(), 1);

        // Add the genotype as an individual to generation 1
        let individuals = vec![Individual::new(genotype_id, request_id, 1)];
        add_to_population(&pool, &individuals).await?;

        // Second call - should return empty because generation 1 now has individuals
        let more_genotypes = vec![Genotype::new("test", 1, vec![4, 5, 6])];
        let second_result =
            create_generation_if_empty(&pool, request_id, 1, more_genotypes).await?;
        assert_eq!(second_result.len(), 0);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_returns_empty_for_empty_input(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        // Empty genotypes should return empty immediately
        let result = create_generation_if_empty(&pool, request_id, 1, vec![]).await?;
        assert_eq!(result.len(), 0);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_handles_concurrent_creation(pool: sqlx::PgPool) -> anyhow::Result<()> {
        use std::sync::Arc;
        use tokio::task;
        use crate::models::Individual;
        use crate::repositories::populations::queries::add_to_population;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Strategy::Generational {
                max_generations: 100,
                population_size: 10,
            },
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        // Create first genotype and add as individual to establish generation 1
        let initial_genotypes = vec![Genotype::new("test", 1, vec![0, 0, 0])];
        let genotype_id = initial_genotypes[0].id;
        let first_result = create_generation_if_empty(&pool, request_id, 1, initial_genotypes).await?;
        assert_eq!(first_result.len(), 1);
        
        let individuals = vec![Individual::new(genotype_id, request_id, 1)];
        add_to_population(&pool, &individuals).await?;

        let pool = Arc::new(pool);

        // Now spawn two concurrent tasks trying to create more genotypes for the same generation
        // Both should return empty since generation 1 already has individuals
        let pool1 = pool.clone();
        let task1 = task::spawn(async move {
            let genotypes = vec![Genotype::new("test", 1, vec![1, 2, 3])];
            create_generation_if_empty(&*pool1, request_id, 1, genotypes).await
        });

        let pool2 = pool.clone();
        let task2 = task::spawn(async move {
            let genotypes = vec![Genotype::new("test", 1, vec![4, 5, 6])];
            create_generation_if_empty(&*pool2, request_id, 1, genotypes).await
        });

        let (result1, result2) = tokio::try_join!(task1, task2)?;
        let result1 = result1?;
        let result2 = result2?;

        // Both should return empty since generation 1 already exists
        let total_inserted = result1.len() + result2.len();
        assert_eq!(total_inserted, 0);

        Ok(())
    }
}
