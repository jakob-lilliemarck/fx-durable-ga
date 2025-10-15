use crate::models::Genotype;
use sqlx::PgExecutor;
use std::fmt::Display;
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
            genome,
            request_id,
            generation_id
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
            .push(", ")
            .push_bind(g.request_id)
            .push(", ")
            .push_bind(g.generation_id)
            .push(")");
    }
    query_builder.push(" RETURNING id, generated_at, type_name, type_hash, genome, request_id, generation_id, fitness");

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
        let request_id = Uuid::now_v7();
        let genotypes = vec![Genotype::new("test", 1, vec![1, 2, 3], request_id, 1)];
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
        assert_eq!(genotypes_clone[0].request_id, inserted[0].request_id);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);
        let genotype_clone = genotype.clone();

        new_genotypes(&pool, vec![genotype]).await?;
        let result = new_genotypes(&pool, vec![genotype_clone]).await;

        assert!(result.is_err());
        Ok(())
    }
}

#[instrument(level = "debug", skip(tx), fields(genotype_id = %id, fitness = fitness))]
pub(crate) async fn set_fitness<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: Uuid,
    fitness: f64,
) -> Result<(), super::Error> {
    sqlx::query!(
        r#"
            UPDATE fx_durable_ga.genotypes
            SET fitness = $2
            WHERE id = $1 AND fitness IS NULL
            RETURNING id;
        "#,
        id,
        fitness
    )
    .fetch_one(tx)
    .await?;

    Ok(())
}

#[cfg(test)]
mod set_fitness_tests {
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_updates_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);

        let inserted = new_genotypes(&pool, vec![genotype]).await?;
        set_fitness(&pool, inserted[0].id, 0.9).await?;

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_if_fitness_is_already_set(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();
        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);

        let inserted = new_genotypes(&pool, vec![genotype]).await?;
        set_fitness(&pool, inserted[0].id, 0.9).await?;
        let result = set_fitness(&pool, inserted[0].id, 0.8).await;

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
                genome,
                request_id,
                fitness,
                generation_id
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
            request_id: Uuid::now_v7(),
            fitness: None,
            generation_id: 1,
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

#[derive(Debug)]
pub(crate) struct Filter {
    request_ids: Option<Vec<Uuid>>,
    has_fitness: Option<bool>,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            has_fitness: None,
            request_ids: None,
        }
    }
}

impl Filter {
    #[instrument(level = "debug", fields(has_fitness = has_fitness))]
    pub(crate) fn with_fitness(mut self, has_fitness: bool) -> Self {
        self.has_fitness = Some(has_fitness);
        self
    }

    #[instrument(level = "debug", fields(request_ids_count = request_ids.len()))]
    pub(crate) fn with_request_ids(mut self, request_ids: Vec<Uuid>) -> Self {
        self.request_ids = Some(request_ids);
        self
    }
}

#[instrument(level = "debug", skip(tx, filter))]
pub(crate) async fn count_genotypes_in_latest_iteration<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &Filter,
) -> Result<i64, super::Error> {
    let count = sqlx::query_scalar!(
        r#"
            WITH latest_generations AS (
                SELECT request_id, MAX(generation_id) as max_generation_id
                FROM fx_durable_ga.genotypes
                GROUP BY request_id
            )
            SELECT COUNT(*) "count!:i64"
            FROM fx_durable_ga.genotypes g
            JOIN latest_generations lg
                ON g.request_id = lg.request_id
                AND g.generation_id = lg.max_generation_id
            WHERE (
                $1::bool IS NULL OR
                CASE
                    WHEN $1 = true THEN g.fitness IS NOT NULL
                    ELSE g.fitness IS NULL
                END
            )
            AND (
                $2::uuid[] IS NULL OR g.request_id = ANY($2)
            );
        "#,
        filter.has_fitness,
        filter.request_ids.as_deref(),
    )
    .fetch_one(tx)
    .await?;

    Ok(count)
}

#[cfg(test)]
mod count_genotypes_in_latest_iteration_tests {
    use super::*;
    use crate::repositories::genotypes::Filter;
    use chrono::{SubsecRound, Utc};

    #[sqlx::test(migrations = "./migrations")]
    async fn it_counts_all_genotypes(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_counts_genotypes_with_fitness(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_counts_genotypes_without_fitness(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_counts_genotypes_of_requests(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_counts_some_genotypes(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let req_id_1 = Uuid::now_v7();
        let req_id_2 = Uuid::now_v7();

        assert_eq!(
            count_genotypes_in_latest_iteration(&pool, &Filter::default()).await?,
            0
        );

        new_genotypes(
            &pool,
            vec![Genotype {
                id: Uuid::now_v7(),
                generated_at: Utc::now().trunc_subsecs(6),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                request_id: req_id_1,
                fitness: None,
                generation_id: 1,
            }],
        )
        .await?;
        assert_eq!(
            count_genotypes_in_latest_iteration(&pool, &Filter::default()).await?,
            1
        );

        // SECOND
        let genotype_id_2 = Uuid::now_v7();
        new_genotypes(
            &pool,
            vec![Genotype {
                id: genotype_id_2,
                generated_at: Utc::now().trunc_subsecs(6),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                request_id: req_id_1,
                fitness: None,
                generation_id: 1,
            }],
        )
        .await?;
        set_fitness(&pool, genotype_id_2, 0.123).await?;

        assert_eq!(
            count_genotypes_in_latest_iteration(&pool, &Filter::default()).await?,
            2
        );
        assert_eq!(
            count_genotypes_in_latest_iteration(&pool, &Filter::default().with_fitness(false))
                .await?,
            1
        );
        assert_eq!(
            count_genotypes_in_latest_iteration(&pool, &Filter::default().with_fitness(true))
                .await?,
            1
        );

        // THIRD
        new_genotypes(
            &pool,
            vec![Genotype {
                id: Uuid::now_v7(),
                generated_at: Utc::now().trunc_subsecs(6),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                request_id: req_id_2,
                fitness: None,
                generation_id: 1,
            }],
        )
        .await?;
        assert_eq!(
            count_genotypes_in_latest_iteration(
                &pool,
                &Filter::default().with_request_ids(vec![req_id_1])
            )
            .await?,
            2
        );
        assert_eq!(
            count_genotypes_in_latest_iteration(
                &pool,
                &Filter::default().with_request_ids(vec![req_id_2])
            )
            .await?,
            1
        );
        assert_eq!(
            count_genotypes_in_latest_iteration(
                &pool,
                &Filter::default().with_request_ids(vec![req_id_1, req_id_2])
            )
            .await?,
            3
        );
        assert_eq!(
            count_genotypes_in_latest_iteration(
                &pool,
                &Filter::default()
                    .with_request_ids(vec![req_id_1, req_id_2])
                    .with_fitness(false)
            )
            .await?,
            2
        );
        assert_eq!(
            count_genotypes_in_latest_iteration(
                &pool,
                &Filter::default()
                    .with_request_ids(vec![req_id_1, req_id_2])
                    .with_fitness(true)
            )
            .await?,
            1
        );
        Ok(())
    }
}

pub(crate) enum Order {
    Fitness,
    Random,
}

impl Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Order::Fitness => write!(f, "fitness"),
            Order::Random => write!(f, "random"),
        }
    }
}

#[instrument(level = "debug", skip(tx, filter), fields(limit = limit, order = %order))]
pub(crate) async fn search_genotypes_in_latest_generation<'tx, E: PgExecutor<'tx>>(
    tx: E,
    limit: i64,
    order: Order,
    filter: &Filter,
) -> Result<Vec<Genotype>, super::Error> {
    // Return i64, not Genotype - we're counting!
    let genotypes = sqlx::query_as!(
        Genotype,
        r#"
        WITH latest_generations AS (
            SELECT request_id, MAX(generation_id) as max_generation_id
            FROM fx_durable_ga.genotypes
            GROUP BY request_id
        )
        SELECT
            g.id,
            g.generated_at,
            g.type_name,
            g.type_hash,
            g.genome,
            g.request_id,
            g.fitness,
            g.generation_id
        FROM fx_durable_ga.genotypes g
        JOIN latest_generations lg
            ON g.request_id = lg.request_id
            AND g.generation_id = lg.max_generation_id
        WHERE (
            $1::bool IS NULL OR
            CASE
                WHEN $1 = true THEN g.fitness IS NOT NULL
                ELSE g.fitness IS NULL
            END
        )
        AND (
            $2::uuid[] IS NULL OR g.request_id = ANY($2)
        )
        ORDER BY
            CASE
                WHEN $3 = 'fitness' THEN g.fitness
                ELSE NULL
            END DESC NULLS LAST,
            CASE
                WHEN $3 = 'random' THEN RANDOM()
                ELSE NULL
            END NULLS LAST
        LIMIT $4;
        "#,
        filter.has_fitness,
        filter.request_ids.as_deref(),
        order.to_string(),
        limit
    )
    .fetch_all(tx)
    .await?;

    Ok(genotypes)
}

#[cfg(test)]
mod search_genotypes_in_latest_generation_tests {
    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_all_genotypes_of_generation(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_with_fitness(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_without_fitness(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_of_requests(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_orders_by_fitness(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_orders_by_random(_pool: sqlx::PgPool) -> anyhow::Result<()> {
        todo!()
    }
}

#[instrument(level = "debug", skip(tx), fields(request_id = %request_id))]
pub(crate) async fn get_generation_count<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
) -> Result<i32, super::Error> {
    let count = sqlx::query_scalar!(
        r#"
            SELECT COALESCE(MAX(generation_id), 0) "generation_id!:i32"
            FROM fx_durable_ga.genotypes
            WHERE request_id = $1
        "#,
        request_id
    )
    .fetch_one(tx)
    .await?;

    Ok(count)
}

#[cfg(test)]
mod get_generation_count_tests {
    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_the_number_of_generations_of_a_request(
        _pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        todo!()
    }
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
        genome,
        request_id,
        generation_id
    )
    SELECT
        id,
        generated_at,
        type_name,
        type_hash,
        genome,
        request_id,
        generation_id
    FROM (VALUES {values}) AS new_genotypes(
        id,
        generated_at,
        type_name,
        type_hash,
        genome,
        request_id,
        generation_id
    )
    WHERE
        EXISTS(SELECT 1 FROM generation_lock)
        AND NOT EXISTS(
            SELECT 1 FROM fx_durable_ga.genotypes
            WHERE request_id = $2 AND generation_id = $3
        )
    RETURNING
        id,
        generated_at,
        type_name,
        type_hash,
        genome,
        request_id,
        generation_id,
        fitness;
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
            let base = i * 7 + 4; // Start after the 3 fixed parameters
            format!(
                "(${}, ${}, ${}, ${}, ${}, ${}, ${})",
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let sql = CREATE_GENERATION_IF_EMPTY_SQL.replace("{values}", &values_clause);

    let mut query = sqlx::query_as::<_, Genotype>(&sql)
        .bind(request_id)
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
            .bind(g.request_id)
            .bind(g.generation_id);
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
            Genotype::new("test", 1, vec![1, 2, 3], request_id, 1),
            Genotype::new("test", 1, vec![4, 5, 6], request_id, 1),
        ];

        let result = create_generation_if_empty(&pool, request_id, 1, genotypes).await?;

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].request_id, request_id);
        assert_eq!(result[0].generation_id, 1);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_returns_empty_when_generation_exists(pool: sqlx::PgPool) -> anyhow::Result<()> {
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
        let genotypes = vec![Genotype::new("test", 1, vec![1, 2, 3], request_id, 1)];
        let first_result = create_generation_if_empty(&pool, request_id, 1, genotypes).await?;
        assert_eq!(first_result.len(), 1);

        // Second call - should return empty because generation exists
        let more_genotypes = vec![Genotype::new("test", 1, vec![4, 5, 6], request_id, 1)];
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

        let pool = Arc::new(pool);

        // Spawn two concurrent tasks trying to create the same generation
        let pool1 = pool.clone();
        let task1 = task::spawn(async move {
            let genotypes = vec![Genotype::new("test", 1, vec![1, 2, 3], request_id, 1)];
            create_generation_if_empty(&*pool1, request_id, 1, genotypes).await
        });

        let pool2 = pool.clone();
        let task2 = task::spawn(async move {
            let genotypes = vec![Genotype::new("test", 1, vec![4, 5, 6], request_id, 1)];
            create_generation_if_empty(&*pool2, request_id, 1, genotypes).await
        });

        let (result1, result2) = tokio::try_join!(task1, task2)?;
        let result1 = result1?;
        let result2 = result2?;

        // Exactly one should succeed, one should return empty
        let total_inserted = result1.len() + result2.len();
        assert_eq!(total_inserted, 1);

        Ok(())
    }
}
