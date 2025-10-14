use crate::models::Genotype;
use sqlx::PgExecutor;
use std::fmt::Display;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx), fields(genotypes_count = genotypes.len()))]
pub(crate) async fn new_genotypes<'tx, E: PgExecutor<'tx>>(
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
