use std::fmt::Display;

use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

use crate::repositories::genotypes::Genotype;

#[instrument(level = "debug", skip(tx), fields(genotype_id = %genotype.id, type_name = %genotype.type_name, type_hash = genotype.type_hash, request_id = %genotype.request_id))]
pub(crate) async fn new_genotype<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotype: Genotype,
) -> Result<Genotype, super::Error> {
    let genotype = sqlx::query_as!(
        Genotype,
        r#"
            INSERT INTO fx_durable_ga.genotypes (
                id,
                generated_at,
                type_name,
                type_hash,
                genome,
                request_id,
                generation_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING
                id,
                generated_at,
                type_name,
                type_hash,
                genome,
                request_id,
                fitness,
                generation_id;
            "#,
        genotype.id,
        genotype.generated_at,
        genotype.type_name,
        genotype.type_hash,
        &genotype.genome,
        genotype.request_id,
        genotype.generation_id
    )
    .fetch_one(tx)
    .await?;

    Ok(genotype)
}

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

#[derive(Debug)]
pub(crate) struct Filter {
    ids: Option<Vec<Uuid>>,
    request_ids: Option<Vec<Uuid>>,
    has_fitness: Option<bool>,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            ids: None,
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

    #[instrument(level = "debug", fields(ids_count = ids.len()))]
    pub(crate) fn with_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.ids = Some(ids);
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
                $2::uuid[] IS NULL OR g.id = ANY($2)
            )
            AND (
                $3::uuid[] IS NULL OR g.request_id = ANY($3)
            );
        "#,
        filter.has_fitness,
        filter.ids.as_deref(),
        filter.request_ids.as_deref(),
    )
    .fetch_one(tx)
    .await?;

    Ok(count)
}

pub(crate) enum Order {
    Fitness,
    GeneratedAt,
    Random,
}

impl Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Order::Fitness => write!(f, "fitness"),
            Order::GeneratedAt => write!(f, "generated_at"),
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
            $2::uuid[] IS NULL OR g.id = ANY($2)
        )
        AND (
            $3::uuid[] IS NULL OR g.request_id = ANY($3)
        )
        ORDER BY
            CASE
                WHEN $4 = 'fitness' THEN g.fitness
                ELSE NULL
            END DESC NULLS LAST,
            CASE
                WHEN $4 = 'generated_at' THEN g.generated_at
                ELSE NULL
            END DESC NULLS LAST,
            CASE
                WHEN $4 = 'random' THEN RANDOM()
                ELSE NULL
            END NULLS LAST
        LIMIT $5;
        "#,
        filter.has_fitness,
        filter.ids.as_deref(),
        filter.request_ids.as_deref(),
        order.to_string(),
        limit
    )
    .fetch_all(tx)
    .await?;

    Ok(genotypes)
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
