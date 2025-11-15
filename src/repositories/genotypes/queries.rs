use crate::models::{Fitness, Genotype, Population};
use sqlx::PgExecutor;
use std::fmt::Display;
use tracing::instrument;
use uuid::Uuid;

/// Inserts multiple genotypes into the database in a single transaction.
/// Returns the inserted genotypes with database-generated fields.
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
            genome_hash,
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

        let type_name = g.type_name().to_string();
        query_builder
            .push("(")
            .push_bind(g.id())
            .push(", ")
            .push_bind(g.generated_at())
            .push(", ")
            .push_bind(type_name)
            .push(", ")
            .push_bind(g.type_hash())
            .push(", ")
            .push_bind(g.genome())
            .push(", ")
            .push_bind(g.genome_hash())
            .push(", ")
            .push_bind(g.request_id())
            .push(", ")
            .push_bind(g.generation_id())
            .push(")");
    }
    query_builder.push(
        " RETURNING id, generated_at, type_name, type_hash, genome, genome_hash, request_id, generation_id",
    );

    let genotypes = query_builder
        .build_query_as::<Genotype>()
        .fetch_all(tx)
        .await?;

    Ok(genotypes)
}

#[cfg(test)]
mod new_genotypes_tests {
    use super::*;
    use crate::models::{
        Crossover, Distribution, FitnessGoal, Mutagen, Request, Schedule, Selector,
    };
    use crate::repositories::requests::queries::new_request;
    use chrono::SubsecRound;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotypes = vec![Genotype::new("test", 1, vec![1, 2, 3], request_id, 1)];
        let genotypes_clone = genotypes.clone();

        let inserted = new_genotypes(&pool, genotypes).await?;

        assert_eq!(genotypes_clone[0].id(), inserted[0].id());
        assert_eq!(
            genotypes_clone[0].generated_at().trunc_subsecs(6),
            inserted[0].generated_at()
        );
        assert_eq!(genotypes_clone[0].type_name(), inserted[0].type_name());
        assert_eq!(genotypes_clone[0].type_hash(), inserted[0].type_hash());
        assert_eq!(genotypes_clone[0].genome(), inserted[0].genome());

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotype = Genotype::new("test", 1, vec![1, 2, 3], request_id, 1);
        let genotype_clone = genotype.clone();

        new_genotypes(&pool, vec![genotype]).await?;
        let result = new_genotypes(&pool, vec![genotype_clone]).await;

        assert!(result.is_err());
        Ok(())
    }
}

/// Checks if any genotypes exist for the given request and generation.
#[instrument(level = "debug", skip(tx), fields(request_id = %request_id, generation_id=%generation_id))]
pub(crate) async fn check_if_generation_exists<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
    generation_id: i32,
) -> Result<bool, super::Error> {
    let exists = sqlx::query_scalar!(
        r#"
            SELECT EXISTS(
                SELECT 1
                FROM fx_durable_ga.genotypes
                WHERE request_id = $1 AND generation_id = $2
            ) "exists!:bool";
        "#,
        request_id,
        generation_id,
    )
    .fetch_one(tx)
    .await?;

    Ok(exists)
}

#[cfg(test)]
mod check_if_generation_exists_tests {
    use uuid::Uuid;

    use crate::{
        models::{Crossover, Distribution, FitnessGoal, Mutagen, Request, Schedule, Selector},
        repositories::{
            genotypes::{new_genotypes, queries::check_if_generation_exists},
            requests::queries::new_request,
        },
    };

    use super::Genotype;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_returns_true_when_generation_exists(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let mut genotypes = Vec::with_capacity(3);

        genotypes.push(Genotype {
            id: Uuid::now_v7(),
            generated_at: chrono::Utc::now(),
            type_name: "test".to_string(),
            type_hash: 1,
            genome: vec![1, 2, 3],
            genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
            request_id,
            generation_id: 1,
        });

        new_genotypes(&pool, genotypes).await?;

        let exists = check_if_generation_exists(&pool, request_id, 1).await?;
        assert!(exists);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_returns_false_when_none_exist(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let exists = check_if_generation_exists(&pool, request_id, 1).await?;
        assert!(!exists);
        Ok(())
    }
}

/// Retrieves a single genotype by its ID.
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
                genome_hash,
                request_id,
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
    use crate::models::{
        Crossover, Distribution, FitnessGoal, Mutagen, Request, Schedule, Selector,
    };
    use crate::repositories::requests::queries::new_request;
    use chrono::{SubsecRound, Utc};

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotype = Genotype {
            id: Uuid::now_v7(),
            generated_at: Utc::now().trunc_subsecs(6),
            type_name: "test".to_string(),
            type_hash: 1,
            genome: vec![1, 2, 3],
            genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
            request_id,
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

/// Records a fitness evaluation result for a genotype.
#[instrument(level = "debug", skip(tx), fields(fitness = ?fitness))]
pub(crate) async fn record_fitness<'tx, E: PgExecutor<'tx>>(
    tx: E,
    fitness: &Fitness,
) -> Result<Fitness, super::Error> {
    let inserted = sqlx::query_as!(
        Fitness,
        r#"
            INSERT INTO fx_durable_ga.fitness (genotype_id, fitness, evaluated_at)
            VALUES ($1, $2, $3)
            RETURNING genotype_id, fitness, evaluated_at;
        "#,
        fitness.genotype_id,
        fitness.fitness,
        fitness.evaluated_at
    )
    .fetch_one(tx)
    .await?;

    Ok(inserted) // ← Return the inserted record
}

#[cfg(test)]
mod record_fitness_tests {
    use super::record_fitness;
    use crate::models::{
        Crossover, Distribution, Fitness, FitnessGoal, Genotype, Mutagen, Request, Schedule,
        Selector,
    };
    use crate::repositories::genotypes::new_genotypes;
    use crate::repositories::requests::queries::new_request;
    use chrono::SubsecRound;
    use uuid::Uuid;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_records_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotype_id = Uuid::now_v7();

        // Create a genotype
        let genotype = Genotype {
            id: genotype_id,
            generated_at: chrono::Utc::now(),
            type_name: "test".to_string(),
            type_hash: 1,
            genome: vec![1, 2, 3],
            genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
            request_id,
            generation_id: 1,
        };
        new_genotypes(&pool, vec![genotype]).await?;

        let fitness = Fitness::new(genotype_id, 0.543);
        let recorded = record_fitness(&pool, &fitness).await?;

        assert_eq!(recorded.genotype_id, fitness.genotype_id);
        assert_eq!(recorded.fitness, fitness.fitness);
        assert_eq!(recorded.evaluated_at, fitness.evaluated_at.trunc_subsecs(6));
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_clone = request.clone();

        let first = new_request(&pool, request).await;
        let second = new_request(&pool, request_clone).await;

        assert!(first.is_ok());
        assert!(second.is_err());
        Ok(())
    }
}

/// Gets population statistics for an optimization request.
/// Returns default values if no genotypes exist yet.
#[instrument(level = "debug", skip(tx), fields(request_id = %request_id))]
pub(crate) async fn get_population<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: &Uuid,
) -> Result<Population, super::Error> {
    let state = sqlx::query_as!(
        Population,
        r#"
            SELECT
                request_id "request_id!:Uuid",
                evaluated_genotypes "evaluated_genotypes!:i64",
                live_genotypes "live_genotypes!:i64",
                current_generation "current_generation!:i32",
                min_fitness "min_fitness",
                max_fitness "max_fitness"
            FROM fx_durable_ga.populations
            WHERE request_id = $1;
        "#,
        request_id
    )
    .fetch_optional(tx)
    .await?;

    // Handle case where request has no genotypes yet
    Ok(state.unwrap_or(Population {
        request_id: *request_id,
        evaluated_genotypes: 0,
        live_genotypes: 0,
        current_generation: 0,
        min_fitness: None,
        max_fitness: None,
    }))
}

#[cfg(test)]
mod get_population_tests {
    use super::{get_population, record_fitness};
    use crate::models::{
        Crossover, Distribution, Fitness, FitnessGoal, Genotype, Mutagen, Population, Request,
        Schedule, Selector,
    };
    use crate::repositories::genotypes::new_genotypes;
    use crate::repositories::requests::queries::new_request;
    use uuid::Uuid;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_a_population(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;
        let mut genotypes = Vec::with_capacity(3);
        for i in 1..=3 {
            genotypes.push(Genotype {
                id: Uuid::now_v7(),
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
                request_id,
                generation_id: i as i32,
            });
        }
        new_genotypes(&pool, genotypes).await?;

        let population = get_population(&pool, &request_id).await?;
        assert_eq!(
            population,
            Population {
                request_id,
                evaluated_genotypes: 0,
                live_genotypes: 3,
                current_generation: 3,
                min_fitness: None,
                max_fitness: None,
            }
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_handles_empty_population(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();

        let population = get_population(&pool, &request_id).await?;

        assert_eq!(
            population,
            Population {
                request_id,
                evaluated_genotypes: 0,
                live_genotypes: 0,
                current_generation: 0,
                min_fitness: None,
                max_fitness: None,
            }
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_population_with_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1)?,
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(200),
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        // Create genotypes
        let a_id = Uuid::now_v7();
        let b_id = Uuid::now_v7();
        let c_id = Uuid::now_v7();

        let genotypes = vec![
            Genotype {
                id: a_id,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
                request_id,
                generation_id: 1,
            },
            Genotype {
                id: b_id,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![4, 5, 6],
                genome_hash: Genotype::compute_genome_hash(&[4, 5, 6]),
                request_id,
                generation_id: 1,
            },
            Genotype {
                id: c_id,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![7, 8, 9],
                genome_hash: Genotype::compute_genome_hash(&[7, 8, 9]),
                request_id,
                generation_id: 1,
            },
        ];

        new_genotypes(&pool, genotypes).await?;

        // Record fitness values
        record_fitness(&pool, &Fitness::new(a_id, 0.50)).await?;
        record_fitness(&pool, &Fitness::new(b_id, 0.99)).await?;
        record_fitness(&pool, &Fitness::new(c_id, 0.01)).await?;

        let population = get_population(&pool, &request_id).await?;

        assert_eq!(
            population,
            Population {
                request_id,
                evaluated_genotypes: 3,
                live_genotypes: 0,
                current_generation: 1,
                min_fitness: Some(0.01),
                max_fitness: Some(0.99),
            }
        );

        Ok(())
    }
}

/// SQL sort order for query results.
#[derive(Debug)]
enum SortOrder {
    Asc,
    Desc,
}

impl Display for SortOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Asc => write!(f, "asc"),
            Self::Desc => write!(f, "desc"),
        }
    }
}

/// Query ordering options for genotype searches.
#[derive(Debug)]
enum Order {
    Random,
    Fitness(SortOrder),
}

impl Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Random => write!(f, "random"),
            Self::Fitness(order) => write!(f, "fitness_{}", order),
        }
    }
}

/// Filter criteria for searching genotypes with various conditions.
#[derive(Debug)]
pub(crate) struct Filter {
    request_id: Option<Uuid>,
    generation_id: Option<i32>,
    has_fitness: Option<bool>,
    order: Option<Order>,
}

impl Default for Filter {
    fn default() -> Self {
        Filter {
            request_id: None,
            generation_id: None,
            has_fitness: None,
            order: None,
        }
    }
}

impl Filter {
    /// Filters genotypes by request ID.
    pub(crate) fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn with_generation_id(mut self, generation_id: i32) -> Self {
        self.generation_id = Some(generation_id);
        self
    }

    /// Filters genotypes based on whether they have fitness evaluations.
    pub(crate) fn with_fitness(mut self, has_fitness: bool) -> Self {
        self.has_fitness = Some(has_fitness);
        self
    }

    pub(crate) fn with_order_random(mut self) -> Self {
        self.order = Some(Order::Random);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn with_order_fitness_asc(mut self) -> Self {
        self.order = Some(Order::Fitness(SortOrder::Asc));
        self
    }

    #[allow(dead_code)]
    pub(crate) fn with_order_fitness_desc(mut self) -> Self {
        self.order = Some(Order::Fitness(SortOrder::Desc));
        self
    }
}

/// Searches genotypes with optional filtering, ordering, and limits.
/// Returns genotypes paired with their fitness values (if available).
#[instrument(level = "debug", skip(tx), fields(filter = ?filter))]
pub(crate) async fn search_genotypes<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &Filter,
    limit: i64,
) -> Result<Vec<(Genotype, Option<f64>)>, super::Error> {
    let rows = sqlx::query!(
        r#"
            SELECT
                g.id,
                g.generated_at,
                g.type_name,
                g.type_hash,
                g.genome,
                g.genome_hash,
                g.request_id,
                g.generation_id,
                f.fitness "fitness!:Option<f64>"
            FROM fx_durable_ga.genotypes g
            LEFT JOIN fx_durable_ga.fitness f ON g.id = f.genotype_id
            WHERE (
                $1::uuid IS NULL OR g.request_id = $1
            )
            AND (
                $2::int IS NULL OR g.generation_id = $2
            )
            AND (
                $3::bool IS NULL OR
                CASE
                    WHEN $3 = true THEN f.fitness IS NOT NULL
                    ELSE f.fitness IS NULL
                END
            )
            ORDER BY
                CASE
                    WHEN $4 = 'fitness_desc' THEN f.fitness
                    ELSE NULL
                END DESC NULLS LAST,
                CASE
                    WHEN $4 = 'fitness_asc' THEN f.fitness
                    ELSE NULL
                END ASC NULLS LAST,
                CASE
                    WHEN $4 = 'random' THEN RANDOM()
                    ELSE NULL
                END NULLS LAST,
                g.id ASC
            LIMIT $5;
        "#,
        filter.request_id,
        filter.generation_id,
        filter.has_fitness,
        filter.order.as_ref().map(|o| o.to_string()),
        limit
    )
    .fetch_all(tx)
    .await?;

    let genotypes_with_fitness = rows
        .into_iter()
        .map(|row| {
            let genotype = Genotype {
                id: row.id,
                generated_at: row.generated_at,
                type_name: row.type_name,
                type_hash: row.type_hash,
                genome: row.genome,
                genome_hash: row.genome_hash,
                request_id: row.request_id,
                generation_id: row.generation_id,
            };
            (genotype, row.fitness)
        })
        .collect();

    Ok(genotypes_with_fitness)
}

#[cfg(test)]
mod search_genotypes_tests {
    use super::{Filter, search_genotypes};
    use uuid::Uuid;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_with_request_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let (request_id_1, _) = super::seeding::seed(&pool).await;

        let found =
            search_genotypes(&pool, &Filter::default().with_request_id(request_id_1), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();
        assert_eq!(
            vec![
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                    Some(0.11)
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
                    None
                )
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_with_generation_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        super::seeding::seed(&pool).await;

        let found = search_genotypes(&pool, &Filter::default().with_generation_id(2), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();
        assert_eq!(
            vec![
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
                    None
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000005").unwrap(),
                    None
                )
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_with_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        super::seeding::seed(&pool).await;

        let found = search_genotypes(&pool, &Filter::default().with_fitness(true), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();
        assert_eq!(
            vec![
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                    Some(0.11)
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
                    Some(0.12)
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap(),
                    Some(0.42)
                )
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_with_fitness_desc(pool: sqlx::PgPool) -> anyhow::Result<()> {
        super::seeding::seed(&pool).await;

        let found = search_genotypes(
            &pool,
            &Filter::default()
                .with_fitness(true)
                .with_order_fitness_desc(),
            5,
        )
        .await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();
        assert_eq!(
            vec![
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap(),
                    Some(0.42)
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
                    Some(0.12)
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                    Some(0.11)
                ),
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_with_fitness_asc(pool: sqlx::PgPool) -> anyhow::Result<()> {
        super::seeding::seed(&pool).await;

        let found = search_genotypes(
            &pool,
            &Filter::default()
                .with_fitness(true)
                .with_order_fitness_asc(),
            5,
        )
        .await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();
        assert_eq!(
            vec![
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                    Some(0.11)
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap(),
                    Some(0.12)
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap(),
                    Some(0.42)
                ),
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_without_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        super::seeding::seed(&pool).await;

        let found = search_genotypes(&pool, &Filter::default().with_fitness(false), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();
        assert_eq!(
            vec![
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
                    None
                ),
                (
                    Uuid::parse_str("00000000-0000-0000-0000-000000000005").unwrap(),
                    None
                )
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_genotypes_with_request_id_fitness_and_random_order(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        let (request_id_1, _) = super::seeding::seed(&pool).await;

        let found = search_genotypes(
            &pool,
            &Filter::default()
                .with_request_id(request_id_1)
                .with_fitness(true)
                .with_order_random(),
            5,
        )
        .await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();

        assert_eq!(
            vec![(
                Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
                Some(0.11)
            ),],
            actual
        );

        Ok(())
    }
}

/// Finds which genome hashes already exist for a given request.
/// Used for deduplication during breeding to avoid creating duplicate genomes.
#[instrument(level = "debug", skip(tx), fields(request_id=?request_id))]
pub(crate) async fn get_intersection<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
    hashes: &[i64],
) -> Result<Vec<i64>, super::Error> {
    let intersection = sqlx::query_scalar!(
        r#"
            SELECT genome_hash
            FROM fx_durable_ga.genotypes
            WHERE request_id = $1 AND genome_hash = ANY($2);
        "#,
        request_id,
        hashes,
    )
    .fetch_all(tx)
    .await?;

    Ok(intersection)
}

#[cfg(test)]
mod tests {
    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_the_intersection(pool: sqlx::PgPool) -> anyhow::Result<()> {
        use crate::models::Genotype;

        let (request_id_1, _request_id_2) = super::seeding::seed(&pool).await;

        // Test hashes for genomes in request_id_1: [1,2,3] and [4,5,6]
        let hash_1_2_3 = Genotype::compute_genome_hash(&[1, 2, 3]);
        let hash_4_5_6 = Genotype::compute_genome_hash(&[4, 5, 6]);
        let hash_nonexistent = Genotype::compute_genome_hash(&[99, 100, 101]);

        let candidate_hashes = vec![hash_1_2_3, hash_4_5_6, hash_nonexistent];

        let intersection = super::get_intersection(&pool, request_id_1, &candidate_hashes).await?;

        // Should return the two hashes that exist for request_id_1
        assert_eq!(intersection.len(), 2);
        assert!(intersection.contains(&hash_1_2_3));
        assert!(intersection.contains(&hash_4_5_6));

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_returns_empty_vector_when_there_is_no_intersection(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        use crate::models::Genotype;

        let (request_id_1, _) = super::seeding::seed(&pool).await;

        // Test with hashes that don't exist in the database
        let nonexistent_hashes = vec![
            Genotype::compute_genome_hash(&[99, 100, 101]),
            Genotype::compute_genome_hash(&[200, 201, 202]),
        ];

        let intersection =
            super::get_intersection(&pool, request_id_1, &nonexistent_hashes).await?;

        assert!(intersection.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_isolates_intersection_by_request_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        use crate::models::Genotype;

        let (request_id_1, request_id_2) = super::seeding::seed(&pool).await;

        // Test with hashes from request_2: [7,8,9], [10,11,12], [13,14,15]
        let hash_7_8_9 = Genotype::compute_genome_hash(&[7, 8, 9]);
        let hash_10_11_12 = Genotype::compute_genome_hash(&[10, 11, 12]);

        let candidate_hashes = vec![hash_7_8_9, hash_10_11_12];

        // Query for request_1 should return empty (these hashes exist but in request_2)
        let intersection = super::get_intersection(&pool, request_id_1, &candidate_hashes).await?;
        assert!(intersection.is_empty());

        // Query for request_2 should return both hashes
        let intersection = super::get_intersection(&pool, request_id_2, &candidate_hashes).await?;
        assert_eq!(intersection.len(), 2);
        assert!(intersection.contains(&hash_7_8_9));
        assert!(intersection.contains(&hash_10_11_12));

        Ok(())
    }
}

#[cfg(test)]
mod seeding {
    use super::record_fitness;
    use crate::models::{
        Crossover, Distribution, Fitness, FitnessGoal, Genotype, Mutagen, Request, Schedule,
        Selector,
    };
    use crate::repositories::genotypes::new_genotypes;
    use crate::repositories::requests::queries::new_request;
    use uuid::Uuid;

    pub(super) async fn seed(pool: &sqlx::PgPool) -> (Uuid, Uuid) {
        // Create requests first
        let request_1 = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1).unwrap(),
            Crossover::uniform(0.5).unwrap(),
            Distribution::latin_hypercube(200),
        )
        .unwrap();
        let request_2 = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(10, 20),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1).unwrap(),
            Crossover::uniform(0.5).unwrap(),
            Distribution::latin_hypercube(200),
        )
        .unwrap();

        let request_id_1 = request_1.id;
        let request_id_2 = request_2.id;

        new_request(pool, request_1).await.unwrap();
        new_request(pool, request_2).await.unwrap();

        let genotype_id_1 = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let genotype_id_2 = Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
        let genotype_id_3 = Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();
        let genotype_id_4 = Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap();
        let genotype_id_5 = Uuid::parse_str("00000000-0000-0000-0000-000000000005").unwrap();

        let genotypes = vec![
            Genotype {
                id: genotype_id_1,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![1, 2, 3],
                genome_hash: Genotype::compute_genome_hash(&[1, 2, 3]),
                request_id: request_id_1,
                generation_id: 1,
            },
            Genotype {
                id: genotype_id_2,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![4, 5, 6],
                genome_hash: Genotype::compute_genome_hash(&[4, 5, 6]),
                request_id: request_id_1,
                generation_id: 2,
            },
            Genotype {
                id: genotype_id_3,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![7, 8, 9],
                genome_hash: Genotype::compute_genome_hash(&[7, 8, 9]),
                request_id: request_id_2,
                generation_id: 1,
            },
            Genotype {
                id: genotype_id_4,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![10, 11, 12],
                genome_hash: Genotype::compute_genome_hash(&[10, 11, 12]),
                request_id: request_id_2,
                generation_id: 1,
            },
            Genotype {
                id: genotype_id_5,
                generated_at: chrono::Utc::now(),
                type_name: "test".to_string(),
                type_hash: 1,
                genome: vec![13, 14, 15],
                genome_hash: Genotype::compute_genome_hash(&[13, 14, 15]),
                request_id: request_id_2,
                generation_id: 2,
            },
        ];

        new_genotypes(pool, genotypes).await.unwrap();

        record_fitness(pool, &Fitness::new(genotype_id_1, 0.11))
            .await
            .unwrap();
        // genotype_id_2 has not fitness
        record_fitness(pool, &Fitness::new(genotype_id_3, 0.12))
            .await
            .unwrap();
        record_fitness(pool, &Fitness::new(genotype_id_4, 0.42))
            .await
            .unwrap();
        // genotype_id_5 has not fitness
        (request_id_1, request_id_2)
    }
}
