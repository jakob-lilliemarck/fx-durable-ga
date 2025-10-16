use crate::models::{Fitness, Individual, Population};
use sqlx::PgExecutor;
use std::fmt::Display;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx), fields(individuals_count = individuals.len()))]
pub(crate) async fn add_to_population<'tx, E: PgExecutor<'tx>>(
    tx: E,
    individuals: &[Individual],
) -> Result<Vec<Individual>, super::Error> {
    if individuals.is_empty() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.individuals (genotype_id, request_id, generation_id) ",
    );

    query_builder.push_values(individuals, |mut b, individual| {
        b.push_bind(individual.genotype_id)
            .push_bind(individual.request_id)
            .push_bind(individual.generation_id);
    });

    query_builder.push(" RETURNING genotype_id, request_id, generation_id");

    let inserted: Vec<Individual> = query_builder
        .build_query_as::<Individual>()
        .fetch_all(tx)
        .await?;

    Ok(inserted)
}

#[cfg(test)]
mod add_to_population_tests {
    use super::add_to_population;
    use crate::models::Individual;
    use uuid::Uuid;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_adds_to_the_population(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();
        let mut individuals = Vec::with_capacity(3);
        for _ in 0..2 {
            individuals.push(Individual::new(Uuid::now_v7(), request_id, 1));
        }

        let inserted = add_to_population(&pool, &individuals).await?;

        assert_eq!(individuals, inserted);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_handles_empty_individuals(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let inserted = add_to_population(&pool, &[]).await?;
        assert_eq!(inserted.len(), 0);
        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_duplicate_genotype_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();
        let genotype_id = Uuid::now_v7();

        // First insert should succeed
        let result = add_to_population(
            &pool,
            &[
                Individual::new(genotype_id, request_id, 1),
                Individual::new(genotype_id, request_id, 1),
            ],
        )
        .await;

        assert!(result.is_err());
        Ok(())
    }
}

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
        RETURNING genotype_id, fitness, evaluated_at
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
    use crate::{
        models::{Fitness, Individual},
        repositories::populations::queries::add_to_population,
    };
    use chrono::SubsecRound;
    use uuid::Uuid;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_records_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();
        let genotype_id = Uuid::now_v7();

        add_to_population(&pool, &[Individual::new(genotype_id, request_id, 1)]).await?;

        let fitness = Fitness::new(genotype_id, 0.543);
        let recorded = record_fitness(&pool, &fitness).await?;

        assert_eq!(recorded.genotype_id, fitness.genotype_id);
        assert_eq!(recorded.fitness, fitness.fitness);
        assert_eq!(recorded.evaluated_at, fitness.evaluated_at.trunc_subsecs(6));
        Ok(())
    }
}

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
            evaluated_individuals "evaluated_individuals!:i64",
            live_individuals "live_individuals!:i64",
            current_generation "current_generation!:i32",
            best_fitness "best_fitness"
        FROM fx_durable_ga.populations
        WHERE request_id = $1
        "#,
        request_id
    )
    .fetch_optional(tx)
    .await?;

    // Handle case where request has no individuals yet
    Ok(state.unwrap_or(Population {
        request_id: *request_id,
        evaluated_individuals: 0,
        live_individuals: 0,
        current_generation: 0,
        best_fitness: None,
    }))
}

#[cfg(test)]
mod get_population_tests {
    use super::{get_population, record_fitness};
    use crate::{
        models::{Fitness, Individual, Population},
        repositories::populations::queries::add_to_population,
    };
    use uuid::Uuid;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_a_population(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();
        let mut individuals = Vec::with_capacity(3);
        for i in 1..=3 {
            individuals.push(Individual::new(Uuid::now_v7(), request_id, i as i32));
        }
        add_to_population(&pool, &individuals).await?;

        let population = get_population(&pool, &request_id).await?;
        assert_eq!(
            population,
            Population {
                request_id,
                evaluated_individuals: 0,
                live_individuals: 3,
                current_generation: 3,
                best_fitness: None
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
                evaluated_individuals: 0,
                live_individuals: 0,
                current_generation: 0,
                best_fitness: None
            }
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_population_with_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let request_id = Uuid::now_v7();

        // Add individual
        let a = Individual::new(Uuid::now_v7(), request_id, 1);
        let b = Individual::new(Uuid::now_v7(), request_id, 1);
        let c = Individual::new(Uuid::now_v7(), request_id, 1);

        let a_fitness = Fitness::new(a.genotype_id, 0.50);
        let b_fitness = Fitness::new(b.genotype_id, 0.99);
        let c_fitness = Fitness::new(c.genotype_id, 0.01);

        add_to_population(&pool, &[a, b, c]).await?;

        // Record fitness values
        record_fitness(&pool, &a_fitness).await?;
        record_fitness(&pool, &b_fitness).await?;
        record_fitness(&pool, &c_fitness).await?;

        let population = get_population(&pool, &request_id).await?;

        assert_eq!(
            population,
            Population {
                request_id,
                evaluated_individuals: 3,
                live_individuals: 0,
                current_generation: 1,
                best_fitness: Some(0.99)
            }
        );

        Ok(())
    }
}

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
    pub(crate) fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn with_generation_id(mut self, generation_id: i32) -> Self {
        self.generation_id = Some(generation_id);
        self
    }

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

#[instrument(level = "debug", skip(tx), fields(filter = ?filter))]
pub(crate) async fn search_individuals<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &Filter,
    limit: i64,
) -> Result<Vec<(Uuid, Option<f64>)>, super::Error> {
    let rows = sqlx::query!(
        r#"
        SELECT
            genotype_id "genotype_id!:Uuid",
            fitness
        FROM fx_durable_ga.individuals_with_fitness
        WHERE (
            $1::uuid IS NULL OR request_id = $1
        )
        AND (
            $2::int IS NULL OR generation_id = $2
        )
        AND (
            $3::bool IS NULL OR
            CASE
                WHEN $3 = true THEN fitness IS NOT NULL
                ELSE fitness IS NULL
            END
        )
        ORDER BY
            CASE
                WHEN $4 = 'fitness_desc' THEN fitness
                ELSE NULL
            END DESC NULLS LAST,
            CASE
                WHEN $4 = 'fitness_asc' THEN fitness
                ELSE NULL
            END ASC NULLS LAST,
            CASE
                WHEN $4 = 'random' THEN RANDOM()
                ELSE NULL
            END NULLS LAST,
            genotype_id ASC
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

    let individuals = rows
        .into_iter()
        .map(|row| (row.genotype_id, row.fitness))
        .collect();

    Ok(individuals)
}

#[cfg(test)]
mod search_individuals_tests {
    use super::{Filter, add_to_population, record_fitness, search_individuals};
    use crate::models::{Fitness, Individual};
    use uuid::Uuid;

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<(Uuid, Uuid)> {
        let request_id_1 = Uuid::now_v7();
        let request_id_2 = Uuid::now_v7();

        let genotype_id_1 = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let genotype_id_2 = Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
        let genotype_id_3 = Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();
        let genotype_id_4 = Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap();
        let genotype_id_5 = Uuid::parse_str("00000000-0000-0000-0000-000000000005").unwrap();

        let individuals = vec![
            Individual::new(genotype_id_1, request_id_1, 1),
            Individual::new(genotype_id_2, request_id_1, 2),
            Individual::new(genotype_id_3, request_id_2, 1),
            Individual::new(genotype_id_4, request_id_2, 1),
            Individual::new(genotype_id_5, request_id_2, 2),
        ];

        add_to_population(pool, &individuals).await?;

        record_fitness(pool, &Fitness::new(genotype_id_1, 0.11)).await?;
        // genotype_id_2 has not fitness
        record_fitness(pool, &Fitness::new(genotype_id_3, 0.12)).await?;
        record_fitness(pool, &Fitness::new(genotype_id_4, 0.42)).await?;
        // genotype_id_5 has not fitness
        Ok((request_id_1, request_id_2))
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_individuals_with_request_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let (request_id_1, _) = seed(&pool).await?;

        let found =
            search_individuals(&pool, &Filter::default().with_request_id(request_id_1), 5).await?;

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
            found
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_individuals_with_generation_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let found = search_individuals(&pool, &Filter::default().with_generation_id(2), 5).await?;

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
            found
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_individuals_with_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let found = search_individuals(&pool, &Filter::default().with_fitness(true), 5).await?;

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
            found
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_individuals_with_fitness_desc(pool: sqlx::PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let found = search_individuals(
            &pool,
            &Filter::default()
                .with_fitness(true)
                .with_order_fitness_desc(),
            5,
        )
        .await?;

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
            found
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_individuals_with_fitness_asc(pool: sqlx::PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let found = search_individuals(
            &pool,
            &Filter::default()
                .with_fitness(true)
                .with_order_fitness_asc(),
            5,
        )
        .await?;

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
            found
        );

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_individuals_without_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let found = search_individuals(&pool, &Filter::default().with_fitness(false), 5).await?;

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
            found
        );

        Ok(())
    }
}
