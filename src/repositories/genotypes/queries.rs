use crate::models::{Evaluation, Genotype, Population, TimingsSummary};
use chrono::{DateTime, Utc};
use sqlx::{PgExecutor, Row};
use std::{fmt::Display, time::Duration};
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
            generation_id,
            parent_a,
            parent_b
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
            .push_bind(g.id())
            .push(", ")
            .push_bind(g.generated_at())
            .push(", ")
            .push_bind(g.type_name().to_string())
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
            .push(", ")
            .push_bind(g.parent_a)
            .push(", ")
            .push_bind(g.parent_b)
            .push(")");
    }
    query_builder.push(
        " RETURNING id, generated_at, type_name, type_hash, genome, genome_hash, request_id, generation_id, parent_a, parent_b",
    );

    let genotypes = query_builder
        .build_query_as::<Genotype>()
        .fetch_all(tx)
        .await?;

    Ok(genotypes)
}

#[cfg(test)]
mod search_filter_ordering_tests {
    use super::{SearchFilter, SearchResultsOrder, SortOrder};

    #[test]
    fn it_sets_completed_at_desc_order() {
        let filter = SearchFilter::default().with_order_completed_at_desc();
        assert!(matches!(
            filter.order,
            Some(SearchResultsOrder::CompletedAt(SortOrder::Desc))
        ));
    }

    #[test]
    fn it_sets_completed_at_asc_order() {
        let filter = SearchFilter::default().with_order_completed_at_asc();
        assert!(matches!(
            filter.order,
            Some(SearchResultsOrder::CompletedAt(SortOrder::Asc))
        ));
    }
}

#[cfg(test)]
mod new_genotypes_tests {
    use super::*;
    use crate::models::{FitnessGoal, Request, Schedule, Selector};
    use crate::repositories::requests::queries::new_request;
    use chrono::SubsecRound;

    #[sqlx::test(migrations = false)]
    async fn it_inserts_a_new_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotypes = vec![Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            request_id,
            1,
            None,
            None,
        )];
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

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotype = Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            request_id,
            1,
            None,
            None,
        );
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
    use crate::{
        models::{FitnessGoal, Request, Schedule, Selector},
        repositories::{
            genotypes::{new_genotypes, queries::check_if_generation_exists},
            requests::queries::new_request,
        },
    };

    use super::Genotype;

    #[sqlx::test(migrations = false)]
    async fn it_returns_true_when_generation_exists(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotype = Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            request_id,
            1,
            None,
            None,
        );

        new_genotypes(&pool, vec![genotype]).await?;

        let exists = check_if_generation_exists(&pool, request_id, 1).await?;
        assert!(exists);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_false_when_none_exist(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
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
    let genotype = sqlx::query!(
        r#"
            SELECT
                id,
                generated_at,
                type_name,
                type_hash,
                genome,
                genome_hash,
                request_id,
                generation_id,
                parent_a,
                parent_b
            FROM fx_durable_ga.genotypes
            WHERE id = $1;
        "#,
        id
    )
    .map(|row| Genotype {
        id: row.id,
        generated_at: row.generated_at,
        type_name: row.type_name,
        type_hash: row.type_hash,
        genome: row.genome,
        genome_hash: row.genome_hash,
        request_id: row.request_id,
        generation_id: row.generation_id,
        parent_a: row.parent_a,
        parent_b: row.parent_b,
    })
    .fetch_one(tx)
    .await?;

    Ok(genotype)
}

#[cfg(test)]
mod get_genotype_tests {
    use super::*;
    use crate::models::{FitnessGoal, Request, Schedule, Selector};
    use crate::repositories::requests::queries::new_request;

    #[sqlx::test(migrations = false)]
    async fn it_gets_an_existing_genotype(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotype = Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            request_id,
            1,
            None,
            None,
        );
        let genotype_id = genotype.id;

        new_genotypes(&pool, vec![genotype]).await?;

        let selected = get_genotype(&pool, &genotype_id).await?;

        assert_eq!(genotype_id, selected.id);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let non_existent_id = Uuid::now_v7();

        let result = get_genotype(&pool, &non_existent_id).await;

        assert!(result.is_err());
        Ok(())
    }
}

/// Records a fitness evaluation result for a genotype.
#[instrument(level = "debug", skip(tx), fields(evaluation = ?evaluation))]
pub(crate) async fn record_evaluation<'tx, E: PgExecutor<'tx>>(
    tx: E,
    evaluation: &Evaluation,
) -> Result<Evaluation, super::Error> {
    let row = sqlx::query!(
        r#"
            INSERT INTO fx_durable_ga.evaluations (genotype_id, fitness, started_at, completed_at, evaluated_by, copied_from)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING genotype_id, fitness, started_at, completed_at, evaluated_by, copied_from;
        "#,
        evaluation.genotype_id,
        evaluation.fitness,
        evaluation.started_at,
        evaluation.completed_at,
        evaluation.evaluated_by,
        evaluation.copied_from
    )
    .fetch_one(tx)
    .await?;
    Ok(Evaluation {
        genotype_id: row.genotype_id,
        fitness: row.fitness,
        started_at: row.started_at,
        completed_at: row.completed_at,
        evaluated_by: row.evaluated_by,
        copied_from: row.copied_from,
    })
}

#[instrument(level = "debug", skip(tx), fields(evaluations = ?evaluations))]
pub(crate) async fn record_evaluations<'tx, E: PgExecutor<'tx>>(
    tx: E,
    evaluations: &[Evaluation],
) -> Result<Vec<Evaluation>, super::Error> {
    if evaluations.is_empty() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.evaluations (
            genotype_id,
            fitness,
            started_at,
            completed_at,
            evaluated_by,
            copied_from
        ) VALUES ",
    );

    let mut first = true;
    for e in evaluations {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        query_builder
            .push("(")
            .push_bind(e.genotype_id)
            .push(", ")
            .push_bind(e.fitness)
            .push(", ")
            .push_bind(e.started_at)
            .push(", ")
            .push_bind(e.completed_at)
            .push(", ")
            .push_bind(e.evaluated_by)
            .push(", ")
            .push_bind(e.copied_from)
            .push(")");
    }

    query_builder.push(
        " RETURNING genotype_id, fitness, started_at, completed_at, evaluated_by, copied_from",
    );
    let rows = query_builder.build().fetch_all(tx).await?;
    let evaluations = rows
        .into_iter()
        .map(|row| Evaluation {
            genotype_id: row.get::<Uuid, _>("genotype_id"),
            fitness: row.get::<f64, _>("fitness"),
            started_at: row.get::<Option<DateTime<Utc>>, _>("started_at"),
            completed_at: row.get::<Option<DateTime<Utc>>, _>("completed_at"),
            evaluated_by: row.get::<Option<Uuid>, _>("evaluated_by"),
            copied_from: row.get::<Option<Uuid>, _>("copied_from"),
        })
        .collect();

    Ok(evaluations)
}

#[cfg(test)]
mod record_fitness_tests {
    use super::{record_evaluation, record_evaluations};
    use crate::models::{Evaluation, FitnessGoal, Genotype, Request, Schedule, Selector};
    use crate::repositories::genotypes::new_genotypes;
    use crate::repositories::requests::queries::new_request;
    use chrono::{SubsecRound, Utc};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_records_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let host_id = Uuid::now_v7();

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        // Create a genotype
        let genotype = Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            request_id,
            1,
            None,
            None,
        );
        let genotype_id = genotype.id;
        new_genotypes(&pool, vec![genotype]).await?;

        let fitness = Evaluation::new(
            genotype_id,
            0.543,
            Some(Utc::now()),
            Some(Utc::now()),
            Some(host_id.clone()),
        );
        let recorded = record_evaluation(&pool, &fitness).await?;

        assert_eq!(recorded.genotype_id, fitness.genotype_id);
        assert_eq!(recorded.fitness, fitness.fitness);
        assert_eq!(
            recorded.started_at,
            fitness.started_at.map(|ts| ts.trunc_subsecs(6))
        );
        assert_eq!(
            recorded.completed_at,
            fitness.completed_at.map(|ts| ts.trunc_subsecs(6))
        );
        assert_eq!(recorded.copied_from, None);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_clone = request.clone();

        let first = new_request(&pool, request).await;
        let second = new_request(&pool, request_clone).await;

        assert!(first.is_ok());
        assert!(second.is_err());
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_records_multiple_evaluations(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotypes = vec![
            Genotype::new(
                "test",
                1,
                serde_json::json!([1, 2, 3]),
                request_id,
                1,
                None,
                None,
            ),
            Genotype::new(
                "test",
                1,
                serde_json::json!([4, 5, 6]),
                request_id,
                1,
                None,
                None,
            ),
        ];
        let ids: Vec<Uuid> = genotypes.iter().map(|g| g.id()).collect();
        new_genotypes(&pool, genotypes).await?;

        let host_id = Uuid::now_v7();
        let evaluations = vec![
            Evaluation::new(
                ids[0],
                0.1,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id),
            ),
            Evaluation::new(ids[1], 0.2, Some(Utc::now()), Some(Utc::now()), None),
        ];

        let recorded = record_evaluations(&pool, &evaluations).await?;
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].fitness, evaluations[0].fitness);
        assert_eq!(recorded[1].fitness, evaluations[1].fitness);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_when_batch_contains_conflicts(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotypes = vec![Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            request_id,
            1,
            None,
            None,
        )];
        let genotype_id = genotypes[0].id();
        new_genotypes(&pool, genotypes).await?;

        let evaluations = vec![
            Evaluation::new(genotype_id, 0.1, Some(Utc::now()), Some(Utc::now()), None),
            Evaluation::new(genotype_id, 0.2, Some(Utc::now()), Some(Utc::now()), None),
        ];

        let result = record_evaluations(&pool, &evaluations).await;
        assert!(result.is_err());

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
    let population = sqlx::query_as!(
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
    Ok(population.unwrap_or(Population {
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
    use super::{get_population, record_evaluation};
    use crate::models::{
        Evaluation, FitnessGoal, Genotype, Population, Request, Schedule, Selector,
    };
    use crate::repositories::genotypes::new_genotypes;
    use crate::repositories::requests::queries::new_request;
    use chrono::Utc;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_a_population(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;
        let mut genotypes = Vec::with_capacity(3);
        for i in 1..=3 {
            genotypes.push(Genotype::new(
                "test",
                1,
                serde_json::json!([1, 2, 3]),
                request_id,
                i as i32,
                None,
                None,
            ));
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

    #[sqlx::test(migrations = false)]
    async fn it_handles_empty_population(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

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

    #[sqlx::test(migrations = false)]
    async fn it_gets_population_with_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // Create a request first
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(&pool, request).await?;

        let genotypes = vec![
            Genotype::new(
                "test",
                1,
                serde_json::json!([1, 2, 3]),
                request_id,
                1,
                None,
                None,
            ),
            Genotype::new(
                "test",
                1,
                serde_json::json!([4, 5, 6]),
                request_id,
                1,
                None,
                None,
            ),
            Genotype::new(
                "test",
                1,
                serde_json::json!([7, 8, 9]),
                request_id,
                1,
                None,
                None,
            ),
        ];

        let host_id = Uuid::now_v7();

        // Create genotypes
        let a_id = genotypes[0].id;
        let b_id = genotypes[1].id;
        let c_id = genotypes[2].id;

        new_genotypes(&pool, genotypes).await?;

        // Record fitness values
        record_evaluation(
            &pool,
            &Evaluation::new(
                a_id,
                0.50,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id.clone()),
            ),
        )
        .await?;
        record_evaluation(
            &pool,
            &Evaluation::new(
                b_id,
                0.99,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id.clone()),
            ),
        )
        .await?;
        record_evaluation(
            &pool,
            &Evaluation::new(
                c_id,
                0.01,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id.clone()),
            ),
        )
        .await?;

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
enum SearchResultsOrder {
    Random,
    Fitness(SortOrder),
    CompletedAt(SortOrder),
}

impl Display for SearchResultsOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Random => write!(f, "random"),
            Self::Fitness(order) => write!(f, "fitness_{}", order),
            Self::CompletedAt(order) => write!(f, "completed_at_{}", order),
        }
    }
}

/// Filter criteria for searching genotypes with various conditions.
#[derive(Debug)]
pub struct SearchFilter {
    request_id: Option<Uuid>,
    generation_id: Option<i32>,
    has_evaluation: Option<bool>,
    order: Option<SearchResultsOrder>,
}

impl Default for SearchFilter {
    fn default() -> Self {
        SearchFilter {
            request_id: None,
            generation_id: None,
            has_evaluation: None,
            order: None,
        }
    }
}

impl SearchFilter {
    /// Filters genotypes by request ID.
    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    #[allow(dead_code)]
    pub fn with_generation_id(mut self, generation_id: i32) -> Self {
        self.generation_id = Some(generation_id);
        self
    }

    /// Filters genotypes based on whether they have fitness evaluations.
    pub fn with_evaluation(mut self, has_evaluation: bool) -> Self {
        self.has_evaluation = Some(has_evaluation);
        self
    }

    pub fn with_order_random(mut self) -> Self {
        self.order = Some(SearchResultsOrder::Random);
        self
    }

    #[allow(dead_code)]
    pub fn with_order_fitness_asc(mut self) -> Self {
        self.order = Some(SearchResultsOrder::Fitness(SortOrder::Asc));
        self
    }

    #[allow(dead_code)]
    pub fn with_order_fitness_desc(mut self) -> Self {
        self.order = Some(SearchResultsOrder::Fitness(SortOrder::Desc));
        self
    }

    #[allow(dead_code)]
    pub fn with_order_completed_at_desc(mut self) -> Self {
        self.order = Some(SearchResultsOrder::CompletedAt(SortOrder::Desc));
        self
    }

    #[allow(dead_code)]
    pub fn with_order_completed_at_asc(mut self) -> Self {
        self.order = Some(SearchResultsOrder::CompletedAt(SortOrder::Asc));
        self
    }
}

/// Searches genotypes with optional filtering, ordering, and limits.
/// Returns genotypes paired with their fitness values (if available).
#[instrument(level = "debug", skip(tx), fields(filter = ?filter))]
pub(crate) async fn search<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchFilter,
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
                g.parent_a,
                g.parent_b,
                e.fitness "fitness!:Option<f64>"
            FROM fx_durable_ga.genotypes g
            LEFT JOIN fx_durable_ga.evaluations e ON g.id = e.genotype_id
            WHERE (
                $1::uuid IS NULL OR g.request_id = $1
            )
            AND (
                $2::int IS NULL OR g.generation_id = $2
            )
            AND (
                $3::bool IS NULL OR
                CASE
                    WHEN $3 = true THEN e.fitness IS NOT NULL
                    ELSE e.fitness IS NULL
                END
            )
            ORDER BY
                CASE
                    WHEN $4 = 'fitness_desc' THEN e.fitness
                    ELSE NULL
                END DESC NULLS LAST,
                CASE
                    WHEN $4 = 'fitness_asc' THEN e.fitness
                    ELSE NULL
                END ASC NULLS LAST,
                CASE
                    WHEN $4 = 'random' THEN RANDOM()
                    ELSE NULL
                END NULLS LAST,
                CASE
                    WHEN $4 = 'completed_at_desc' THEN e.completed_at
                    ELSE NULL
                END DESC NULLS LAST,
                CASE
                    WHEN $4 = 'completed_at_asc' THEN e.completed_at
                    ELSE NULL
                END ASC NULLS LAST,
                g.id ASC
            LIMIT $5;
        "#,
        filter.request_id,
        filter.generation_id,
        filter.has_evaluation,
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
                genome: serde_json::to_value(row.genome).expect("Expected genome"),
                genome_hash: row.genome_hash,
                request_id: row.request_id,
                generation_id: row.generation_id,
                parent_a: row.parent_a,
                parent_b: row.parent_b,
            };
            (genotype, row.fitness)
        })
        .collect();

    Ok(genotypes_with_fitness)
}

#[cfg(test)]
mod search_genotypes_tests {
    use super::{SearchFilter, search};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_genotypes_with_request_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (rid_1, _, gids) = super::seeding::seed(&pool).await;

        let found = search(&pool, &SearchFilter::default().with_request_id(rid_1), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();

        assert_eq!(vec![(gids[0], Some(0.11)), (gids[1], None)], actual);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_genotypes_with_generation_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (.., gids) = super::seeding::seed(&pool).await;

        let found = search(&pool, &SearchFilter::default().with_generation_id(2), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();

        assert_eq!(vec![(gids[1], None), (gids[4], None)], actual);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_genotypes_with_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (.., gids) = super::seeding::seed(&pool).await;

        let found = search(&pool, &SearchFilter::default().with_evaluation(true), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();

        assert_eq!(
            vec![
                (gids[0], Some(0.11)),
                (gids[2], Some(0.12)),
                (gids[3], Some(0.42))
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_genotypes_with_fitness_desc(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (.., gids) = super::seeding::seed(&pool).await;

        let found = search(
            &pool,
            &SearchFilter::default()
                .with_evaluation(true)
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
                (gids[3], Some(0.42)),
                (gids[2], Some(0.12)),
                (gids[0], Some(0.11)),
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_genotypes_with_fitness_asc(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (.., gids) = super::seeding::seed(&pool).await;

        let found = search(
            &pool,
            &SearchFilter::default()
                .with_evaluation(true)
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
                (gids[0], Some(0.11)),
                (gids[2], Some(0.12)),
                (gids[3], Some(0.42)),
            ],
            actual
        );

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_genotypes_without_fitness(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (.., gids) = super::seeding::seed(&pool).await;

        let found = search(&pool, &SearchFilter::default().with_evaluation(false), 5).await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();

        assert_eq!(vec![(gids[1], None), (gids[4], None)], actual);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_genotypes_with_request_id_fitness_and_random_order(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (rid_1, _, gids) = super::seeding::seed(&pool).await;

        let found = search(
            &pool,
            &SearchFilter::default()
                .with_request_id(rid_1)
                .with_evaluation(true)
                .with_order_random(),
            5,
        )
        .await?;

        let actual: Vec<(Uuid, Option<f64>)> = found
            .iter()
            .map(|(genotype, fitness)| (genotype.id, *fitness))
            .collect();

        assert_eq!(vec![(gids[0], Some(0.11)),], actual);

        Ok(())
    }
}

/// Finds which genome hashes already exist for a given request, along with their earliest evaluations.
#[instrument(level = "debug", skip(tx), fields(request_id=?request_id))]
pub(crate) async fn get_intersection<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
    hashes: &[i64],
) -> Result<Vec<(Genotype, Evaluation)>, super::Error> {
    if hashes.is_empty() {
        return Ok(vec![]);
    }

    let rows = sqlx::query!(
        r#"
            SELECT DISTINCT ON (g.genome_hash)
                g.id,
                g.generated_at,
                g.type_name,
                g.type_hash,
                g.genome,
                g.genome_hash,
                g.request_id,
                g.generation_id,
                g.parent_a,
                g.parent_b,
                e.genotype_id as "eval_genotype_id!",
                e.fitness as "fitness!",
                e.started_at as "started_at?",
                e.completed_at as "completed_at?",
                e.evaluated_by as "evaluated_by?",
                e.copied_from as "copied_from?"
            FROM fx_durable_ga.genotypes g
            JOIN fx_durable_ga.evaluations e ON g.id = e.genotype_id
            WHERE g.request_id = $1
              AND g.genome_hash = ANY($2)
            ORDER BY g.genome_hash, e.completed_at ASC NULLS LAST, g.id;
        "#,
        request_id,
        hashes,
    )
    .fetch_all(tx)
    .await?;

    let intersection = rows
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
                parent_a: row.parent_a,
                parent_b: row.parent_b,
            };

            let evaluation = Evaluation {
                genotype_id: row.eval_genotype_id,
                fitness: row.fitness,
                started_at: row.started_at,
                completed_at: row.completed_at,
                evaluated_by: row.evaluated_by,
                copied_from: row.copied_from,
            };

            (genotype, evaluation)
        })
        .collect();

    Ok(intersection)
}

#[cfg(test)]
mod tests {
    use crate::models::Genotype;

    #[sqlx::test(migrations = false)]
    async fn it_gets_the_intersection(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (rid_1, ..) = super::seeding::seed(&pool).await;

        // Test hashes for genomes in request_id_1: [1,2,3] and [4,5,6]
        let hash_1_2_3 = Genotype::compute_genome_hash(&[1, 2, 3]);
        let hash_4_5_6 = Genotype::compute_genome_hash(&[4, 5, 6]);
        let hash_nonexistent = Genotype::compute_genome_hash(&[99, 100, 101]);

        let candidate_hashes = vec![hash_1_2_3, hash_4_5_6, hash_nonexistent];

        let intersection = super::get_intersection(&pool, rid_1, &candidate_hashes).await?;

        // Should return only hashes that already have evaluations, which for request_id_1 is just hash_1_2_3
        assert_eq!(intersection.len(), 1);
        assert_eq!(intersection[0].0.genome_hash(), hash_1_2_3);
        assert!(intersection[0].1.fitness() > 0.0);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_vector_when_there_is_no_intersection(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (rid_1, ..) = super::seeding::seed(&pool).await;

        // Test with hashes that don't exist in the database
        let nonexistent_hashes = vec![
            Genotype::compute_genome_hash(&[99, 100, 101]),
            Genotype::compute_genome_hash(&[200, 201, 202]),
        ];

        let intersection = super::get_intersection(&pool, rid_1, &nonexistent_hashes).await?;

        assert!(intersection.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_isolates_intersection_by_request_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (rid_1, rid_2, ..) = super::seeding::seed(&pool).await;

        // Test with hashes from request_2: [7,8,9], [10,11,12], [13,14,15]
        let hash_7_8_9 = Genotype::compute_genome_hash(&[7, 8, 9]);
        let hash_10_11_12 = Genotype::compute_genome_hash(&[10, 11, 12]);

        let candidate_hashes = vec![hash_7_8_9, hash_10_11_12];

        // Query for request_1 should return empty (these hashes exist but in request_2)
        let intersection = super::get_intersection(&pool, rid_1, &candidate_hashes).await?;
        assert!(intersection.is_empty());

        // Query for request_2 should return both hashes
        let intersection = super::get_intersection(&pool, rid_2, &candidate_hashes).await?;

        let intersecting_hashes = intersection
            .iter()
            .map(|(g, _)| g.genome_hash())
            .collect::<Vec<i64>>();

        assert_eq!(intersection.len(), 2);
        assert!(intersecting_hashes.contains(&hash_7_8_9));
        assert!(intersecting_hashes.contains(&hash_10_11_12));

        Ok(())
    }
}

/// Get the anscestors of a genotype
#[instrument(level = "debug", skip(tx), fields(genotype_id = %genotype_id, degree=?degree))]
pub(crate) async fn get_ancestors<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotype_id: &Uuid,
    degree: i32, // Limits recursive search to a specified degree
) -> Result<Vec<(Genotype, Evaluation)>, super::Error> {
    let rows = sqlx::query!(
        r#"
            WITH RECURSIVE ancestor_tree AS (
                SELECT g.*, 0 AS degree
                FROM fx_durable_ga.genotypes g
                WHERE g.id = $1
                UNION
                SELECT parent.*, child.degree + 1
                FROM ancestor_tree child
                JOIN fx_durable_ga.genotypes parent
                ON parent.id = child.parent_a
                OR parent.id = child.parent_b
                WHERE child.degree + 1 <= $2::INTEGER
            )
            select *
            FROM ancestor_tree a
            JOIN evaluations e ON e.genotype_id = a.id
            ORDER BY a.degree, a.id ASC;
        "#,
        genotype_id,
        degree,
    )
    .fetch_all(tx)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| {
            // These fields are options due to the CTE, however they're not nullable in the schema
            // expect() is used to get a better error message in case of panic than sqlx would provide
            let genotype = Genotype {
                id: row.id.expect("Expected id"),
                generated_at: row.generated_at.expect("Expected generated_at"),
                type_name: row.type_name.expect("Expected type_name"),
                type_hash: row.type_hash.expect("Expected type_hash"),
                genome: row.genome.expect("Expected genome"),
                genome_hash: row.genome_hash.expect("Expected genome_hash"),
                request_id: row.request_id.expect("Expected request_id"),
                generation_id: row.generation_id.expect("Expected generation_id"),
                parent_a: row.parent_a,
                parent_b: row.parent_b,
            };

            let evaluation = Evaluation {
                genotype_id: row.genotype_id,
                fitness: row.fitness,
                started_at: row.started_at,
                completed_at: row.completed_at,
                evaluated_by: row.evaluated_by,
                copied_from: row.copied_from,
            };

            (genotype, evaluation)
        })
        .collect())
}

#[cfg(test)]
mod get_ancestors_tests {
    use super::get_ancestors;
    use uuid::Uuid;
    #[sqlx::test(migrations = false)]
    async fn it_returns_each_ancestor(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = super::seeding::seed_lineage(&pool).await;
        // indexes: [0] root, [1] child, [2] grandchild
        let ancestors = get_ancestors(&pool, &lineage[2], 5).await?;
        let ids: Vec<Uuid> = ancestors
            .iter()
            .map(|(genotype, _)| genotype.id())
            .collect();

        assert_eq!(ids, vec![lineage[2], lineage[1], lineage[0]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_is_bounded_by_degree(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = super::seeding::seed_lineage(&pool).await;
        // indexes: [0] root, [1] child, [2] grandchild
        let ancestors = get_ancestors(&pool, &lineage[2], 1).await?;
        let ids: Vec<Uuid> = ancestors
            .iter()
            .map(|(genotype, _)| genotype.id())
            .collect();

        assert_eq!(ids, vec![lineage[2], lineage[1]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_no_ancestors_when_there_are_none(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = super::seeding::seed_lineage(&pool).await;
        // indexes: [0] root, [1] child, [2] grandchild
        let ancestors = get_ancestors(&pool, &lineage[0], 5).await?;
        let ids: Vec<Uuid> = ancestors
            .iter()
            .map(|(genotype, _)| genotype.id())
            .collect();

        assert_eq!(ids, vec![lineage[0]]);
        Ok(())
    }
}

/// Get the descendants of a genotype
#[instrument(level = "debug", skip(tx), fields(genotype_id = %genotype_id, degree=?degree))]
pub(crate) async fn get_descendants<'tx, E: PgExecutor<'tx>>(
    tx: E,
    genotype_id: &Uuid,
    degree: i32, // Limits recursive search to a specified degree
) -> Result<Vec<(Genotype, Evaluation)>, super::Error> {
    let rows = sqlx::query!(
        r#"
            WITH RECURSIVE descendant_tree AS (
                SELECT g.*, 0 AS degree
                FROM fx_durable_ga.genotypes g
                WHERE g.id = $1
                UNION
                SELECT child.*, parent.degree + 1
                FROM descendant_tree parent
                JOIN fx_durable_ga.genotypes child
                ON child.parent_a = parent.id
                OR child.parent_b = parent.id
                WHERE parent.degree + 1 <= $2::INTEGER
            )
            SELECT *
            FROM descendant_tree d
            JOIN fx_durable_ga.evaluations e ON e.genotype_id = d.id
            ORDER BY d.degree, d.id ASC
        "#,
        genotype_id,
        degree,
    )
    .fetch_all(tx)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| {
            // These fields are options due to the CTE, however they're not nullable in the schema
            // expect() is used to get a better error message in case of panic than sqlx would provide
            let genotype = Genotype {
                id: row.id.expect("Expected id"),
                generated_at: row.generated_at.expect("Expected generated_at"),
                type_name: row.type_name.expect("Expected type_name"),
                type_hash: row.type_hash.expect("Expected type_hash"),
                genome: row.genome.expect("Expected genome"),
                genome_hash: row.genome_hash.expect("Expected genome_hash"),
                request_id: row.request_id.expect("Expected request_id"),
                generation_id: row.generation_id.expect("Expected generation_id"),
                parent_a: row.parent_a,
                parent_b: row.parent_b,
            };

            let evaluation = Evaluation {
                genotype_id: row.genotype_id,
                fitness: row.fitness,
                started_at: row.started_at,
                completed_at: row.completed_at,
                evaluated_by: row.evaluated_by,
                copied_from: row.copied_from,
            };

            (genotype, evaluation)
        })
        .collect())
}

#[cfg(test)]
mod get_descendants_tests {
    use super::get_descendants;
    use uuid::Uuid;
    #[sqlx::test(migrations = false)]
    async fn it_returns_each_descendants(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = super::seeding::seed_lineage(&pool).await;
        // indexes: [0] root, [1] child, [2] grandchild
        let descendants = get_descendants(&pool, &lineage[0], 5).await?;
        let ids: Vec<Uuid> = descendants
            .iter()
            .map(|(genotype, _)| genotype.id())
            .collect();

        assert_eq!(ids, vec![lineage[0], lineage[1], lineage[2]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_is_bounded_by_degree(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = super::seeding::seed_lineage(&pool).await;
        // indexes: [0] root, [1] child, [2] grandchild
        let descendants = get_descendants(&pool, &lineage[0], 1).await?;
        let ids: Vec<Uuid> = descendants
            .iter()
            .map(|(genotype, _)| genotype.id())
            .collect();

        assert_eq!(ids, vec![lineage[0], lineage[1]]);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_no_descendant_when_there_are_none(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let lineage = super::seeding::seed_lineage(&pool).await;
        // indexes: [0] root, [1] child, [2] grandchild
        let descendants = get_descendants(&pool, &lineage[2], 5).await?;
        let ids: Vec<Uuid> = descendants
            .iter()
            .map(|(genotype, _)| genotype.id())
            .collect();

        assert_eq!(ids, vec![lineage[2]]);
        Ok(())
    }
}

#[derive(Debug, Hash)]
pub struct GetTimingsFilter<'a> {
    ids: Option<&'a [Uuid]>,
    type_name: Option<String>,
    type_hash: Option<i32>,
    request_id: Option<Uuid>,
    generation_id: Option<i32>,
    started_within: Option<(DateTime<Utc>, DateTime<Utc>)>,
    completed_within: Option<(DateTime<Utc>, DateTime<Utc>)>,
    evaluated_by: Option<Uuid>,
}

impl<'a> Default for GetTimingsFilter<'a> {
    fn default() -> Self {
        Self {
            ids: None,
            type_name: None,
            type_hash: None,
            request_id: None,
            generation_id: None,
            started_within: None,
            completed_within: None,
            evaluated_by: None,
        }
    }
}

impl<'a> GetTimingsFilter<'a> {
    pub fn with_ids(mut self, ids: &'a [Uuid]) -> Self {
        self.ids = Some(ids);
        self
    }

    pub fn with_type_name(mut self, type_name: String) -> Self {
        self.type_name = Some(type_name);
        self
    }

    pub fn with_type_hash(mut self, type_hash: i32) -> Self {
        self.type_hash = Some(type_hash);
        self
    }

    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_generation_id(mut self, generation_id: i32) -> Self {
        self.generation_id = Some(generation_id);
        self
    }

    pub fn with_started_within(mut self, since: DateTime<Utc>, until: DateTime<Utc>) -> Self {
        self.started_within = Some((since, until));
        self
    }

    pub fn with_completed_within(mut self, since: DateTime<Utc>, until: DateTime<Utc>) -> Self {
        self.completed_within = Some((since, until));
        self
    }

    pub fn with_evaluated_by(mut self, evaluated_by: Uuid) -> Self {
        self.evaluated_by = Some(evaluated_by);
        self
    }
}

struct DbTimingsSummary {
    total: i64,
    min_duration_micros: i64,
    max_duration_micros: i64,
    avg_duration_micros: i64,
    percentile_durations: Option<Vec<i64>>,
}

impl From<DbTimingsSummary> for TimingsSummary {
    fn from(value: DbTimingsSummary) -> Self {
        TimingsSummary {
            records: value.total,
            min: Duration::from_micros(value.min_duration_micros as u64),
            max: Duration::from_micros(value.max_duration_micros as u64),
            avg: Duration::from_micros(value.avg_duration_micros as u64),
            percentiles: value.percentile_durations.map_or(vec![], |values| {
                values
                    .iter()
                    .map(|micros| Duration::from_micros(*micros as u64))
                    .collect()
            }),
        }
    }
}

/// Get evaluation timing summary information
#[instrument(level = "debug", skip(tx), fields())]
pub(crate) async fn get_timings<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &GetTimingsFilter<'tx>,
    percentiles: &[f64],
) -> Result<TimingsSummary, super::Error> {
    let db_timings = sqlx::query_as!(
        DbTimingsSummary,
        r#"
        SELECT
            COUNT(*) AS "total!",
            MIN(duration_micros) AS "min_duration_micros!",
            MAX(duration_micros) AS "max_duration_micros!",
            AVG(duration_micros)::bigint AS "avg_duration_micros!",
            ARRAY(
                SELECT ROUND(val)::bigint
                FROM UNNEST(
                    percentile_cont($11::double PRECISION[])
                    WITHIN GROUP (ORDER BY duration_micros)
                ) AS val
            ) AS "percentile_durations"
        FROM
            (
            SELECT DISTINCT ON (g.genome_hash)
                e.genotype_id,
                (EXTRACT(EPOCH FROM e.completed_at - e.started_at) * 1e6)::bigint AS duration_micros
            FROM
                fx_durable_ga.genotypes g
            JOIN fx_durable_ga.evaluations e ON
                e.genotype_id = g.id
            WHERE
                ($1::uuid[] IS NULL
                    OR g.id = ANY($1::uuid[]))
                AND ($2::text IS NULL
                    OR g.type_name = $2::text)
                AND ($3::integer IS NULL
                    OR g.type_hash = $3::integer)
                AND ($4::uuid IS NULL
                    OR g.request_id = $4::uuid)
                AND ($5::integer IS NULL
                    OR g.generation_id = $5::integer)
                AND (
                    ($6::timestamptz IS NULL
                    OR $7::timestamptz IS NULL)
                OR e.started_at BETWEEN $6::timestamptz AND $7::timestamptz
              )
                AND (
                    ($8::timestamptz IS NULL
                    OR $9::timestamptz IS NULL)
                OR e.completed_at BETWEEN $8::timestamptz AND $9::timestamptz
              )
                AND ($10::uuid IS NULL
                    OR e.evaluated_by = $10::uuid)
            ORDER BY g.genome_hash, e.completed_at ASC NULLS LAST, g.id
        ) timed;
		"#,
        filter.ids,
        filter.type_name,
        filter.type_hash,
        filter.request_id,
        filter.generation_id,
        filter.started_within.map(|(since, ..)| since),
        filter.started_within.map(|(until, ..)| until),
        filter.completed_within.map(|(since, ..)| since),
        filter.started_within.map(|(until, ..)| until),
        filter.evaluated_by,
        percentiles
    )
    .fetch_one(tx)
    .await?;

    Ok(db_timings.into())
}

#[cfg(test)]
mod get_timings_tests {
    use super::{GetTimingsFilter, get_timings};
    use crate::models::{Evaluation, FitnessGoal, Genotype, Request, Schedule, Selector};
    use crate::repositories::{genotypes::new_genotypes, requests::queries::new_request};
    use chrono::{Duration as ChronoDuration, TimeZone, Utc};
    use std::time::Duration;
    use uuid::Uuid;

    async fn seed_timed_genotypes(pool: &sqlx::PgPool) -> anyhow::Result<(Uuid, Vec<Uuid>)> {
        let request = Request::new(
            "timings",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )?;
        let request_id = request.id;
        new_request(pool, request).await?;

        let genotypes = vec![
            Genotype::new(
                "timings",
                1,
                serde_json::json!([1]),
                request_id,
                1,
                None,
                None,
            ),
            Genotype::new(
                "timings",
                1,
                serde_json::json!([2]),
                request_id,
                1,
                None,
                None,
            ),
            Genotype::new(
                "timings",
                1,
                serde_json::json!([3]),
                request_id,
                1,
                None,
                None,
            ),
        ];
        let ids: Vec<Uuid> = genotypes.iter().map(|g| g.id).collect();
        new_genotypes(pool, genotypes).await?;

        let host = Uuid::now_v7();
        let base = Utc
            .with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
            .single()
            .expect("valid timestamp");
        let durations = [100_000i64, 200_000, 300_000]; // microseconds

        for (idx, genotype_id) in ids.iter().enumerate() {
            let started_at = base + ChronoDuration::seconds(idx as i64);
            let completed_at = started_at + ChronoDuration::microseconds(durations[idx]);
            super::record_evaluation(
                pool,
                &Evaluation::new(
                    *genotype_id,
                    0.5,
                    Some(started_at),
                    Some(completed_at),
                    Some(host),
                ),
            )
            .await?;
        }

        Ok((request_id, ids))
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_timing_data_with_percentiles(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (request_id, _) = seed_timed_genotypes(&pool).await?;

        let summary = get_timings(
            &pool,
            &GetTimingsFilter::default().with_request_id(request_id),
            &[0.5, 0.9],
        )
        .await?;

        assert_eq!(summary.records(), 3);
        assert_eq!(summary.min(), &Duration::from_micros(100_000));
        assert_eq!(summary.max(), &Duration::from_micros(300_000));
        assert_eq!(summary.avg(), &Duration::from_micros(200_000));
        assert_eq!(summary.percentiles().len(), 2);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_timing_data_without_percentiles(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (request_id, _) = seed_timed_genotypes(&pool).await?;

        let summary = get_timings(
            &pool,
            &GetTimingsFilter::default().with_request_id(request_id),
            &[],
        )
        .await?;

        assert_eq!(summary.records(), 3);
        assert!(summary.percentiles().is_empty());
        Ok(())
    }
}

#[cfg(test)]
mod seeding {
    use super::record_evaluation;
    use crate::models::{Evaluation, FitnessGoal, Genotype, Request, Schedule, Selector};
    use crate::repositories::genotypes::new_genotypes;
    use crate::repositories::requests::queries::new_request;
    use chrono::Utc;
    use uuid::Uuid;

    pub(super) async fn seed(pool: &sqlx::PgPool) -> (Uuid, Uuid, [Uuid; 5]) {
        // Create requests first
        let request_1 = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )
        .unwrap();
        let request_2 = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )
        .unwrap();

        let rid_1 = request_1.id;
        let rid_2 = request_2.id;

        new_request(pool, request_1).await.unwrap();
        new_request(pool, request_2).await.unwrap();

        let genotypes = vec![
            Genotype::new(
                "test",
                1,
                serde_json::json!([1, 2, 3]),
                rid_1,
                1,
                None,
                None,
            ),
            Genotype::new(
                "test",
                1,
                serde_json::json!([4, 5, 6]),
                rid_1,
                2,
                None,
                None,
            ),
            Genotype::new(
                "test",
                1,
                serde_json::json!([7, 8, 9]),
                rid_2,
                1,
                None,
                None,
            ),
            Genotype::new(
                "test",
                1,
                serde_json::json!([10, 11, 12]),
                rid_2,
                1,
                None,
                None,
            ),
            Genotype::new(
                "test",
                1,
                serde_json::json!([13, 14, 15]),
                rid_2,
                2,
                None,
                None,
            ),
        ];

        let gid_1 = genotypes[0].id;
        let gid_2 = genotypes[1].id;
        let gid_3 = genotypes[2].id;
        let gid_4 = genotypes[3].id;
        let gid_5 = genotypes[4].id;

        new_genotypes(pool, genotypes).await.unwrap();

        let host_id = Uuid::now_v7();

        // FIXME:
        // Use the new batch insert method record_evaluations instead!
        record_evaluation(
            pool,
            &Evaluation::new(
                gid_1,
                0.11,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id.clone()),
            ),
        )
        .await
        .unwrap();
        // genotype_id_2 has not fitness
        record_evaluation(
            pool,
            &Evaluation::new(
                gid_3,
                0.12,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id.clone()),
            ),
        )
        .await
        .unwrap();
        record_evaluation(
            pool,
            &Evaluation::new(
                gid_4,
                0.42,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id.clone()),
            ),
        )
        .await
        .unwrap();
        // genotype_id_5 has not fitness
        (rid_1, rid_2, [gid_1, gid_2, gid_3, gid_4, gid_5])
    }

    pub(super) async fn seed_lineage(pool: &sqlx::PgPool) -> Vec<Uuid> {
        let request = Request::new(
            "lineage",
            1,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(10),
            Schedule::generational(100, 10),
            serde_json::json!({ "Uniform": { "probability": 0.5 } }),
            None::<()>,
        )
        .unwrap();
        let request_id = request.id;
        new_request(pool, request).await.unwrap();

        let root = Genotype::new(
            "lineage",
            1,
            serde_json::json!([0]),
            request_id,
            1,
            None,
            None,
        );
        let root_id = root.id();

        let child = Genotype::new(
            "lineage",
            1,
            serde_json::json!([1]),
            request_id,
            2,
            Some(&root_id),
            None,
        );
        let child_id = child.id();

        let grandchild = Genotype::new(
            "lineage",
            1,
            serde_json::json!([2]),
            request_id,
            3,
            Some(&child_id),
            None,
        );
        let grandchild_id = grandchild.id();

        new_genotypes(pool, vec![root, child, grandchild])
            .await
            .unwrap();

        let host_id = Uuid::now_v7();
        for genotype_id in [root_id, child_id, grandchild_id] {
            record_evaluation(
                pool,
                &Evaluation::new(
                    genotype_id,
                    0.5,
                    Some(Utc::now()),
                    Some(Utc::now()),
                    Some(host_id),
                ),
            )
            .await
            .unwrap();
        }

        // indexes: [0] root, [1] child, [2] grandchild
        vec![root_id, child_id, grandchild_id]
    }
}
