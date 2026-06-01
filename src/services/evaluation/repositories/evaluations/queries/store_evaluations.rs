use super::super::Error as RepositoryError;
use super::super::Evaluation;
use sqlx::{PgExecutor, Row};
use tracing::instrument;

/// Stores multiple evaluations in a single batch insert
#[instrument(level = "debug", skip(tx, evaluations))]
pub(crate) async fn store_evaluations<'tx, 'a, I, E>(
    tx: E,
    evaluations: I,
) -> Result<Vec<Evaluation>, RepositoryError>
where
    E: PgExecutor<'tx>,
    I: IntoIterator<Item = &'a Evaluation>,
{
    let mut evaluations = evaluations.into_iter().peekable();
    if evaluations.peek().is_none() {
        return Ok(vec![]);
    }

    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO evaluation.evaluations (
            id,
            genotype_id,
            fitness,
            started_at,
            completed_at,
            evaluated_by,
            request_id,
            generated_at
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
            .push_bind(e.id)
            .push(", ")
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
            .push_bind(e.request_id)
            .push(", ")
            .push_bind(e.generated_at)
            .push(")");
    }

    query_builder.push(" RETURNING id, genotype_id, fitness, started_at, completed_at, evaluated_by, request_id, generated_at");

    let rows = query_builder.build().fetch_all(tx).await?;

    let evaluations = rows
        .iter()
        .map(|row| Evaluation {
            id: row.get("id"),
            genotype_id: row.get("genotype_id"),
            fitness: row.get("fitness"),
            started_at: row.get("started_at"),
            completed_at: row.get("completed_at"),
            evaluated_by: row.get("evaluated_by"),
            request_id: row.get("request_id"),
            generated_at: row.get("generated_at"),
        })
        .collect();

    Ok(evaluations)
}

#[cfg(test)]
mod tests_store_evaluations {
    use super::store_evaluations;
    use crate::services::evaluation::repositories::evaluations::Evaluation;
    use crate::repositories::genotypes::Genotype;
    use crate::repositories::genotypes::store_genotypes;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use chrono::Utc;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: PgPool) -> anyhow::Result<Vec<Uuid>> {
        let request = Request::new(
            "test",
            1,
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request = store_request(&pool, request).await?;

        let genotypes = vec![
            Genotype::new(
                "test",
                1,
                serde_json::json!([1, 2, 3]),
                Some(request.id),
                Some(1),
                None,
                None,
            )?,
            Genotype::new(
                "test",
                1,
                serde_json::json!([4, 5, 6]),
                Some(request.id),
                Some(1),
                None,
                None,
            )?,
        ];
        let genotypes = store_genotypes(&pool, &genotypes).await?;
        let genotype_ids = genotypes.iter().map(|g| g.id()).collect::<Vec<Uuid>>();
        Ok(genotype_ids)
    }

    #[sqlx::test(migrations = false)]
    async fn it_stores_evaluations(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let host_id = Uuid::now_v7();
        let genotype_ids = seed(pool.clone()).await?;
        let evaluations = vec![
            Evaluation::new(
                genotype_ids[0],
                0.1,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id),
            ),
            Evaluation::new(
                genotype_ids[1],
                0.2,
                Some(Utc::now()),
                Some(Utc::now()),
                None,
            ),
        ];

        let evaluations = store_evaluations(&pool, &evaluations).await?;
        assert_eq!(evaluations.len(), 2);
        assert_eq!(evaluations[0].fitness, evaluations[0].fitness);
        assert_eq!(evaluations[1].fitness, evaluations[1].fitness);

        Ok(())
    }
}
