use super::super::Error as RepositoryError;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

/// Checks if any genotypes exist for the given request and generation.
#[instrument(level = "debug", skip(tx))]
pub async fn check_if_generation_exists<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: &Uuid,
    generation_id: i32,
) -> Result<bool, RepositoryError> {
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
mod tests_generation_exists {
    use super::check_if_generation_exists;
    use crate::repositories::genotypes::{Genotype, store_genotypes};
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};

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
        store_request(&pool, request).await?;

        let genotype = Genotype::new(
            "test",
            1,
            serde_json::json!([1, 2, 3]),
            Some(request_id),
            Some(1),
            None,
            None,
        )?;

        store_genotypes(&pool, &[genotype]).await?;

        let exists = check_if_generation_exists(&pool, &request_id, 1).await?;
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
        store_request(&pool, request).await?;

        let exists = check_if_generation_exists(&pool, &request_id, 1).await?;
        assert!(!exists);
        Ok(())
    }
}
