use super::super::Error;
use chrono::Utc;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx))]
pub async fn store_noise_diagnostic_run<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: &Uuid,
    noise_diagnostic_config_id: &Uuid,
) -> Result<Uuid, Error> {
    let initiated_at = Utc::now();

    let run_id = sqlx::query_scalar!(
        r#"
            INSERT INTO noise_diagnostic_runs (
                id,
                noise_diagnostic_config_id,
                initiated_at
            )
            VALUES ($1, $2, $3)
            RETURNING
                id
        "#,
        id,
        noise_diagnostic_config_id,
        initiated_at,
    )
    .fetch_one(tx)
    .await?;

    Ok(run_id)
}

#[cfg(test)]
mod tests_new_noise_diagnostic_run {
    use super::store_noise_diagnostic_run;
    use crate::repositories::noise_diagnostics::models::NoiseDiagnosticConfig;
    use crate::repositories::noise_diagnostics::queries::store_noise_diagnostic_config;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<NoiseDiagnosticConfig> {
        crate::migrations::run_default_migrations(pool).await?;
        let config =
            store_noise_diagnostic_config(pool, "test_optimization", 10, 100, 1000, &Uuid::nil())
                .await?;
        Ok(config)
    }

    #[sqlx::test(migrations = false)]
    async fn it_creates_a_new_noise_diagnostic_run(pool: PgPool) -> anyhow::Result<()> {
        let config = seed(&pool).await?;

        let run_id = store_noise_diagnostic_run(&pool, &Uuid::now_v7(), &config.id).await?;

        assert!(!run_id.is_nil());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_when_noise_diagnostic_config_id_does_not_exist(
        pool: PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let non_existent_id = Uuid::now_v7();

        let result = store_noise_diagnostic_run(&pool, &Uuid::now_v7(), &non_existent_id).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::repositories::noise_diagnostics::Error::Database(_)
        ));

        Ok(())
    }
}
