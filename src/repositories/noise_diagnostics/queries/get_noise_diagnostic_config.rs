use super::super::Error;
use super::super::models::NoiseDiagnosticConfig;
use sqlx::PgExecutor;
use tracing::instrument;

#[instrument(level = "debug", skip(tx))]
pub async fn get_noise_diagnostic_config<'tx, E: PgExecutor<'tx>>(
    tx: E,
    optimization_type_name: &str,
) -> Result<NoiseDiagnosticConfig, Error> {
    let config = sqlx::query_as!(
        NoiseDiagnosticConfig,
        r#"
            SELECT
                id,
                optimization_type_name,
                probe_population_size,
                probe_min_evaluations,
                probe_max_evaluations,
                budget_id
            FROM
                noise_diagnostic_configs
            WHERE
                optimization_type_name = $1
            ORDER BY
                revised_at DESC
            LIMIT 1
        "#,
        optimization_type_name
    )
    .fetch_one(tx)
    .await
    .map_err(|err| match err {
        sqlx::Error::RowNotFound => Error::NotFound(optimization_type_name.to_string()),
        _ => Error::from(err),
    })?;

    Ok(config)
}

#[cfg(test)]
mod tests_get_noise_diagnostic_config {
    use crate::repositories::noise_diagnostics::Error;
    use crate::repositories::noise_diagnostics::queries::{
        get_noise_diagnostic_config, store_noise_diagnostic_config,
    };
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(pool).await?;

        store_noise_diagnostic_config(pool, "test_optimization", 10, 100, 1000, &Uuid::nil())
            .await?;

        store_noise_diagnostic_config(pool, "test_optimization", 20, 200, 2000, &Uuid::nil())
            .await?;

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_the_latest_noise_diagnostic_config(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let config = get_noise_diagnostic_config(&pool, "test_optimization").await?;

        assert_eq!(config.optimization_type_name, "test_optimization");
        assert_eq!(config.probe_population_size, 20);
        assert_eq!(config.probe_min_evaluations, 200);
        assert_eq!(config.probe_max_evaluations, 2000);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_not_found_when_no_config_exists(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let result = get_noise_diagnostic_config(&pool, "non_existent_optimization").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));

        Ok(())
    }
}
