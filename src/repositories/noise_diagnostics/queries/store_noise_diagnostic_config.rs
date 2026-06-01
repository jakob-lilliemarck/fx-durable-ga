use super::super::Error;
use super::super::models::NoiseDiagnosticConfig;
use chrono::Utc;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx))]
pub async fn store_noise_diagnostic_config<'tx, E: PgExecutor<'tx>>(
    tx: E,
    optimization_type_name: &str,
    probe_population_size: i32,
    probe_min_evaluations: i32,
    probe_max_evaluations: i32,
    budget_id: &Uuid,
) -> Result<NoiseDiagnosticConfig, Error> {
    // We validate before the query, so we can return a rich error.
    let config = NoiseDiagnosticConfig::new(
        optimization_type_name,
        probe_population_size,
        probe_min_evaluations,
        probe_max_evaluations,
        budget_id,
    )?;

    let config = sqlx::query_as!(
        NoiseDiagnosticConfig,
        r#"
            INSERT INTO noise_diagnostic_configs (
                id,
                optimization_type_name,
                probe_population_size,
                probe_min_evaluations,
                probe_max_evaluations,
                budget_id,
                revised_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING
                id,
                optimization_type_name,
                probe_population_size,
                probe_min_evaluations,
                probe_max_evaluations,
                budget_id
        "#,
        config.id,
        config.optimization_type_name,
        config.probe_population_size,
        config.probe_min_evaluations,
        config.probe_max_evaluations,
        config.budget_id,
        Utc::now(),
    )
    .fetch_one(tx)
    .await?;

    Ok(config)
}

#[cfg(test)]
mod tests_new_noise_diagnostic_config {
    use crate::repositories::noise_diagnostics::queries::{
        get_noise_diagnostic_config, store_noise_diagnostic_config,
    };
    use sqlx::PgPool;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_creates_a_new_noise_diagnostic_config(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let config =
            store_noise_diagnostic_config(&pool, "test_optimization", 10, 100, 1000, &Uuid::nil())
                .await?;

        let persisted = get_noise_diagnostic_config(&pool, "test_optimization").await?;
        assert_eq!(
            config.optimization_type_name,
            persisted.optimization_type_name
        );
        assert_eq!(
            config.probe_population_size,
            persisted.probe_population_size
        );
        assert_eq!(
            config.probe_min_evaluations,
            persisted.probe_min_evaluations
        );
        assert_eq!(
            config.probe_max_evaluations,
            persisted.probe_max_evaluations
        );
        assert_eq!(config.budget_id, persisted.budget_id);

        Ok(())
    }
}
