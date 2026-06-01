use super::super::Error;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx))]
pub async fn get_noise_diagnostic_runs<'tx, E: PgExecutor<'tx>>(
    tx: E,
    noise_diagnostic_config_id: &Uuid,
) -> Result<Vec<Uuid>, Error> {
    let run_ids = sqlx::query_scalar!(
        r#"
            SELECT
                id
            FROM
                fx_durable_ga.noise_diagnostic_runs
            WHERE
                noise_diagnostic_config_id = $1
        "#,
        noise_diagnostic_config_id
    )
    .fetch_all(tx)
    .await?;

    Ok(run_ids)
}

#[cfg(test)]
mod tests_get_noise_diagnostic_runs {
    use super::get_noise_diagnostic_runs;
    use crate::repositories::noise_diagnostics::queries::{
        store_noise_diagnostic_config, store_noise_diagnostic_run,
    };
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<(Uuid, Uuid)> {
        crate::migrations::run_default_migrations(pool).await?;

        let budget1_id = Uuid::now_v7();
        let budget2_id = Uuid::now_v7();
        let budget3_id = Uuid::now_v7();

        // Config 1 with 3 runs
        let config1 =
            store_noise_diagnostic_config(pool, "test_optimization_1", 10, 100, 1000, &budget1_id)
                .await?;
        store_noise_diagnostic_run(pool, &Uuid::now_v7(), &config1.id).await?;
        store_noise_diagnostic_run(pool, &Uuid::now_v7(), &config1.id).await?;
        store_noise_diagnostic_run(pool, &Uuid::now_v7(), &config1.id).await?;

        // Config 2 with 1 run
        let config2 =
            store_noise_diagnostic_config(pool, "test_optimization_2", 10, 100, 1000, &budget2_id)
                .await?;
        store_noise_diagnostic_run(pool, &Uuid::now_v7(), &config2.id).await?;

        // Config 3 with 0 runs
        let config3 =
            store_noise_diagnostic_config(pool, "test_optimization_3", 10, 100, 1000, &budget3_id)
                .await?;

        Ok((config1.id, config3.id))
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_all_run_ids_for_a_given_config(pool: PgPool) -> anyhow::Result<()> {
        let (config1_id, _) = seed(&pool).await?;

        let run_ids = get_noise_diagnostic_runs(&pool, &config1_id).await?;

        assert_eq!(run_ids.len(), 3);
        for run_id in run_ids {
            assert!(!run_id.is_nil());
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_an_empty_vec_for_a_config_with_no_runs(pool: PgPool) -> anyhow::Result<()> {
        let (_, config3_id) = seed(&pool).await?;

        let run_ids = get_noise_diagnostic_runs(&pool, &config3_id).await?;

        assert!(run_ids.is_empty());

        Ok(())
    }
}
