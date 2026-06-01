use super::super::Error;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx))]
pub async fn get_noise_diagnostic_evaluations<'tx, E: PgExecutor<'tx>>(
    tx: E,
    noise_diagnostic_run_id: &Uuid,
) -> Result<Vec<Uuid>, Error> {
    let evaluation_ids = sqlx::query_scalar!(
        r#"
            SELECT
                evaluation_id
            FROM
                fx_durable_ga.noise_diagnostic_run_evaluations
            WHERE
                noise_diagnostic_run_id = $1
        "#,
        noise_diagnostic_run_id
    )
    .fetch_all(tx)
    .await?;

    Ok(evaluation_ids)
}

#[cfg(test)]
mod tests_get_noise_diagnostic_evaluations {
    use super::get_noise_diagnostic_evaluations;
    use crate::repositories::noise_diagnostics::queries::{
        store_noise_diagnostic_config, store_noise_diagnostic_run,
        store_noise_diagnostic_run_evaluations::store_noise_diagnostic_run_evaluations,
    };
    use chrono::Utc;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<(Uuid, Vec<Uuid>)> {
        crate::migrations::run_default_migrations(pool).await?;

        let config =
            store_noise_diagnostic_config(pool, "test_optimization", 10, 100, 1000, &Uuid::nil())
                .await?;
        let run_id = store_noise_diagnostic_run(pool, &Uuid::now_v7(), &config.id).await?;

        let evaluation_ids = vec![Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()];

        store_noise_diagnostic_run_evaluations(pool, &run_id, &evaluation_ids, &Utc::now()).await?;

        Ok((run_id, evaluation_ids))
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_all_evaluation_ids_for_a_run(pool: PgPool) -> anyhow::Result<()> {
        let (run_id, expected_ids) = seed(&pool).await?;

        let evaluation_ids = get_noise_diagnostic_evaluations(&pool, &run_id).await?;

        // We sort the vecs because the order of IDs is not guaranteed.
        let mut sorted_actual = evaluation_ids;
        let mut sorted_expected = expected_ids;
        sorted_actual.sort();
        sorted_expected.sort();

        assert_eq!(sorted_actual, sorted_expected);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_an_empty_vec_when_no_evaluations_are_associated(
        pool: PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let config =
            store_noise_diagnostic_config(&pool, "test_optimization", 10, 100, 1000, &Uuid::nil())
                .await?;
        let run_id = store_noise_diagnostic_run(&pool, &Uuid::now_v7(), &config.id).await?;

        let evaluation_ids = get_noise_diagnostic_evaluations(&pool, &run_id).await?;

        assert!(evaluation_ids.is_empty());

        Ok(())
    }
}
