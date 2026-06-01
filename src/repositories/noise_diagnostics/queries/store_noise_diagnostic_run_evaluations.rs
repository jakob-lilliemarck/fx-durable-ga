use super::super::Error;
use chrono::{DateTime, Utc};
use sqlx::{PgExecutor, QueryBuilder};
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx, evaluation_ids))]
pub async fn store_noise_diagnostic_run_evaluations<'tx, E: PgExecutor<'tx>>(
    tx: E,
    noise_diagnostic_run_id: &Uuid,
    evaluation_ids: &[Uuid],
    referenced_at: &DateTime<Utc>,
) -> Result<(), Error> {
    if evaluation_ids.is_empty() {
        return Ok(());
    }

    let mut query_builder: QueryBuilder<sqlx::Postgres> = QueryBuilder::new(
        r#"
            INSERT INTO fx_durable_ga.noise_diagnostic_run_evaluations (
                noise_diagnostic_run_id,
                evaluation_id,
                referenced_at
        )
        "#,
    );

    query_builder.push_values(evaluation_ids.iter(), |mut b, evaluation_id| {
        b.push_bind(noise_diagnostic_run_id)
            .push_bind(evaluation_id)
            .push_bind(referenced_at);
    });

    let query = query_builder.build();
    query.execute(tx).await?;

    Ok(())
}

#[cfg(test)]
mod tests_store_noise_diagnostic_run_evaluations {
    use super::store_noise_diagnostic_run_evaluations;
    use crate::repositories::noise_diagnostics::queries::{
        get_noise_diagnostic_evaluations::get_noise_diagnostic_evaluations,
        store_noise_diagnostic_config, store_noise_diagnostic_run,
    };
    use chrono::Utc;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<Uuid> {
        crate::migrations::run_default_migrations(pool).await?;

        let config =
            store_noise_diagnostic_config(pool, "test_optimization", 10, 100, 1000, &Uuid::nil())
                .await?;

        let run_id = store_noise_diagnostic_run(pool, &Uuid::now_v7(), &config.id).await?;

        Ok(run_id)
    }

    #[sqlx::test(migrations = false)]
    async fn it_stores_noise_diagnostic_run_evaluations(pool: PgPool) -> anyhow::Result<()> {
        let run_id = seed(&pool).await?;
        let evaluation_ids = vec![Uuid::now_v7(), Uuid::now_v7()];

        store_noise_diagnostic_run_evaluations(&pool, &run_id, &evaluation_ids, &Utc::now())
            .await?;

        let persisted_ids = get_noise_diagnostic_evaluations(&pool, &run_id).await?;
        assert_eq!(persisted_ids, evaluation_ids);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_when_noise_diagnostic_run_id_does_not_exist(
        pool: PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let run_id = Uuid::now_v7();
        let evaluation_ids = vec![Uuid::now_v7(), Uuid::now_v7()];

        let result =
            store_noise_diagnostic_run_evaluations(&pool, &run_id, &evaluation_ids, &Utc::now())
                .await;

        assert!(result.is_err());

        Ok(())
    }
}
