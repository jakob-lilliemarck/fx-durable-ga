use super::Error;
use super::models::{NoiseProbe, SearchNoiseProbesFilter};
use chrono::Utc;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx))]
pub async fn store_noise_probe<'tx, E: PgExecutor<'tx>>(
    tx: E,
    probe: &NoiseProbe,
) -> Result<NoiseProbe, Error> {
    sqlx::query_as!(
        NoiseProbe,
        r#"
            INSERT INTO fx_durable_ga.noise_probes (id, genotype_id, request_id, evaluation_count, created_at)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, genotype_id, request_id, evaluation_count, created_at
        "#,
        probe.id,
        probe.genotype_id,
        probe.request_id,
        probe.evaluation_count,
        probe.created_at,
    )
    .fetch_one(tx)
    .await
    .map_err(Error::from)
}

#[instrument(level = "debug", skip(tx))]
pub async fn search_noise_probes<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchNoiseProbesFilter,
) -> Result<Vec<NoiseProbe>, Error> {
    sqlx::query_as!(
        NoiseProbe,
        r#"
            SELECT id, genotype_id, request_id, evaluation_count, created_at
            FROM fx_durable_ga.noise_probes
            WHERE ($1::uuid IS NULL OR request_id = $1)
            ORDER BY created_at ASC
        "#,
        filter.request_id,
    )
    .fetch_all(tx)
    .await
    .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migrations;
    use sqlx::PgPool;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn store_and_search_by_request_id(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let request_id = Uuid::now_v7();
        let probe = NoiseProbe::new(Uuid::now_v7(), request_id, 10);

        let stored = store_noise_probe(&pool, &probe).await?;
        assert_eq!(stored.id, probe.id);

        let results = search_noise_probes(
            &pool,
            &SearchNoiseProbesFilter::default().with_request_id(request_id),
        )
        .await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, probe.id);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn search_returns_empty_for_unknown_request(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let results = search_noise_probes(
            &pool,
            &SearchNoiseProbesFilter::default().with_request_id(Uuid::now_v7()),
        )
        .await?;

        assert!(results.is_empty());

        Ok(())
    }
}
