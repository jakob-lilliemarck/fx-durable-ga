use super::super::SEMAPHORE_CHANNEL;
use sqlx::PgExecutor;
use tracing::instrument;

/// Publish a notification for a semaphore
#[instrument(level = "debug", skip(tx))]
pub(crate) async fn notify<'tx, E: PgExecutor<'tx>>(
    tx: E,
    semaphore: &super::Semaphore,
) -> Result<(), super::Error> {
    let payload = serde_json::to_string(semaphore)?;

    sqlx::query("SELECT pg_notify($1, $2)")
        .bind(SEMAPHORE_CHANNEL)
        .bind(payload)
        .execute(tx)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests_notify {
    use super::super::{Semaphore, notify, store};
    use sqlx::PgPool;

    async fn seed(pool: &PgPool) -> anyhow::Result<Semaphore> {
        let semaphore = store(pool, "notify-test-semaphore").await?;
        Ok(semaphore)
    }

    #[sqlx::test(migrations = false)]
    async fn it_notifies_about_a_semaphore(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let semaphore = seed(&pool).await?;

        let result = notify(&pool, &semaphore).await;

        assert!(result.is_ok());

        Ok(())
    }
}
