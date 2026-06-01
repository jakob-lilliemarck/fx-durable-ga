use chrono::Utc;
use sqlx::PgExecutor;
use tracing::instrument;

#[instrument(level = "debug", skip(tx))]
pub(crate) async fn store<'tx, E: PgExecutor<'tx>>(
    tx: E,
    name: &str,
) -> Result<super::Semaphore, super::Error> {
    let now = Utc::now();

    let semaphore = sqlx::query_as!(
        super::Semaphore,
        r#"
        INSERT INTO synchronization.semaphores (name, raised_at)
        VALUES ($1, $2)
        RETURNING
            name,
            raised_at;
        "#,
        name,
        now
    )
    .fetch_one(tx)
    .await?;

    Ok(semaphore)
}

#[cfg(test)]
mod tests_store {
    use super::super::store;

    #[sqlx::test(migrations = false)]
    async fn it_stores_a_semaphore(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let semaphore = store(&pool, "test-semaphore").await?;

        assert_eq!(semaphore.name, "test-semaphore");
        assert!(semaphore.raised_at <= chrono::Utc::now());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_stores_a_second_semaphore_with_the_same_name(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let first = store(&pool, "duplicate-name").await?;
        let second = store(&pool, "duplicate-name").await?;

        assert_eq!(first.name, "duplicate-name");
        assert_eq!(second.name, "duplicate-name");
        assert!(second.raised_at >= first.raised_at);

        Ok(())
    }
}
