use super::super::Semaphore;
use chrono::{DateTime, Utc};
use sqlx::PgExecutor;
use tracing::instrument;

#[derive(Debug)]
pub struct Filter {
    name: Option<String>,
}

impl Default for Filter {
    fn default() -> Self {
        Self { name: None }
    }
}

impl Filter {
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}

/// Poll for all semaphores raised since the provided time,
/// optionally filtered by Filter
#[instrument(level = "debug", skip(tx))]
pub async fn poll<'tx, E: PgExecutor<'tx>>(
    tx: E,
    since: &DateTime<Utc>,
    filter: &Filter,
) -> Result<Vec<Semaphore>, super::Error> {
    let semaphores = sqlx::query_as!(
        super::super::Semaphore,
        r#"
        SELECT name, raised_at
        FROM synchronization.semaphores
        WHERE raised_at > $1::TIMESTAMPTZ
            AND ($2::TEXT IS NULL OR $2::TEXT = name);
        "#,
        since,
        filter.name
    )
    .fetch_all(tx)
    .await?;

    Ok(semaphores)
}

#[cfg(test)]
mod tests_poll {
    use super::super::store;
    use super::{Filter, poll};
    use chrono::{Duration, Utc};
    use sqlx::PgPool;

    struct TestData {
        semaphore_a_name: String,
        raised_at: chrono::DateTime<Utc>,
    }

    async fn seed(pool: &PgPool) -> anyhow::Result<TestData> {
        crate::migrations::run_default_migrations(pool).await?;

        let a = store(pool, "test-a").await?;
        let _b = store(pool, "test-b").await?;

        Ok(TestData {
            semaphore_a_name: a.name,
            raised_at: a.raised_at,
        })
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_when_no_semaphores_raised(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let since = Utc::now();
        let results = poll(&pool, &since, &Filter::default()).await?;

        assert!(results.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_semaphores_raised_after_since(pool: PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let since = data.raised_at - Duration::seconds(1);
        let results = poll(&pool, &since, &Filter::default()).await?;

        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_excludes_semaphores_raised_before_since(pool: PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let since = data.raised_at + Duration::seconds(1);
        let results = poll(&pool, &since, &Filter::default()).await?;

        assert!(results.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_name(pool: PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let since = data.raised_at - Duration::seconds(1);
        let filter = Filter::default().with_name(data.semaphore_a_name.clone());
        let results = poll(&pool, &since, &filter).await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, data.semaphore_a_name);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_for_unmatched_name(pool: PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let since = data.raised_at - Duration::seconds(1);
        let filter = Filter::default().with_name("non-existent".to_string());
        let results = poll(&pool, &since, &filter).await?;

        assert!(results.is_empty());

        Ok(())
    }
}
