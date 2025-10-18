use sqlx::{PgPool, postgres::PgAdvisoryLock};

pub(crate) struct Service {
    pool: PgPool,
}

impl Service {
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub(crate) async fn lock_while<F, Fut, T>(&self, key: &str, f: F) -> Result<T, super::Error>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let lock = PgAdvisoryLock::new(&key);
        let conn = self.pool.acquire().await?;
        let acquired = lock.acquire(conn).await?;

        // Execute the passed function within the duration of the global lock
        let ret = f().await;

        acquired.release_now().await?;
        Ok(ret)
    }
}
