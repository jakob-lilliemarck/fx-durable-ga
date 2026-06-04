use futures::{
    SinkExt,
    channel::{mpsc, oneshot},
    lock::Mutex,
};
use sqlx::{PgPool, postgres::PgAdvisoryLock};
use std::sync::Arc;
use tracing::instrument;

/// Shared channel for coordinating lock release across concurrent operations.
pub type TxControl = Arc<Mutex<mpsc::Sender<oneshot::Sender<()>>>>;

/// Provides advisory locking for synchronous execution of critical sections.
pub struct Service {
    pool: PgPool,
    tx: Option<TxControl>,
}

impl Service {
    /// Creates a new locking service with the given database pool.
    pub fn new(pool: PgPool, tx: Option<TxControl>) -> Self {
        Self { pool, tx }
    }

    /// Acquires an advisory lock and executes the given function exclusively.
    #[instrument(level = "debug", skip(self, f))]
    pub(crate) async fn lock_while<F, Fut, T>(&self, key: &str, f: F) -> Result<T, super::Error>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let lock = PgAdvisoryLock::new(&key);
        let conn = self.pool.acquire().await?;
        let acquired = lock.acquire(conn).await?;

        if let Some(ref mutex) = self.tx {
            let (tx_release, rx_release) = oneshot::channel();
            let mut tx = mutex.lock().await;
            tx.send(tx_release).await?;
            rx_release.await?;
        }

        // Execute the passed function within the duration of the global lock
        let ret = f().await;

        acquired.release_now().await?;

        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::Service;
    use std::sync::{
        Arc,
        atomic::{AtomicI32, Ordering},
    };

    #[sqlx::test(migrations = false)]
    async fn it_executes_function_under_lock(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let service = Service::new(pool, None);
        let result = service.lock_while("test_key", || async { 42 }).await?;

        assert_eq!(result, 42);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_serializes_concurrent_calls(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let service = Arc::new(Service::new(pool, None));
        let counter = Arc::new(AtomicI32::new(0));
        let (lock_acquired_tx, lock_acquired_rx) = tokio::sync::oneshot::channel::<()>();

        let s1 = service.clone();
        let c1 = counter.clone();
        let t1 = tokio::spawn(async move {
            s1.lock_while("concurrent", || {
                let c = c1.clone();
                let tx = lock_acquired_tx;
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    let _ = tx.send(());
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                }
            })
            .await
        });

        let s2 = service.clone();
        let c2 = counter.clone();
        let t2 = tokio::spawn(async move {
            let _ = lock_acquired_rx.await;
            s2.lock_while("concurrent", || {
                let c = c2.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                }
            })
            .await
        });

        let (r1, r2) = tokio::join!(t1, t2);
        r1??;
        r2??;

        assert_eq!(counter.load(Ordering::SeqCst), 3);
        Ok(())
    }
}
