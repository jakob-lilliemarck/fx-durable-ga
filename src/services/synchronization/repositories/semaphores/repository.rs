use chrono::{DateTime, Utc};
use sqlx::PgPool;
use tracing::instrument;

/// Repository for storing and polling semaphores.
pub struct Repository {
    pool: PgPool,
}

impl Repository {
    /// Creates a new semaphore repository.
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Polls for semaphores raised since the given time, optionally filtered by name.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn poll(
        &self,
        since: &DateTime<Utc>,
        filter: &super::queries::PollFilter,
    ) -> Result<Vec<super::Semaphore>, super::Error> {
        super::queries::poll(&self.pool, since, filter).await
    }
}
