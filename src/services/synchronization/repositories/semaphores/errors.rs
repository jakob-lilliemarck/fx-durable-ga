/// Errors that can occur during semaphore operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("PgMux error: {0}")]
    PgMux(#[from] fx_pgmux::Error),

    #[error("Tx error: {0}")]
    Tx(anyhow::Error),

    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
}
