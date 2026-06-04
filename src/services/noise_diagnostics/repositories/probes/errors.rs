use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Tx error: {0}")]
    Tx(#[from] anyhow::Error),

    #[error("Probe not found: {0}")]
    NotFound(Uuid),
}
