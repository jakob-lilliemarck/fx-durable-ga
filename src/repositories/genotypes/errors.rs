/// Errors that can occur in genotypes repository operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Tx error: {0}")]
    Tx(anyhow::Error),
}
