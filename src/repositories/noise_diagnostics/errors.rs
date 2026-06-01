#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Tx error: {0}")]
    Tx(anyhow::Error),

    #[error("Config not found for optimization type: {0}")]
    NotFound(String),

    #[error("Validation error: {0}")]
    Validation(#[from] super::models::ValidationError),
}
