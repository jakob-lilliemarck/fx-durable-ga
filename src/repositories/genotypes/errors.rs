use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Tx error: {0}")]
    Tx(anyhow::Error),

    #[error("Not found: {0}")]
    NotFound(Uuid),

    #[error("Database deserialization error: expected non-null value for field '{0}'")]
    DatabaseDeserialization(&'static str),
}
