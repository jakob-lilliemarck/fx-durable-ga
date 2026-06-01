use super::Digest;
use std::sync::PoisonError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Not found: {0}")]
    NotFound(Digest),

    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Tx error: {0}")]
    Tx(anyhow::Error),
}

impl<T> From<PoisonError<T>> for Error {
    fn from(err: PoisonError<T>) -> Self {
        Error::LockPoisoned(err.to_string())
    }
}
