use futures::channel::{mpsc::SendError, oneshot::Canceled};

/// Errors that can occur during locking operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Database error during lock acquisition or release.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Failed to send lock release signal.
    #[error("SendError: {0}")]
    SendError(#[from] SendError),

    /// Lock release signal was cancelled.
    #[error("Canceled: {0}")]
    Canceled(#[from] Canceled),

    /// Catch-all for unclassified errors.
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
