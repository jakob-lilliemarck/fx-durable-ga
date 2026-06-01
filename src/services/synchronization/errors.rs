use super::repositories::semaphores;
use tokio::sync::broadcast;
use tokio::task::JoinError;

/// Errors that can occur during synchronization operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The broadcast channel was closed.
    #[error("Could not receive messages over closed broadcast channel")]
    Closed,

    /// An error from the pgmux multiplexer.
    #[error("PgMux repository: {0}")]
    PgMux(#[from] fx_pgmux::Error),

    /// Failed to broadcast a semaphore notification.
    #[error("Broadcast error: {0}")]
    Broadcast(#[from] broadcast::error::SendError<semaphores::Semaphore>),

    /// Failed to send stop signal to the agent.
    #[error("Failed to send stop signal - receiver was dropped")]
    OneshotSend,

    /// The agent's termination channel was already taken.
    #[error("Missing termination channel")]
    MissingTerminationChannel,

    /// The agent task failed to join.
    #[error("Agent task join error: {0}")]
    Join(#[from] JoinError),

    /// Failed to parse a notification payload.
    #[error("Parsing error: {0}")]
    Parsing(#[from] serde_json::Error),

    /// An error from the semaphores repository.
    #[error("Semaphores repository: {0}")]
    SemaphoresRepository(#[from] semaphores::Error),

    /// The service has not been initialized via `register`.
    #[error("Uninitialized access")]
    Uninitialized,
}
