use fx_event_bus::PublisherError;

use super::indexable;
use super::repositories::embeddings;
use crate::services::locking;

/// Errors that can occur during indexing
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Embeddings repository error: {0}")]
    EmbeddingsRepositoryError(#[from] embeddings::Error),

    #[error("Encoders repository error: {0}")]
    EncodersRepository(#[from] super::repositories::encoders::Error),

    #[error("Event publisher error: {0}")]
    EventPublisher(#[from] PublisherError),

    #[error("Job publisher error: {0}")]
    JobPublisher(#[from] fx_mq_jobs::PublishError),

    #[error("Model serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Recorder error: {0}")]
    Recorder(#[from] burn::record::RecorderError),

    #[error("Tensor data error: {0}")]
    TensorData(#[from] burn::tensor::DataError),

    #[error("Unsupported model type: {0}")]
    UnsupportedModel(String),

    #[error("Indexable error: {0}")]
    Indexable(#[from] indexable::Error),

    #[error("Locking error: {0}")]
    LockError(#[from] locking::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
