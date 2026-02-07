use crate::repositories;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Encoders repository error: {0}")]
    EncodersRepository(#[from] crate::repositories::encoders::Error),

    #[error("Embeddings repository error: {0}")]
    EmbeddingsRepositoryError(#[from] repositories::embeddings::Error),

    #[error("Model serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Recorder error: {0}")]
    Recorder(#[from] burn::record::RecorderError),

    #[error("Tensor data error: {0}")]
    TensorData(#[from] burn::tensor::DataError),

    #[error("Unsupported model type: {0}")]
    UnsupportedModel(String),
}
