use crate::repositories;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Encoders repository error: {0}")]
    EncodersRepository(#[from] crate::repositories::encoders::Error),

    #[error("Embeddings repository error: {0}")]
    EmbeddingsRepositoryError(#[from] repositories::embeddings::Error),

    #[error("Genotypes repository error: {0}")]
    GenotypesRepository(#[from] crate::repositories::genotypes::Error),

    #[error("The encoder could not be found: {0}")]
    NotFoundEncoder(Uuid),

    #[error("The specified encoder could not be found: {0}")]
    NoEncoder(Uuid),

    #[error("Model serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Recorder error: {0}")]
    Recorder(#[from] burn::record::RecorderError),

    #[error("Tensor data error: {0}")]
    TensorData(#[from] burn::tensor::DataError),

    #[error("Unsupported model type: {0}")]
    UnsupportedModel(String),

    #[error("MisalignedInput")]
    MisalignedInput,
}
