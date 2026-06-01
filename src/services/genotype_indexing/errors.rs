use crate::repositories::genotypes;

/// Errors that can occur during genotype indexing operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Indexing error: {0}")]
    Indexing(#[from] crate::services::indexing::Error),

    #[error("Genotypes repository error: {0}")]
    GenotypesRepository(#[from] genotypes::Error),

    #[error("Embeddings repository error: {0}")]
    EmbeddingsRepository(#[from] crate::services::indexing::EmbeddingsError),

    #[error("Invalid genotype tag: {tag}")]
    InvalidGenotypeTag {
        tag: String,
        #[source]
        source: Option<uuid::Error>,
    },

    #[error("Invalid indexer tag: {tag}")]
    InvalidIndexerTag {
        tag: String,
        #[source]
        source: crate::services::indexing::EncoderDigestError,
    },

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
