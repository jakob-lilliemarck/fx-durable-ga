use super::service::GroupingKey;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Indexing error: {0}")]
    Indexing(#[from] crate::services::indexing::Error),

    #[error("Genotypes repository error: {0}")]
    GenotypesRepository(#[from] crate::repositories::genotypes::Error),

    #[error("Encoders repository error: {0}")]
    EncodersRepository(#[from] crate::repositories::encoders::Error),

    #[error("No genotype indexer registered for hash: {0}")]
    NoIndexerRegistered(i32),

    #[error(
        "Grouping mismatch for genotype {genotype_id}: expected {grouping:?}, got {actual_type_name} (hash {actual_type_hash})"
    )]
    GroupingMismatch {
        grouping: GroupingKey,
        genotype_id: Uuid,
        actual_type_hash: i32,
        actual_type_name: String,
    },
}
