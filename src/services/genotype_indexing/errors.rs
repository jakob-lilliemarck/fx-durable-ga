#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Indexing error: {0}")]
    Indexing(#[from] crate::services::indexing::Error),
    #[error("Genotypes repository error: {0}")]
    GenotypesRepository(#[from] crate::repositories::genotypes::Error),
}
