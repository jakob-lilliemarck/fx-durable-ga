use crate::repositories::{genotypes, morphologies, populations, requests};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RequestsRepositoryError: {0}")]
    RequestsRepositoryError(#[from] requests::Error),
    #[error("MorphologiesRepositoryError: {0}")]
    MorphologiesRepositoryError(#[from] morphologies::Error),
    #[error("GenotypesRepositoryError: {0}")]
    GenotypesRepositoryError(#[from] genotypes::Error),
    #[error("PopulationsRepositoryError: {0}")]
    PopulationsRepositoryError(#[from] populations::Error),
    #[error("UnknownType: type_name={type_name}, type_hash={type_hash}")]
    UnknownTypeError { type_hash: i32, type_name: String },
    #[error("EvaluationError: {0}")]
    EvaluationError(#[from] anyhow::Error),
    #[error("NoValidParents")]
    NoValidParents,
}
