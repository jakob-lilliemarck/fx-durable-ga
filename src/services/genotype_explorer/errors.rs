use crate::repositories::genotypes;

/// Errors that can occur during genotype exploration operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("GenotypesRepositoryError: {0}")]
    GenotypesRepositoryError(#[from] genotypes::Error),
    #[error("Degree must fit within i32, got: {degree}")]
    DegreeOverflow { degree: u32 },
}
