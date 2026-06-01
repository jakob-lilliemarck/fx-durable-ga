use crate::repositories::genotypes;

/// Errors that can occur during genotype exploration operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error from the genotypes repository.
    #[error("GenotypesRepositoryError: {0}")]
    GenotypesRepositoryError(#[from] genotypes::Error),

    /// The supplied degree exceeds the maximum supported value (i32::MAX).
    #[error("Degree must fit within i32, got: {degree}")]
    DegreeOverflow { degree: u32 },
}
