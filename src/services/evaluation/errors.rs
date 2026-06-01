use crate::services::evaluation::repositories::evaluations;
use crate::{repositories::genotypes, services::synchronization};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("No evaluator registered for {type_name}")]
    NotFound { type_name: String },

    #[error("Could not deserialize genome: {0}")]
    Deserialize(#[from] serde_json::error::Error),

    #[error("Could not evaluate genome: {0}")]
    Evaluation(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("Evaluation repository error: {0}")]
    EvaluationRepository(#[from] evaluations::Error),

    #[error("Genotype repository error: {0}")]
    GenotypeRepository(#[from] genotypes::Error),

    #[error("Synchronization service error: {0}")]
    SynchronizationService(#[from] synchronization::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
