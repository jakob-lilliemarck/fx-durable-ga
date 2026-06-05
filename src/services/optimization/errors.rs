use crate::repositories::genotypes;
use crate::repositories::genotypes::GenotypeError;
use crate::services::budgeting::{self, TransactionsError};
use crate::services::evaluation;
use crate::services::evaluation::repositories::evaluations;
use crate::services::locking;
use crate::services::optimization::SelectionError;
use crate::services::optimization::repositories::requests;
use crate::services::synchronization;
use uuid::Uuid;

/// Errors that can occur during optimization operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RequestsRepositoryError: {0}")]
    RequestsRepositoryError(#[from] requests::Error),

    #[error("GenotypesRepositoryError: {0}")]
    GenotypesRepositoryError(#[from] genotypes::Error),

    #[error("UnknownType: type_name={type_name}")]
    UnknownTypeError { type_name: String },

    #[error("Selection error: {0}")]
    SelectionError(#[from] SelectionError),

    #[error("Lock error: {0}")]
    LockError(#[from] locking::Error),

    #[error("No fitness available for genotype: {genotype_id}")]
    NoFitness { genotype_id: Uuid },

    #[error("Genotype error {0}")]
    Genotype(#[from] GenotypeError),

    #[error("Sync error: {0}")]
    Sync(#[from] synchronization::Error),

    #[error("Budgeting error: {0}")]
    Budgeting(#[from] budgeting::Error),

    #[error("Transactions repository error: {0}")]
    TransactionsRepository(#[from] TransactionsError),

    #[error("Publish error: {0}")]
    Publish(#[from] fx_mq_jobs::PublishError),

    #[error("No requeust associated with account: {0}")]
    NoRequestOfAccount(Uuid),

    #[error("Breeding called with zero or negative count: {0}")]
    CouldNotBreed(i64),

    #[error(
        "Invalid configuration: population_size ({population_size}) must be >= {required} for the selected selector"
    )]
    InvalidConfiguration {
        population_size: u32,
        required: usize,
    },

    #[error("Evaluation service error: {0}")]
    EvaluationService(#[from] evaluation::Error),

    #[error("Evaluations repository error: {0}")]
    EvaluationsRepository(#[from] evaluations::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
