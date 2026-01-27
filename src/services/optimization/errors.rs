use crate::optimization::termination_listener::TerminationListenerError;
use crate::services::lock;
use crate::{
    models::SelectionError,
    repositories::{genotypes, requests},
};

/// Errors that can occur during optimization operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("RequestsRepositoryError: {0}")]
    RequestsRepositoryError(#[from] requests::Error),
    #[error("GenotypesRepositoryError: {0}")]
    GenotypesRepositoryError(#[from] genotypes::Error),
    #[error("UnknownType: type_name={type_name}, type_hash={type_hash}")]
    UnknownTypeError { type_hash: i32, type_name: String },
    #[error("EvaluationError: {0}")]
    EvaluationError(#[from] anyhow::Error),
    #[error("Selection error: {0}")]
    SelectionError(#[from] SelectionError),
    #[error("Lock error: {0}")]
    LockError(#[from] lock::Error),
    #[error("Termination listener error: {0}")]
    TerminationListener(#[from] TerminationListenerError),
}
