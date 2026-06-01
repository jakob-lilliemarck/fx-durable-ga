use crate::services::budgeting::repositories;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Budgeting error: {0}")]
    Transactions(#[from] repositories::transactions::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
