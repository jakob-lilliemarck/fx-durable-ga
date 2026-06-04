#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("evaluation_count must be positive, was {0}")]
    InvalidEvaluationCount(i32),

    #[error(transparent)]
    Database(#[from] sqlx::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
