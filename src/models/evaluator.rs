use futures::future::BoxFuture;
use uuid::Uuid;

/// Provides termination checking capability for long-running evaluations.
/// Allows evaluators to check if they should abort early due to request cancellation.
pub trait Terminated: Send + Sync {
    /// Returns true if the evaluation should be terminated early.
    fn is_terminated(&self) -> BoxFuture<'_, bool>;
}

/// Objective function that evaluates phenotypes and returns fitness scores.
/// Fitness values can be any finite f64 values appropriate for your problem domain.
pub trait Evaluator<P> {
    /// Evaluates a phenotype and returns its fitness score.
    /// Use the terminated checker for long-running evaluations to support early termination.
    fn fitness<'a>(
        &self,
        genotype_id: Uuid,
        phenotype: P,
        terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}
