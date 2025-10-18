use futures::future::BoxFuture;

// In models/mod.rs - clean trait, no service dependency
pub trait Terminated: Send + Sync {
    fn is_terminated(&self) -> BoxFuture<'_, bool>;
}

/// Objective function returning the fitness of the evaluated individual.
/// The returned fitness value should be between 0.0 and 1.0. Negative errors may error depending on selection method
pub trait Evaluator<P> {
    fn fitness<'a>(
        &self,
        phenotype: P,
        terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}
