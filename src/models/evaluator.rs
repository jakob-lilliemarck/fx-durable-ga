use futures::future::BoxFuture;

/// Objective function returning the fitness of the evaluated individual.
/// The returned fitness value should be between 0.0 and 1.0. Negative errors may error depending on selection method
pub trait Evaluator<P> {
    fn fitness<'a>(&self, phenotype: P) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}
