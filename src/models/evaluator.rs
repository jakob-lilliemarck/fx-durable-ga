use futures::future::BoxFuture;

pub trait Evaluator<P> {
    fn fitness<'a>(&self, phenotype: P) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}
