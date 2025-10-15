use crate::models::Evaluator;
use futures::future::BoxFuture;
use tracing::instrument;

pub(crate) trait TypeErasedEvaluator: Send + Sync {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}

pub(crate) struct ErasedEvaluator<P, E: Evaluator<P>> {
    evaluator: E,
    decode: fn(&[i64]) -> P,
}

impl<P, E: Evaluator<P>> ErasedEvaluator<P, E> {
    pub(crate) fn new(evaluator: E, decode: fn(&[i64]) -> P) -> Self {
        Self { evaluator, decode }
    }
}

impl<P, E> TypeErasedEvaluator for ErasedEvaluator<P, E>
where
    E: Evaluator<P> + Send + Sync + 'static,
{
    #[instrument(level = "debug", skip(self, genes), fields(genome_length = genes.len()))]
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        let phenotype = (self.decode)(genes);
        self.evaluator.fitness(phenotype)
    }
}
