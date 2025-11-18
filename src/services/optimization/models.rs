use crate::{
    models::{Evaluator, Genotype, Request, Terminated},
    repositories::requests,
};
use futures::future::BoxFuture;
use tracing::instrument;
use uuid::Uuid;

/// Checks if an optimization request has been terminated by querying for request conclusions.
pub(crate) struct Terminator {
    request_id: Uuid,
    requests: requests::Repository,
}

impl Terminator {
    /// Creates a new terminator for the given request.
    #[instrument(level = "debug", skip(requests), fields(request_id = %request_id))]
    pub(crate) fn new(requests: requests::Repository, request_id: Uuid) -> Self {
        Terminator {
            request_id,
            requests,
        }
    }
}

impl Terminated for Terminator {
    #[instrument(level = "debug", skip(self), fields(request_id = %self.request_id))]
    fn is_terminated(&self) -> BoxFuture<'_, bool> {
        let requests = self.requests.clone();

        Box::pin(async move {
            match requests.get_request_conclusion(&self.request_id).await {
                Ok(Some(_)) => true, // Any conclusion means terminate
                Err(err) => {
                    tracing::warn!(message = "Failed to check request conclusion", err = ?err);
                    false
                }
                _ => false, // No conclusion yet, keep going
            }
        })
    }
}

/// Type-erased evaluator trait that allows storing different evaluator types in a collection.
/// Converts gene arrays to phenotypes and delegates to the underlying evaluator.
pub(crate) trait TypeErasedEvaluator: Send + Sync {
    fn fitness<'a>(
        &self,
        genotype: &Genotype,
        request: &'a Request,
        terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>>;
}

/// Wraps a typed evaluator to implement the type-erased interface.
/// Handles decoding genes to phenotypes before evaluation.
pub(crate) struct ErasedEvaluator<P, E: Evaluator<P>> {
    evaluator: E,
    decode: fn(&[i64]) -> P,
}

impl<P, E: Evaluator<P>> ErasedEvaluator<P, E> {
    /// Creates a new erased evaluator with the given evaluator and decode function.
    pub(crate) fn new(evaluator: E, decode: fn(&[i64]) -> P) -> Self {
        Self { evaluator, decode }
    }
}

impl<P, E> TypeErasedEvaluator for ErasedEvaluator<P, E>
where
    E: Evaluator<P> + Send + Sync + 'static,
{
    #[instrument(level = "debug", skip(self, genotype, request, terminated), fields(genotype_id = %genotype.id(), genome_length = genotype.genome().len()))]
    fn fitness<'a>(
        &self,
        genotype: &Genotype,
        request: &'a Request,
        terminated: &'a Box<dyn Terminated>,
    ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        let phenotype = (self.decode)(&genotype.genome());
        self.evaluator
            .fitness(genotype.id(), phenotype, request, terminated)
    }
}
