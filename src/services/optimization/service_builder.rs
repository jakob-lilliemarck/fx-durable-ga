use super::models::TypeErasedEvaluator;
use crate::{
    models::{Encodeable, Evaluator, Morphology},
    repositories::{genotypes, morphologies, requests},
    services::{lock, optimization::Service},
};
use std::collections::HashMap;
use tracing::instrument;

/// Builder for creating optimization services with registered type evaluators.
/// Handles morphology registration and evaluator type erasure.
pub struct ServiceBuilder {
    pub(super) locking: lock::Service,
    pub(super) requests: requests::Repository,
    pub(super) morphologies: morphologies::Repository,
    pub(super) genotypes: genotypes::Repository,
    pub(super) evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
    pub(super) max_deduplication_attempts: i32,
}

impl ServiceBuilder {
    /// Registers an evaluator for a specific type, ensuring its morphology exists in the database.
    /// Creates the morphology if it doesn't exist and stores the type-erased evaluator.
    #[instrument(level = "debug", skip(self, evaluator), fields(type_name = T::NAME, type_hash = T::HASH))]
    pub async fn register<T, E>(mut self, evaluator: E) -> Result<Self, super::Error>
    where
        T: Encodeable + 'static,
        E: Evaluator<T::Phenotype> + Send + Sync + 'static,
    {
        // Insert the morphology of the type if it does not already exist in the database
        if let Err(morphologies::Error::NotFound) = self.morphologies.get_morphology(T::HASH).await
        {
            self.morphologies
                .new_morphology(Morphology::new(T::NAME, T::HASH, T::morphology()))
                .await?;
        }

        // Erase the type and store the evaluator
        let erased = super::models::ErasedEvaluator::new(evaluator, T::decode);
        self.evaluators.insert(T::HASH, Box::new(erased));

        Ok(self)
    }

    /// Sets the maximum number of deduplication attempts when breeding genotypes.
    pub fn with_max_deduplication_attempts(mut self, attempts: i32) -> Self {
        self.max_deduplication_attempts = attempts;
        self
    }

    /// Builds the optimization service with all registered evaluators.
    #[instrument(level = "debug", skip(self), fields(evaluators_count = self.evaluators.len()))]
    pub fn build(self) -> Service {
        Service {
            locking: self.locking,
            requests: self.requests,
            morphologies: self.morphologies,
            genotypes: self.genotypes,
            evaluators: self.evaluators,
            max_deduplication_attempts: self.max_deduplication_attempts,
        }
    }
}
