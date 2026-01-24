use crate::{
    models::GenotypeManager,
    repositories::{genotypes, requests},
    services::{lock, optimization::Service},
};
use std::collections::HashMap;
use tracing::instrument;

/// Builder for creating optimization services with registered genotype managers.
pub struct ServiceBuilder {
    pub(super) locking: lock::Service,
    pub(super) requests: requests::Repository,
    pub(super) genotypes: genotypes::Repository,
    pub(super) genotype_managers: HashMap<i32, Box<dyn GenotypeManager + 'static>>,
    pub(super) max_deduplication_attempts: i32,
}

impl ServiceBuilder {
    #[instrument(level = "debug", skip(self, manager), fields(type_name = manager.name(), type_hash = manager.hash()))]
    pub fn with_genotype_manager<M>(mut self, manager: M) -> Self
    where
        M: GenotypeManager + 'static,
    {
        self.genotype_managers
            .insert(manager.hash(), Box::new(manager));
        self
    }

    /// Sets the maximum number of deduplication attempts when breeding genotypes.
    pub fn with_max_deduplication_attempts(mut self, attempts: i32) -> Self {
        self.max_deduplication_attempts = attempts;
        self
    }

    /// Builds the optimization service with all registered managers.
    #[instrument(level = "debug", skip(self), fields(managers_count = self.genotype_managers.len()))]
    pub fn build(self) -> Service {
        Service {
            locking: self.locking,
            requests: self.requests,
            genotypes: self.genotypes,
            genotype_managers: self.genotype_managers,
            max_deduplication_attempts: self.max_deduplication_attempts,
        }
    }
}
