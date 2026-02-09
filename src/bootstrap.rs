use crate::infrastructure::typesafe_builder::{Set, Unset};
use crate::repositories::{embeddings, encoders, genotypes, requests};
use crate::services::{self, genotype_explorer, genotype_indexing, indexing, lock, optimization};
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

/// Bootstraps the optimization service with all required dependencies.
///
/// Creates and wires together all repositories and services needed for the genetic algorithm
/// optimization system, returning a builder for the optimization service.
pub async fn bootstrap(
    host_id: Uuid,
    pool: PgPool,
) -> anyhow::Result<optimization::ServiceBuilder> {
    let genotypes = Arc::new(genotypes::Repository::new(pool.clone()));

    let requests = Arc::new(requests::Repository::new(pool.clone()));

    let lock = Arc::new(lock::Service::new(pool.clone()));

    let builder = optimization::Service::builder(host_id, &lock, &requests, &genotypes);

    Ok(builder)
}

pub struct ApplicationBuilder<T1> {
    _pool: T1,
    lock: Option<Arc<lock::Service>>,
    genotypes: Option<Arc<genotypes::Repository>>,
    requests: Option<Arc<requests::Repository>>,
    embeddings: Option<Arc<embeddings::Repository>>,
    encoders: Option<Arc<encoders::Repository>>,
}

impl Default for ApplicationBuilder<Unset<PgPool>> {
    fn default() -> Self {
        Self {
            _pool: Unset::new(),
            genotypes: None,
            requests: None,
            lock: None,
            embeddings: None,
            encoders: None,
        }
    }
}

impl ApplicationBuilder<Unset<PgPool>> {
    pub fn with_pool(self, pool: PgPool) -> ApplicationBuilder<Set<PgPool>> {
        // Construct everything that depends on pool
        let genotypes = Arc::new(genotypes::Repository::new(pool.clone()));
        let requests = Arc::new(requests::Repository::new(pool.clone()));
        let lock = Arc::new(lock::Service::new(pool.clone()));
        let embeddings = Arc::new(embeddings::Repository::new(pool.clone()));
        let encoders = Arc::new(encoders::Repository::new(pool.clone()));

        ApplicationBuilder {
            _pool: Set::new(pool),
            genotypes: Some(genotypes),
            requests: Some(requests),
            lock: Some(lock),
            embeddings: Some(embeddings),
            encoders: Some(encoders),
        }
    }
}

impl ApplicationBuilder<Set<PgPool>> {
    pub fn build_explorer_svc(&self) -> services::genotype_explorer::Service {
        match self {
            ApplicationBuilder {
                genotypes: Some(genotypes),
                ..
            } => genotype_explorer::Service::new(&genotypes),
            _ => panic!("Missing dependency while constructing explorer service"),
        }
    }

    pub fn build_optimization_svc(&self, host_id: &Uuid) -> services::optimization::Service {
        match self {
            ApplicationBuilder {
                genotypes: Some(genotypes),
                requests: Some(requests),
                lock: Some(lock),
                ..
            } => optimization::Service::builder(host_id.clone(), &lock, &requests, &genotypes)
                .build(),
            _ => panic!("Missing dependency while constructing optimization service"),
        }
    }

    pub fn indexing_service(&self) -> services::indexing::ServiceBuilder {
        match self {
            ApplicationBuilder {
                embeddings: Some(embeddings),
                encoders: Some(encoders),
                ..
            } => indexing::Service::builder(embeddings.clone(), encoders.clone()),
            _ => panic!("Missing dependency while constructing indexing service"),
        }
    }

    pub fn genotype_indexing_service(
        &self,
        indexing: Arc<services::indexing::Service>,
    ) -> services::genotype_indexing::ServiceBuilder {
        match self {
            ApplicationBuilder {
                genotypes: Some(genotypes),
                ..
            } => genotype_indexing::Service::builder(genotypes.clone(), indexing.clone()),
            _ => panic!("Missing dependency while constructing indexing service"),
        }
    }
}
