use crate::infrastructure::typesafe_builder::{Set, Unset};
use crate::repositories::{embeddings, encoders, genotypes, requests};
use crate::services::{self, genotype_explorer, genotype_indexing, indexing, lock, optimization};
use fx_mq_jobs::{FX_MQ_JOBS_SCHEMA_NAME, Queries};
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
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
    mq_queries: Arc<Queries>,
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
            mq_queries: Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME)),
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

        // Number of encoders that can be cached at any one time
        let capacity = 20;
        // Expiration time after which cached encoders are re-fetched from the database
        let ttl = Duration::from_secs(60 * 10);
        let encoders = Arc::new(encoders::Repository::new(pool.clone(), ttl, capacity));

        ApplicationBuilder {
            _pool: Set::new(pool),
            genotypes: Some(genotypes),
            requests: Some(requests),
            lock: Some(lock),
            embeddings: Some(embeddings),
            encoders: Some(encoders),
            mq_queries: self.mq_queries,
        }
    }
}

impl ApplicationBuilder<Set<PgPool>> {
    pub fn explorer_service(&self) -> services::genotype_explorer::Service {
        match self {
            ApplicationBuilder {
                genotypes: Some(genotypes),
                ..
            } => genotype_explorer::Service::new(&genotypes),
            _ => panic!("Missing dependency while constructing explorer service"),
        }
    }

    pub fn optimization_service(&self, host_id: &Uuid) -> services::optimization::Service {
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
                encoders: Some(encoders),
                genotypes: Some(genotypes),
                mq_queries,
                ..
            } => genotype_indexing::Service::builder(
                genotypes.clone(),
                encoders.clone(),
                indexing.clone(),
                mq_queries.clone(),
            ),
            _ => panic!("Missing dependency while constructing indexing service"),
        }
    }
}
