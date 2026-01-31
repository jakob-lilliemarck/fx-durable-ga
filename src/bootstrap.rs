use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

use crate::repositories::{genotypes, requests};
use crate::services::{lock, optimization};

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

    let locking = lock::Service::new(pool.clone());

    let builder = optimization::Service::builder(host_id, locking, &requests, &genotypes);

    Ok(builder)
}
