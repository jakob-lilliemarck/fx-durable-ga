use sqlx::PgPool;

use crate::repositories::{genotypes, morphologies, requests};
use crate::services::{lock, optimization};

/// Bootstraps the optimization service with all required dependencies.
/// 
/// Creates and wires together all repositories and services needed for the genetic algorithm
/// optimization system, returning a builder for the optimization service.
pub async fn bootstrap(pool: PgPool) -> anyhow::Result<optimization::ServiceBuilder> {
    let genotypes = genotypes::Repository::new(pool.clone());

    let morphologies = morphologies::Repository::new(pool.clone());

    let requests = requests::Repository::new(pool.clone());

    let locking = lock::Service::new(pool.clone());

    let builder = optimization::Service::builder(locking, requests, morphologies, genotypes);
    Ok(builder)
}
