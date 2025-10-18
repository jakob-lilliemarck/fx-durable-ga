use sqlx::PgPool;

use crate::repositories::{genotypes, morphologies, requests};
use crate::services::{lock, optimization};

pub async fn bootstrap(pool: PgPool) -> anyhow::Result<optimization::ServiceBuilder> {
    let genotypes = genotypes::Repository::new(pool.clone());

    let morphologies = morphologies::Repository::new(pool.clone());

    let requests = requests::Repository::new(pool.clone());

    let locking = lock::Service::new(pool.clone());

    let builder = optimization::Service::builder(locking, requests, morphologies, genotypes);
    Ok(builder)
}
