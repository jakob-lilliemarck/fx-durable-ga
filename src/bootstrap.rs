use sqlx::PgPool;

use crate::{
    repositories::{genotypes, morphologies, populations, requests},
    service,
};

pub struct Configuration {
    database_url: String,
}

pub async fn bootstrap_optimizer<'o>(
    config: Configuration,
) -> anyhow::Result<service::ServiceBuilder<'o>> {
    let pool = PgPool::connect(&config.database_url).await?;

    let genotypes = genotypes::Repository::new(pool.clone());
    let populations = populations::Repository::new(pool.clone());
    let morphologies = morphologies::Repository::new(pool.clone());
    let requests = requests::Repository::new(pool.clone());

    Ok(service::Service::builder(
        requests,
        morphologies,
        genotypes,
        populations,
    ))
}
