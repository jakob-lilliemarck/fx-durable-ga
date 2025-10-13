use sqlx::PgPool;

use crate::{
    repositories::{genotypes, morphologies, populations, requests},
    service,
};

pub async fn bootstrap(pool: PgPool) -> anyhow::Result<service::ServiceBuilder> {
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
