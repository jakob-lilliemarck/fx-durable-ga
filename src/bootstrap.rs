use sqlx::PgPool;

use crate::{
    repositories::{genotypes, morphologies, requests},
    service,
};

pub async fn bootstrap(pool: PgPool) -> anyhow::Result<service::ServiceBuilder> {
    let genotypes = genotypes::Repository::new(pool.clone());
    let morphologies = morphologies::Repository::new(pool.clone());
    let requests = requests::Repository::new(pool.clone());

    let builder = service::Service::builder(requests, morphologies, genotypes);

    Ok(builder)
}
