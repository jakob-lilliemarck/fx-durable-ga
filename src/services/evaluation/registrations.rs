use super::repositories::evaluations;
use crate::{
    configuration::HostId,
    infrastructure::di::{Container, ProviderResult},
    repositories::genotypes,
    services::synchronization,
};
use futures::future::BoxFuture;
use std::sync::Arc;

pub fn provide_evaluation_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let host_id = c.get::<HostId>().await?;

        let synchronization = c.get::<Arc<synchronization::Service>>().await?;

        let genotypes_ro = c.get::<genotypes::Read>().await?;
        let evaluations_ro = c.get::<evaluations::Read>().await?;
        let evaluations_wr = c.get::<evaluations::Write>().await?;

        let service = super::Service::new(
            host_id,
            synchronization,
            genotypes_ro,
            evaluations_ro,
            evaluations_wr,
        );

        Ok(Arc::new(service))
    })
}

pub fn register(c: &mut Container) {
    c.provide(evaluations::registrations::provide_evaluations_repository_ro);
    c.provide(evaluations::registrations::provide_evaluations_repository_wr);
    c.provide(provide_evaluation_service);
}
