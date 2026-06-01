use crate::infrastructure::di::{Container, ProviderResult};
use crate::repositories::genotypes;
use futures::future::BoxFuture;
use std::sync::Arc;

fn provide_genotype_exploration_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let genotypes_ro = c.get::<genotypes::Read>().await?;

        let genotype_exploration = super::Service::new(genotypes_ro);

        Ok(Arc::new(genotype_exploration))
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_genotype_exploration_service);
}
