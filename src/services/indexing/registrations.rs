use super::indexable;
use super::repositories::embeddings;
use super::repositories::encoders;
use crate::infrastructure::di::{Container, InvokeResult, ProviderResult};
use crate::infrastructure::registrations::ProvidedJobHandlerRegistry;
use crate::services::indexing::repositories::embeddings::registrations::{
    provide_embeddings_repository_ro, provide_embeddings_repository_wr,
};
use crate::services::indexing::repositories::encoders::{
    provide_encoders_repository_ro, provide_encoders_repository_wr,
};
use crate::services::locking;
use futures::{future::BoxFuture, lock::Mutex};
use fx_mq_jobs::Queries;
use std::sync::Arc;
use tracing::instrument;

#[instrument(level = "info", skip_all)]
fn provide_indexing_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let embeddings_ro = c.get::<embeddings::Read>().await?;
        let embeddings_wr = c.get::<embeddings::Write>().await?;
        let encoders_ro = c.get::<encoders::Read>().await?;
        let encoders_wr = c.get::<encoders::Write>().await?;
        let registry = c.get::<Arc<Mutex<indexable::Registry>>>().await?;
        let locking = c.get::<Arc<locking::Service>>().await?;
        let mq = c.get::<Arc<fx_mq_jobs::Queries>>().await?;

        let indexing = super::Service::new(
            embeddings_ro,
            embeddings_wr,
            encoders_ro,
            encoders_wr,
            registry,
            locking,
            mq,
        );

        Ok(Arc::new(indexing))
    })
}

fn provide_indexer_registry(
    _: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<Mutex<indexable::Registry>>>> {
    Box::pin(async { Ok(Arc::new(Mutex::new(indexable::Registry::default()))) })
}

#[instrument(level = "info", skip_all)]
fn invoke_job_handler_registration(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        let provided = c.get::<ProvidedJobHandlerRegistry>().await?;
        let mut lock = provided.lock().await;
        let Some(job_handler_registry) = lock.take() else {
            panic!("Could not take job handler registry")
        };

        let indexing = c.get::<Arc<super::Service>>().await?;

        let mq = c.get::<Arc<Queries>>().await?;

        let job_handler_registry =
            super::jobs::register_job_handlers(job_handler_registry, &indexing, &mq);

        *lock = Some(job_handler_registry);

        Ok(())
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_encoders_repository_ro);
    c.provide(provide_encoders_repository_wr);

    c.provide(provide_embeddings_repository_ro);
    c.provide(provide_embeddings_repository_wr);

    c.provide(provide_indexing_service);
    c.provide(provide_indexer_registry);

    c.invokable(invoke_job_handler_registration);
}
