use crate::infrastructure::di::{Container, InvokeResult, ProviderResult};
use crate::infrastructure::registrations::{
    ProvidedEventHandlerRegistry, ProvidedJobHandlerRegistry,
};
use crate::repositories::genotypes;
use crate::services::indexing;
use futures::future::BoxFuture;
use fx_mq_jobs::Queries;
use std::sync::Arc;

fn provide_indexing_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let embeddings_ro = c.get::<indexing::embeddings::Read>().await?;
        let genotypes_ro = c.get::<genotypes::Read>().await?;
        let genotypes_wr = c.get::<genotypes::Write>().await?;
        let indexing = c.get::<Arc<indexing::Service>>().await?;
        let mq = c.get::<Arc<Queries>>().await?;

        let indexing = super::Service::new(embeddings_ro, genotypes_ro, genotypes_wr, indexing, mq);

        Ok(Arc::new(indexing))
    })
}

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

fn invoke_event_handler_registration(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        let provided = c.get::<ProvidedEventHandlerRegistry>().await?;
        let mut lock = provided.lock().await;
        let Some(mut event_handler_registry) = lock.take() else {
            panic!("Could not take event handler registry")
        };

        let mq = c.get::<Arc<Queries>>().await?;
        super::events::register_event_handlers(&mut event_handler_registry, &mq);

        *lock = Some(event_handler_registry);

        Ok(())
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_indexing_service);

    c.invokable(invoke_event_handler_registration);
    c.invokable(invoke_job_handler_registration);
}
