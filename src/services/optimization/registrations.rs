use super::optimizer;
use crate::infrastructure::di::{Container, InvokeResult, ProviderResult};
use crate::infrastructure::registrations::{
    ProvidedEventHandlerRegistry, ProvidedJobHandlerRegistry,
};
use crate::services::budgeting::TransactionsRead;
use crate::services::optimization::repositories::requests;
use crate::services::{budgeting, evaluation};
use crate::{configuration, repositories, services};
use futures::future::BoxFuture;
use futures::lock::Mutex;
use fx_mq_jobs::Queries;
use std::sync::Arc;

fn provide_optimization_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let host_id = c.get::<configuration::HostId>().await?;
        let mq = c.get::<Arc<Queries>>().await?;
        let requests_ro = c.get::<requests::Read>().await?;
        let requests_wr = c.get::<requests::Write>().await?;
        let genotypes_ro = c.get::<repositories::genotypes::Read>().await?;
        let genotypes_wr = c.get::<repositories::genotypes::Write>().await?;
        let evaluations_ro = c.get::<crate::services::evaluation::repositories::evaluations::Read>().await?;
        let optimizers = c.get::<Arc<Mutex<optimizer::OptimizerRegistry>>>().await?;
        let locking = c.get::<Arc<services::locking::Service>>().await?;

        let sync = c.get::<Arc<services::synchronization::Service>>().await?;
        let transactions_ro = c.get::<TransactionsRead>().await?;
        let budgeting = c.get::<Arc<budgeting::Service>>().await?;
        let evaluation = c.get::<Arc<evaluation::Service>>().await?;

        let optimization = super::Service::new(
            host_id.value,
            requests_ro,
            requests_wr,
            genotypes_ro,
            genotypes_wr,
            evaluations_ro,
            optimizers,
            locking,
            sync,
            mq,
            transactions_ro,
            budgeting,
            evaluation,
        );

        Ok(Arc::new(optimization))
    })
}

fn provide_optimizer_registry(
    _: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<Mutex<optimizer::OptimizerRegistry>>>> {
    Box::pin(async {
        Ok(Arc::new(
            Mutex::new(optimizer::OptimizerRegistry::default()),
        ))
    })
}

fn invoke_event_handler_registration(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        let provided = c.get::<ProvidedEventHandlerRegistry>().await?;
        let mut lock = provided.lock().await;
        let Some(mut event_handler_registry) = lock.take() else {
            panic!("Could not take event handler registry")
        };

        let optimization = c.get::<Arc<super::Service>>().await?;
        let mq = c.get::<Arc<Queries>>().await?;
        super::events::register_event_handlers(&mut event_handler_registry, &optimization, &mq);

        *lock = Some(event_handler_registry);

        Ok(())
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

pub fn register(c: &mut Container) {
    c.provide(requests::provide_requests_repository_wr);
    c.provide(requests::provide_requests_repository_ro);

    c.provide(provide_optimization_service);
    c.provide(provide_optimizer_registry);

    c.invokable(invoke_event_handler_registration);
    c.invokable(invoke_job_handler_registration);
}
