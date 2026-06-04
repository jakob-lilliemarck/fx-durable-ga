use super::Service;
use super::repositories::probes::{self as probes_repo, registrations as probes_registrations};
use crate::infrastructure::{
    db,
    di::{Container, InvokeResult, ProviderResult},
    registrations::ProvidedJobHandlerRegistry,
};
use crate::repositories::genotypes;
use crate::services::evaluation;
use futures::future::BoxFuture;
use fx_mq_jobs::Queries;
use std::sync::Arc;

fn provide_noise_diagnostics_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<Service>>> {
    Box::pin(async {
        let probe_wr = c.get::<probes_repo::Write>().await?;
        let genotypes_ro = c.get::<genotypes::Read>().await?;
        let evaluation = c.get::<Arc<evaluation::Service>>().await?;
        let mq = c.get::<Arc<Queries>>().await?;
        let svc = Service::new(probe_wr, genotypes_ro, evaluation, mq);
        Ok(Arc::new(svc))
    })
}

fn invoke_job_handler_registration(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        let provided = c.get::<ProvidedJobHandlerRegistry>().await?;
        let mut lock = provided.lock().await;
        let Some(job_handler_registry) = lock.take() else {
            panic!("Could not take job handler registry")
        };
        let svc = c.get::<Arc<Service>>().await?;
        let job_handler_registry = super::jobs::register_job_handlers(job_handler_registry, &svc);
        *lock = Some(job_handler_registry);
        Ok(())
    })
}

pub fn register(c: &mut Container) {
    c.provide(probes_registrations::provide_probes_repository_ro);
    c.provide(probes_registrations::provide_probes_repository_wr);
    c.provide(provide_noise_diagnostics_service);
    c.invokable(invoke_job_handler_registration);
}
