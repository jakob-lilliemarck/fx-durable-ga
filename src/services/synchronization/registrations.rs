use super::repositories::semaphores;
use super::repositories::semaphores::registrations::provide_semaphores_repository;
use crate::configuration::PollIntervalSeconds;
use crate::infrastructure::di::{Container, InvokeError, InvokeResult, ProviderResult};
use crate::infrastructure::registrations::ProvidedPgMux;
use futures::future::BoxFuture;
use std::{sync::Arc, time::Duration};

fn provide_synchronization_service(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let semaphores = c.get::<Arc<semaphores::Repository>>().await?;

        let poll_interval_seconds = c.get::<PollIntervalSeconds>().await?;

        let synchronization = super::Service::new(
            Some(Duration::from_secs(poll_interval_seconds.value)),
            semaphores,
        );

        Ok(Arc::new(synchronization))
    })
}

fn invoke_semaphore_listening(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        let crate::configuration::EnableListening { value: true } =
            c.get::<crate::configuration::EnableListening>().await?
        else {
            return Ok(());
        };

        let provided_mux = c.get::<ProvidedPgMux>().await?;
        let mut lock = provided_mux.lock().await;
        let Some(mut mux) = lock.take() else {
            panic!("Could not take mux")
        };

        let synchronization = c.get::<Arc<super::Service>>().await?;

        synchronization
            .register(&mut mux)
            .await
            .map_err(|err| InvokeError::new(err))?;

        *lock = Some(mux);

        Ok(())
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_semaphores_repository);
    c.provide(provide_synchronization_service);

    c.invokable(invoke_semaphore_listening);
}
