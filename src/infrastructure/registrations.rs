use super::di::{Container, ProviderResult};
use crate::configuration::{DatabaseReadUrl, DatabaseWriteUrl};
use futures::{future::BoxFuture, lock::Mutex};
use fx_event_bus::EventHandlerRegistry;
use fx_mq_jobs::{FX_MQ_JOBS_SCHEMA_NAME, Queries};
use fx_pgmux::Multiplexer;
use sqlx::postgres::PgPoolOptions;
use std::sync::Arc;

pub type ProvidedMqQueries = Arc<Queries>;
pub type ProvidedPgMux = Arc<Mutex<Option<Multiplexer>>>;
pub type ProvidedEventHandlerRegistry = Arc<Mutex<Option<EventHandlerRegistry>>>;
pub type ProvidedJobHandlerRegistry = Arc<Mutex<Option<fx_mq_jobs::RegistryBuilder>>>;

fn provide_read_pool(
    c: &mut super::di::Container,
) -> BoxFuture<'_, ProviderResult<super::db::ReadPool>> {
    Box::pin(async {
        let database_url = c.get::<DatabaseReadUrl>().await?;

        let pool = PgPoolOptions::new()
            .max_connections(3)
            .connect(&database_url.value)
            .await
            .map_err(|err| {
                super::di::ProviderError::new::<super::db::ReadPool, _>(Box::new(err))
            })?;

        Ok(super::db::ReadPool { pool })
    })
}

fn provide_write_pool(
    c: &mut super::di::Container,
) -> BoxFuture<'_, ProviderResult<super::db::WritePool>> {
    Box::pin(async {
        let database_url = c.get::<DatabaseWriteUrl>().await?;

        let pool = PgPoolOptions::new()
            .max_connections(3)
            .connect(&database_url.value)
            .await
            .map_err(|err| {
                super::di::ProviderError::new::<super::db::WritePool, _>(Box::new(err))
            })?;

        Ok(super::db::WritePool { pool })
    })
}

fn provide_mq_queries(_: &mut super::di::Container) -> BoxFuture<'_, ProviderResult<Arc<Queries>>> {
    Box::pin(async { Ok(Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME))) })
}

fn provide_pg_mux(c: &mut super::di::Container) -> BoxFuture<'_, ProviderResult<ProvidedPgMux>> {
    Box::pin(async {
        let ro = c.get::<super::db::ReadPool>().await?;

        let mux = Multiplexer::new(&ro.pool)
            .await
            .map_err(|err| super::di::ProviderError::new::<Multiplexer, _>(Box::new(err)))?;

        // Option is used to allow taking the owned value out after construction!
        Ok(Arc::new(Mutex::new(Some(mux))))
    })
}

fn provide_event_handler_registry(
    _: &mut Container,
) -> BoxFuture<'_, ProviderResult<ProvidedEventHandlerRegistry>> {
    Box::pin(async { Ok(Arc::new(Mutex::new(Some(EventHandlerRegistry::new())))) })
}

fn provide_job_handler_registry(
    _: &mut Container,
) -> BoxFuture<'_, ProviderResult<ProvidedJobHandlerRegistry>> {
    Box::pin(async {
        Ok(Arc::new(Mutex::new(Some(
            fx_mq_jobs::RegistryBuilder::new(),
        ))))
    })
}

pub fn register(c: &mut Container) {
    c.provide(provide_event_handler_registry);
    c.provide(provide_job_handler_registry);

    c.provide(provide_read_pool);
    c.provide(provide_write_pool);

    c.provide(provide_pg_mux);
    c.provide(provide_mq_queries);
}
