use crate::infrastructure::db::{ReadPool, WritePool};
use crate::infrastructure::di::{Container, InvokeError, InvokeResult, ProviderResult};
use crate::infrastructure::registrations::{
    ProvidedEventHandlerRegistry, ProvidedJobHandlerRegistry, ProvidedPgMux,
};
use crate::services::{self};
use crate::{configuration, repositories};
use futures::future::BoxFuture;
use fx_mq_jobs::Queries;
use std::{net::SocketAddr, sync::Arc};
use tower_http::services::ServeDir;

#[derive(Clone)]
pub struct App {
    pub repositories: repositories::Provider,
    pub services: services::Provider,
    pub mq: Arc<Queries>,
    wr: WritePool,
}

impl App {
    pub fn repositories(&self) -> &repositories::Provider {
        &self.repositories
    }

    pub fn services(&self) -> &services::Provider {
        &self.services
    }

    pub async fn stop(&self) -> anyhow::Result<()> {
        // Raise the shutdown semaphore — in-progress evaluations abort with
        // Err(Aborted) and their jobs are requeued for retry on restart.
        let mut tx = self.wr.pool.begin().await?;
        self.services
            .synchronization
            .raise(&mut tx, crate::services::evaluation::SHUTDOWN_SEMAPHORE)
            .await?;
        tx.commit().await?;

        // Stop the synchronization agent (closes its broadcast channel).
        self.services.synchronization.stop().await?;

        // Stop the optimization service (which also stops its sync agent).
        self.services.optimization.stop().await?;

        Ok(())
    }

    pub async fn serve_http(self: Arc<Self>, addr: SocketAddr) -> anyhow::Result<()> {
        let mut api = aide::openapi::OpenApi {
            info: aide::openapi::Info {
                title: "FX Durable GA API".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                ..aide::openapi::Info::default()
            },
            ..aide::openapi::OpenApi::default()
        };

        let controllers_router = crate::controllers::router(self);

        let docs_router = aide::axum::ApiRouter::new()
            .route(
                "/docs",
                aide::swagger::Swagger::new("/docs/openapi.json").axum_route(),
            )
            .route("/docs/openapi.json", axum::routing::get(serve_openapi));

        let router = docs_router
            .merge(controllers_router)
            .finish_api(&mut api)
            .nest_service(
                "/public",
                axum::routing::get_service(ServeDir::new("public")),
            )
            .layer(axum::Extension(api))
            .into_make_service();

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, router).await?;
        Ok(())
    }
}

async fn serve_openapi(
    axum::Extension(api): axum::Extension<aide::openapi::OpenApi>,
) -> impl aide::axum::IntoApiResponse {
    axum::Json(api)
}

pub fn provide_app(c: &mut Container) -> BoxFuture<'_, ProviderResult<Arc<App>>> {
    Box::pin(async {
        let mq = c.get::<Arc<fx_mq_jobs::Queries>>().await?;
        let wr = c.get::<WritePool>().await?;

        let genotypes_ro = c.get::<super::repositories::genotypes::Read>().await?;
        let requests_ro = c.get::<crate::services::optimization::Read>().await?;
        let evaluations_ro = c.get::<crate::services::evaluation::repositories::evaluations::Read>().await?;
        let embeddings_ro = c
            .get::<super::services::indexing::embeddings::Read>()
            .await?;
        let locking = c.get::<Arc<super::services::locking::Service>>().await?;
        let repositories = super::repositories::Provider {
            genotypes_ro,
            requests_ro,
            evaluations_ro,
            locking,
            embeddings_ro,
        };

        let synchronization = c
            .get::<Arc<super::services::synchronization::Service>>()
            .await?;

        let indexing = c.get::<Arc<super::services::indexing::Service>>().await?;

        let genotype_indexing = c
            .get::<Arc<super::services::genotype_indexing::Service>>()
            .await?;

        let genotype_explorer = c
            .get::<Arc<super::services::genotype_explorer::Service>>()
            .await?;

        let optimization = c
            .get::<Arc<super::services::optimization::Service>>()
            .await?;

        let services = super::services::Provider {
            synchronization,
            indexing,
            genotype_indexing,
            optimization,
            genotype_explorer,
        };

        let app = super::bootstrap::App {
            services,
            repositories,
            mq,
            wr,
        };

        Ok(Arc::new(app))
    })
}

// Invokes
#[derive(thiserror::Error, Debug)]
#[error("{0}:#")]
pub struct EventListenerError(#[from] anyhow::Error);

fn invoke_event_listening(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        // Conditionally start listeners
        let configuration::EnableListening { value: true } =
            c.get::<configuration::EnableListening>().await?
        else {
            return Ok(());
        };

        let Some(event_handler_registry) = c
            .get::<ProvidedEventHandlerRegistry>()
            .await?
            .lock()
            .await
            .take()
        else {
            panic!("Could not take event handler registry")
        };

        let provided_mux = c.get::<ProvidedPgMux>().await?;
        let mut lock = provided_mux.lock().await;
        let Some(mut mux) = lock.take() else {
            panic!("Could not take mux")
        };

        let ro = c.get::<ReadPool>().await?;

        let mut listener = fx_event_bus::Listener::new(ro.pool, event_handler_registry);

        listener
            .register(&mut mux)
            .await
            .map_err(|err| InvokeError::Invoke {
                error: Box::new(EventListenerError(err)),
            })?;

        *lock = Some(mux);

        tokio::spawn(async move { listener.listen(None).await });

        Ok(())
    })
}

fn invoke_job_listening(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        // Conditionally start listeners
        let configuration::EnableListening { value: true } =
            c.get::<configuration::EnableListening>().await?
        else {
            return Ok(());
        };

        let Some(job_handler_registry) = c
            .get::<ProvidedJobHandlerRegistry>()
            .await?
            .lock()
            .await
            .take()
        else {
            panic!("Could not take job handler registry")
        };

        let provided_mux = c.get::<ProvidedPgMux>().await?;
        let mut lock = provided_mux.lock().await;
        let Some(mut mux) = lock.take() else {
            panic!("Could not take mux")
        };

        let ro = c.get::<ReadPool>().await?;
        let host_id = c.get::<configuration::HostId>().await?;
        let job_worker_count = c.get::<configuration::JobWorkerCount>().await?;

        tracing::info!(
            message = "worker count",
            job_worker_count = job_worker_count.value
        );

        let job_lease_duration = c.get::<configuration::JobLeaseDuration>().await?;

        let mut listener = fx_mq_jobs::Listener::new(
            ro.pool,
            job_handler_registry,
            job_worker_count.value,
            host_id.value,
            job_lease_duration.value,
        )
        .await
        .map_err(|err| InvokeError::Invoke {
            error: Box::new(err),
        })?;

        listener
            .register_with_mux(&mut mux)
            .await
            .map_err(|error| InvokeError::Invoke {
                error: Box::new(error),
            })?;

        *lock = Some(mux);

        tokio::spawn(async move { listener.listen().await });

        Ok(())
    })
}

fn invoke_mux_listening(c: &mut Container) -> BoxFuture<'_, InvokeResult> {
    Box::pin(async {
        // Conditionally start listeners
        let configuration::EnableListening { value: true } =
            c.get::<configuration::EnableListening>().await?
        else {
            return Ok(());
        };

        let Some(mux) = c.get::<ProvidedPgMux>().await?.lock().await.take() else {
            panic!("Could not take mux")
        };

        tokio::spawn(mux.listen());

        Ok(())
    })
}

pub fn register(c: &mut Container) {
    // Configuration variables
    super::configuration::register(c);

    // Plumbing
    super::infrastructure::registrations::register(c);

    // Repositories
    super::repositories::genotypes::register(c);
    super::repositories::noise_diagnostics::register(c);

    // Services
    super::services::locking::register(c);
    super::services::budgeting::register(c);
    super::services::synchronization::register(c);
    super::services::indexing::register(c);
    super::services::genotype_indexing::register(c);
    super::services::genotype_explorer::register(c);
    super::services::optimization::register(c);
    super::services::evaluation::register(c);

    // The application instance
    c.provide(provide_app);

    c.invokable(invoke_event_listening);
    c.invokable(invoke_job_listening);
    c.invokable(invoke_mux_listening);
}
