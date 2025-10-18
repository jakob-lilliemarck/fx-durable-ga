use super::jobs::{
    EvaluateGenotypeMessage, GenerateInitialPopulationMessage, MaintainPopulationMessage,
};
use fx_event_bus::Handler;
use fx_mq_building_blocks::queries::Queries;
use serde::{Deserialize, Serialize};
use sqlx::PgTransaction;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// OptimizationRequested
// ============================================================
#[derive(Clone, Serialize, Deserialize)]
pub struct OptimizationRequestedEvent {
    request_id: Uuid,
}

impl fx_event_bus::Event for OptimizationRequestedEvent {
    const NAME: &'static str = "OptimizationRequested";
}

impl OptimizationRequestedEvent {
    #[instrument(level = "debug", fields(request_id = %request_id))]
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

pub struct OptimizationRequestedHandler {
    queries: Arc<Queries>,
}

impl Handler<OptimizationRequestedEvent> for OptimizationRequestedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "info", skip(self, input, tx), fields(request_id = %input.request_id))]
    fn handle<'a>(
        &'a self,
        input: std::sync::Arc<OptimizationRequestedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            let ret = match publisher
                .publish(&GenerateInitialPopulationMessage {
                    request_id: input.request_id,
                })
                .await
            {
                Err(err) => {
                    tracing::error!(
                        message = "Failed to publish GenerateInitialPopulation",
                        request_id = input.request_id.to_string()
                    );
                    Err(err)
                }
                _ => Ok(()),
            };

            (publisher.into(), ret)
        })
    }
}

// ============================================================
// GenotypeGenerated
// ============================================================
#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeGenerated {
    request_id: Uuid,
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeGenerated {
    const NAME: &'static str = "GenotypeGenerated";
}

impl GenotypeGenerated {
    #[instrument(level = "debug", fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub fn new(request_id: Uuid, genotype_id: Uuid) -> Self {
        Self {
            request_id,
            genotype_id,
        }
    }
}

pub struct GenotypeGeneratedHandlerEvent {
    queries: Arc<Queries>,
}

impl Handler<GenotypeGenerated> for GenotypeGeneratedHandlerEvent {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "info", skip(self, input, tx), fields(request_id = %input.request_id, genotype_id = %input.genotype_id))]
    fn handle<'a>(
        &'a self,
        input: Arc<GenotypeGenerated>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            let ret = match publisher
                .publish(&EvaluateGenotypeMessage {
                    request_id: input.request_id,
                    genotype_id: input.genotype_id,
                })
                .await
            {
                Err(err) => {
                    tracing::error!(
                        message = "Failed to publish EvaluateGenotype",
                        request_id = input.request_id.to_string(),
                        genotype_id = input.genotype_id.to_string()
                    );
                    Err(err)
                }
                _ => Ok(()),
            };

            (publisher.into(), ret)
        })
    }
}

// ============================================================
// GenotypeEvaluated
// ============================================================
#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeEvaluatedEvent {
    request_id: Uuid,
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeEvaluatedEvent {
    const NAME: &'static str = "GenotypeEvaluated";
}

impl GenotypeEvaluatedEvent {
    #[instrument(level = "debug", fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub fn new(request_id: Uuid, genotype_id: Uuid) -> Self {
        Self {
            request_id,
            genotype_id,
        }
    }
}

pub struct GenotypeEvaluatedHandler {
    queries: Arc<Queries>,
}

impl Handler<GenotypeEvaluatedEvent> for GenotypeEvaluatedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "info", skip(self, input, tx), fields(request_id = %input.request_id, genotype_id = %input.genotype_id))]
    fn handle<'a>(
        &'a self,
        input: Arc<GenotypeEvaluatedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            let ret = match publisher
                .publish(&MaintainPopulationMessage {
                    request_id: input.request_id,
                })
                .await
            {
                Err(err) => {
                    tracing::error!(
                        message = "Failed to publish MaintainPopulation",
                        request_id = input.request_id.to_string(),
                    );
                    Err(err)
                }
                _ => Ok(()),
            };

            (publisher.into(), ret)
        })
    }
}

// ============================================================
// RequestCompleted
// ============================================================
#[derive(Clone, Serialize, Deserialize)]
pub struct RequestCompletedEvent {
    request_id: Uuid,
}

impl fx_event_bus::Event for RequestCompletedEvent {
    const NAME: &'static str = "RequestCompleted";
}

impl RequestCompletedEvent {
    #[instrument(level = "debug", fields(request_id = %request_id))]
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

pub struct RequestCompletedHandler;

impl Handler<RequestCompletedEvent> for RequestCompletedHandler {
    type Error = super::Error;

    #[instrument(level = "info", skip(self, input, tx), fields(request_id = %input.request_id))]
    fn handle<'a>(
        &'a self,
        input: Arc<RequestCompletedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            tracing::info!(
                message = "Request completed",
                request_id = input.request_id.to_string()
            );
            (tx, Ok(()))
        })
    }
}

// ============================================================
// RequestTerminated
// ============================================================
#[derive(Clone, Serialize, Deserialize)]
pub struct RequestTerminatedEvent {
    request_id: Uuid,
}

impl fx_event_bus::Event for RequestTerminatedEvent {
    const NAME: &'static str = "RequestTerminated";
}

impl RequestTerminatedEvent {
    #[instrument(level = "debug", fields(request_id = %request_id))]
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

pub struct RequestTerminatedHandler;

impl Handler<RequestTerminatedEvent> for RequestTerminatedHandler {
    type Error = super::Error;

    #[instrument(level = "info", skip(self, input, tx), fields(request_id = %input.request_id))]
    fn handle<'a>(
        &'a self,
        input: Arc<RequestTerminatedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            tracing::info!(
                message = "Request terminated",
                request_id = input.request_id.to_string()
            );
            (tx, Ok(()))
        })
    }
}

// ============================================================
// Registration
// ============================================================

pub fn register_event_handlers(
    queries: Arc<Queries>,
    registry: &mut fx_event_bus::EventHandlerRegistry,
) {
    registry.with_handler(OptimizationRequestedHandler {
        queries: queries.clone(),
    });

    registry.with_handler(GenotypeGeneratedHandlerEvent {
        queries: queries.clone(),
    });

    registry.with_handler(GenotypeEvaluatedHandler {
        queries: queries.clone(),
    });

    registry.with_handler(RequestCompletedHandler);

    registry.with_handler(RequestTerminatedHandler);
}
