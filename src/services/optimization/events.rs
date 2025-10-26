use super::jobs::{
    EvaluateGenotypeMessage, GenerateInitialPopulationMessage, MaintainPopulationMessage,
};
use crate::{
    models::{Conclusion, RequestConclusion},
    services::optimization,
};
use chrono::Utc;
use fx_event_bus::Handler;
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use sqlx::PgTransaction;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// OptimizationRequested
// ============================================================

/// Event published when a new optimization request is created.
#[derive(Clone, Serialize, Deserialize)]
pub struct OptimizationRequestedEvent {
    request_id: Uuid,
}

impl fx_event_bus::Event for OptimizationRequestedEvent {
    const NAME: &'static str = "OptimizationRequested";
}

impl OptimizationRequestedEvent {
    /// Creates a new optimization requested event.
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

/// Handler that responds to optimization requests by scheduling initial population generation.
pub struct OptimizationRequestedHandler {
    queries: Arc<Queries>,
}

impl Handler<OptimizationRequestedEvent> for OptimizationRequestedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "debug", skip(self, input, tx), fields(request_id = %input.request_id))]
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

/// Event published when a new genotype is generated for evaluation.
#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeGenerated {
    request_id: Uuid,
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeGenerated {
    const NAME: &'static str = "GenotypeGenerated";
}

impl GenotypeGenerated {
    /// Creates a new genotype generated event.
    pub fn new(request_id: Uuid, genotype_id: Uuid) -> Self {
        Self {
            request_id,
            genotype_id,
        }
    }
}

/// Handler that responds to genotype generation by scheduling evaluation jobs.
pub struct GenotypeGeneratedHandlerEvent {
    queries: Arc<Queries>,
}

impl Handler<GenotypeGenerated> for GenotypeGeneratedHandlerEvent {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "debug", skip(self, input, tx), fields(request_id = %input.request_id, genotype_id = %input.genotype_id))]
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

/// Event published when a genotype's fitness has been evaluated.
#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeEvaluatedEvent {
    request_id: Uuid,
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeEvaluatedEvent {
    const NAME: &'static str = "GenotypeEvaluated";
}

impl GenotypeEvaluatedEvent {
    /// Creates a new genotype evaluated event.
    pub fn new(request_id: Uuid, genotype_id: Uuid) -> Self {
        Self {
            request_id,
            genotype_id,
        }
    }
}

/// Handler that responds to genotype evaluations by scheduling population maintenance.
pub struct GenotypeEvaluatedHandler {
    queries: Arc<Queries>,
}

impl Handler<GenotypeEvaluatedEvent> for GenotypeEvaluatedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "debug", skip(self, input, tx), fields(request_id = %input.request_id, genotype_id = %input.genotype_id))]
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

/// Event published when an optimization request reaches its fitness goal.
#[derive(Clone, Serialize, Deserialize)]
pub struct RequestCompletedEvent {
    request_id: Uuid,
}

impl fx_event_bus::Event for RequestCompletedEvent {
    const NAME: &'static str = "RequestCompleted";
}

impl RequestCompletedEvent {
    /// Creates a new request completed event.
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

/// Handler that concludes optimization requests when they complete successfully.
pub struct RequestCompletedHandler {
    optimization: Arc<optimization::Service>,
}

impl Handler<RequestCompletedEvent> for RequestCompletedHandler {
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, input, tx), fields(request_id = %input.request_id))]
    fn handle<'a>(
        &'a self,
        input: Arc<RequestCompletedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        let optimization = self.optimization.clone();

        Box::pin(async move {
            if let Err(err) = optimization
                .conclude_request(RequestConclusion {
                    request_id: input.request_id,
                    concluded_at: Utc::now(),
                    concluded_with: Conclusion::Completed,
                })
                .await
            {
                tracing::error!(message = "Could not conclude request", error = ?err)
            }

            (tx, Ok(()))
        })
    }
}

// ============================================================
// RequestTerminated
// ============================================================

/// Event published when an optimization request is terminated before completion.
#[derive(Clone, Serialize, Deserialize)]
pub struct RequestTerminatedEvent {
    request_id: Uuid,
}

impl fx_event_bus::Event for RequestTerminatedEvent {
    const NAME: &'static str = "RequestTerminated";
}

impl RequestTerminatedEvent {
    /// Creates a new request terminated event.
    pub fn new(request_id: Uuid) -> Self {
        Self { request_id }
    }
}

/// Handler that concludes optimization requests when they are terminated early.
pub struct RequestTerminatedHandler {
    optimization: Arc<optimization::Service>,
}

impl Handler<RequestTerminatedEvent> for RequestTerminatedHandler {
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, input, tx), fields(request_id = %input.request_id))]
    fn handle<'a>(
        &'a self,
        input: Arc<RequestTerminatedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            let optimization = self.optimization.clone();

            if let Err(err) = optimization
                .conclude_request(RequestConclusion {
                    request_id: input.request_id,
                    concluded_at: Utc::now(),
                    concluded_with: Conclusion::Terminated,
                })
                .await
            {
                tracing::error!(message = "Could not conclude request", error = ?err)
            }

            (tx, Ok(()))
        })
    }
}

// ============================================================
// Registration
// ============================================================

/// Registers all optimization event handlers with the event bus registry.
#[instrument(level = "debug", skip_all)]
pub fn register_event_handlers(
    queries: Arc<Queries>,
    optimization: Arc<optimization::Service>,
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

    registry.with_handler(RequestCompletedHandler {
        optimization: optimization.clone(),
    });

    registry.with_handler(RequestTerminatedHandler {
        optimization: optimization.clone(),
    });
}
