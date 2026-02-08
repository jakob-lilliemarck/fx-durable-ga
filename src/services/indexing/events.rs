use crate::services::{indexing, optimization::GenotypeEvaluatedEvent};
use fx_event_bus::Handler;
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

// ============================================================
// GenotypeIndexed
// ============================================================

/// Event published when a genotype has been indexed
#[derive(Clone, Serialize, Deserialize)]
pub struct GenotypeIndexedEvent {
    genotype_id: Uuid,
}

impl fx_event_bus::Event for GenotypeIndexedEvent {
    const NAME: &'static str = "GenotypeIndexed";
}

// ============================================================
// EncoderTrained
// ============================================================

/// Event published when an encoder has been trained
#[derive(Clone, Serialize, Deserialize)]
pub struct EncoderTrainedEvent {
    encoder_id: Uuid,
}

impl fx_event_bus::Event for EncoderTrainedEvent {
    const NAME: &'static str = "EncoderTrained";
}

// ============================================================
// GenotypeEvaluated - handler only
// ============================================================

/// Handler responding to completed genotype evaluations
pub struct GenotypeEvaluatedHandler {
    queries: Arc<Queries>,
    indexing: Arc<indexing::Service>,
}

impl Handler<GenotypeEvaluatedEvent> for GenotypeEvaluatedHandler {
    type Error = fx_mq_jobs::PublishError;

    #[instrument(level = "debug", skip(self, input, tx))]
    fn handle<'a>(
        &'a self,
        input: Arc<GenotypeEvaluatedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        // Whenever a genotype is evaluated, check:
        // - number of evaluated genotypes for the current request
        // - number of indexed genotypes for the current request
        //
        // Figure out if its time to:
        // - Dispatch an indexing batch job
        // - Re-train a new encoder
        unimplemented!()
    }
}

/// Registers all optimization event handlers with the event bus registry.
#[instrument(level = "debug", skip_all)]
pub fn register_event_handlers(
    queries: Arc<Queries>,
    indexing: Arc<indexing::Service>,
    registry: &mut fx_event_bus::EventHandlerRegistry,
) {
    registry.with_handler(GenotypeEvaluatedHandler {
        queries: queries.clone(),
        indexing: indexing.clone(),
    });
}
