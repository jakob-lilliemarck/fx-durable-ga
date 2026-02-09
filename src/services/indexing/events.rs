use crate::services::indexing;
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

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
// Registration
// ============================================================

/// Registers all optimization event handlers with the event bus registry.
#[instrument(level = "debug", skip_all)]
pub fn register_event_handlers(
    queries: Arc<Queries>,
    indexing: Arc<indexing::Service>,
    registry: &mut fx_event_bus::EventHandlerRegistry,
) {
    // No handlers yet
}
