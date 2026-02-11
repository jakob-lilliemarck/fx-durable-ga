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
// EmbeddingCreated
// ============================================================

/// Event published when an encoder has been trained
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCreatedEvent {
    pub embedding_id: Uuid,
    pub tags: Vec<String>,
}

impl fx_event_bus::Event for EmbeddingCreatedEvent {
    const NAME: &'static str = "EmbeddingCreated";
}

// ============================================================
// EncoderPairingToggled
// ============================================================

/// Event fired whenever a encoder pairing is enabled or disabled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderPairingToggledEvent {
    pub encoder_id: Uuid,
    pub type_hash: i32,
    pub is_enabled: bool,
}

impl fx_event_bus::Event for EncoderPairingToggledEvent {
    const NAME: &'static str = "EncoderPairingToggled";
}

// ============================================================
// Registration
// ============================================================

/// Registers all optimization event handlers with the event bus registry.
#[instrument(level = "debug", skip_all)]
pub fn register_event_handlers(
    _: Arc<Queries>,
    _: Arc<indexing::Service>,
    _: &mut fx_event_bus::EventHandlerRegistry,
) {
    // No handlers yet
}
