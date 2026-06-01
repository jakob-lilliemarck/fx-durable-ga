use super::repositories::encoders::Digest;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Event published when a new embedding has been created
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCreatedEvent {
    pub embedding_id: Uuid,
    pub tags: Vec<String>,
}

impl fx_event_bus::Event for EmbeddingCreatedEvent {
    const NAME: &'static str = "EmbeddingCreated";
}

/// Event published when the availability status of an encoder has changed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderAvailableEvent {
    pub indexer_id: Digest,
}

impl fx_event_bus::Event for EncoderAvailableEvent {
    const NAME: &'static str = "EncoderAvailable";
}
