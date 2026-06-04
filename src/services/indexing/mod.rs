mod errors;
mod events;
mod indexable;
mod jobs;
mod registrations;
mod repositories;
mod service;

pub mod encoder;

pub use errors::Error;
pub use events::{EmbeddingCreatedEvent, EncoderAvailableEvent};
pub use registrations::register;

pub use repositories::embeddings::SearchAssociatedTagsFilter;
pub use repositories::embeddings::SearchEmbeddingsFilter;
pub use repositories::embeddings::SearchRequestedEmbeddingsFilter;
pub use repositories::embeddings::SearchSimilarEmbeddingsFilter;

pub mod embeddings {
    pub(crate) use super::repositories::embeddings::AggregatedDiversity;
    pub(crate) use super::repositories::embeddings::Read;

    #[cfg(test)]
    pub(crate) use super::repositories::embeddings::EmbeddingNew;

    #[cfg(test)]
    pub(crate) use super::repositories::embeddings::TagNew;

    #[cfg(test)]
    pub(crate) use super::repositories::embeddings::Write;

    #[cfg(test)]
    pub(crate) use super::repositories::embeddings::WriteTx;

    #[cfg(any(test, feature = "test-tools"))]
    pub use super::repositories::embeddings::search_requested_embeddings;

    #[cfg(any(test, feature = "test-tools"))]
    pub use super::repositories::embeddings::RequestedEmbedding;
}
pub use indexable::{EncodeInput, Indexer, IndexerErased, Registry};
pub use repositories::embeddings::Error as EmbeddingsError;
pub use repositories::encoders::{Digest, EncoderDigestError};
pub use service::{Service, TrainModelConfig};

#[cfg(test)]
pub(crate) use repositories::encoders::Encoder;

#[cfg(any(test, feature = "test-tools"))]
pub use repositories::encoders::{get_available_encoder_digests, get_encoder};

#[cfg(test)]
pub mod encoder_queries {
    pub(crate) use super::repositories::encoders::{store_encoder, store_encoder_availability};
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use service::MODEL_FORMAT;
