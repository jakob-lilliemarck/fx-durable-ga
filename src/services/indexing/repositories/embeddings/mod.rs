mod errors;
mod queries;
mod repository;

pub mod registrations;

pub use errors::Error;
pub use queries::SearchAssociatedTagsFilter;
pub use queries::{
    SearchEmbeddingsFilter, SearchRequestedEmbeddingsFilter, SearchSimilarEmbeddingsFilter,
};

#[cfg(any(test, feature = "test-tools"))]
pub use queries::RequestedEmbedding;

#[cfg(not(any(test, feature = "test-tools")))]
pub use queries::RequestedEmbedding;

pub(crate) use queries::AggregatedDiversity;
pub(crate) use queries::Embedding;
pub(crate) use queries::EmbeddingNew;
pub(crate) use queries::EmbeddingValue;
pub(crate) use queries::TagNew;

pub(crate) use repository::{Read, Write, WriteTx};

#[cfg(any(test, feature = "test-tools"))]
pub use queries::search_requested_embeddings;
