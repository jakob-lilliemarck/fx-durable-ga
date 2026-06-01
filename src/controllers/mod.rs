use crate::bootstrap::App;
use aide::axum::ApiRouter;
use std::sync::Arc;

pub mod genotypes_embeddings;
pub mod lineage;
pub(super) mod models;
pub mod optimizations;
pub mod requests;

// Re-export request and response types for use by API clients
pub use genotypes_embeddings::BackfillGenotypeEmbeddingsRequest;
pub use requests::{CreateRequestPayload, CreateRequestResponse};

pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .merge(genotypes_embeddings::router(app.clone()))
        .merge(lineage::router(app.clone()))
        .merge(optimizations::router(app.clone()))
        .merge(requests::router(app))
}
