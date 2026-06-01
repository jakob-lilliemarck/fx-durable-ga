use std::sync::Arc;
pub(crate) mod locking;

pub mod budgeting;
pub mod evaluation;
pub mod genotype_explorer;
pub mod genotype_indexing;
pub mod indexing;
pub mod optimization;
pub mod synchronization;

pub mod events {
    pub use super::evaluation::GenotypeEvaluatedEvent;
    pub use super::indexing::EmbeddingCreatedEvent;
    pub use super::indexing::EncoderAvailableEvent;
}

#[derive(Clone)]
pub struct Provider {
    pub optimization: Arc<optimization::Service>,
    pub indexing: Arc<indexing::Service>,
    pub genotype_explorer: Arc<genotype_explorer::Service>,
    pub genotype_indexing: Arc<genotype_indexing::Service>,
    pub synchronization: Arc<synchronization::Service>,
}

impl Provider {
    pub fn indexing(&self) -> &Arc<indexing::Service> {
        &self.indexing
    }

    pub fn genotype_indexing(&self) -> &Arc<genotype_indexing::Service> {
        &self.genotype_indexing
    }

    pub fn genotype_explorer(&self) -> &Arc<genotype_explorer::Service> {
        &self.genotype_explorer
    }

    pub fn optimization(&self) -> &Arc<optimization::Service> {
        &self.optimization
    }

    pub fn synchronization(&self) -> &Arc<synchronization::Service> {
        &self.synchronization
    }
}
