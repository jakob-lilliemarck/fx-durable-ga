use std::sync::Arc;

use crate::services::evaluation::repositories::evaluations;
use crate::services::indexing::embeddings;
pub(crate) use crate::services::optimization::Read;
pub mod genotypes;
pub mod noise_diagnostics;
pub mod ordering;

#[derive(Clone)]
pub struct Provider {
    pub locking: Arc<crate::services::locking::Service>,
    pub requests_ro: Read,
    pub genotypes_ro: genotypes::Read,
    pub evaluations_ro: evaluations::Read,
    pub embeddings_ro: embeddings::Read,
}

impl Provider {
    pub fn lock(&self) -> &Arc<crate::services::locking::Service> {
        &self.locking
    }

    pub fn requests(&self) -> &Read {
        &self.requests_ro
    }

    pub fn genotypes(&self) -> &genotypes::Read {
        &self.genotypes_ro
    }

    pub fn evaluations(&self) -> &evaluations::Read {
        &self.evaluations_ro
    }

    pub fn embeddings(&self) -> &embeddings::Read {
        &self.embeddings_ro
    }
}
