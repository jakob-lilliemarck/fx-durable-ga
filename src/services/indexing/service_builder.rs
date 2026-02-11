use crate::repositories::{embeddings, encoders};
use std::sync::Arc;

pub struct ServiceBuilder {
    pub(super) embeddings: Arc<embeddings::Repository>,
    pub(super) encoders: Arc<encoders::Repository>,
}

impl ServiceBuilder {
    pub fn build(self) -> super::Service {
        super::Service {
            embeddings: self.embeddings,
            encoders: self.encoders,
        }
    }
}
