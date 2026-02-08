use std::{collections::HashMap, sync::Arc};

use crate::{
    models::Indexable,
    repositories::{embeddings, encoders, genotypes},
};

pub struct ServiceBuilder {
    pub(super) genotypes: Arc<genotypes::Repository>,
    pub(super) embeddings: Arc<embeddings::Repository>,
    pub(super) encoders: Arc<encoders::Repository>,
    pub(super) indexable: HashMap<i32, Box<dyn Indexable>>,
}

impl ServiceBuilder {
    pub fn with_indexable<I>(mut self, indexable: I) -> Self
    where
        I: Indexable + 'static,
    {
        self.indexable.insert(indexable.hash(), Box::new(indexable));
        self
    }

    pub fn build(self) -> super::Service {
        super::Service {
            genotypes: self.genotypes,
            embeddings: self.embeddings,
            encoders: self.encoders,
            loaded: None,
            indexable: self.indexable,
        }
    }
}
