use crate::{models::GenotypeIndexer, repositories, services};
use std::{collections::HashMap, sync::Arc};

pub struct ServiceBuilder {
    pub(super) indexing: Arc<services::indexing::Service>,
    pub(super) genotypes: Arc<repositories::genotypes::Repository>,
    pub(super) indexers: HashMap<i32, Box<dyn GenotypeIndexer>>,
}

impl ServiceBuilder {
    pub fn with_indexable<I>(mut self, indexer: I) -> Self
    where
        I: GenotypeIndexer + 'static,
    {
        self.indexers.insert(indexer.type_hash(), Box::new(indexer));
        self
    }

    pub fn build(self) -> super::Service {
        super::Service {
            indexing: self.indexing,
            genotypes: self.genotypes,
            indexers: self.indexers,
        }
    }
}
