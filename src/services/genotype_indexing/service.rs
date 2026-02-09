use crate::{
    models::GenotypeIndexer,
    repositories::genotypes,
    services::{
        genotype_indexing::ServiceBuilder,
        indexing::{self, EncodeInput},
    },
};
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

pub struct Service {
    pub(super) indexing: Arc<indexing::Service>,
    pub(super) genotypes: Arc<genotypes::Repository>,
    pub(super) indexers: HashMap<i32, Box<dyn GenotypeIndexer>>,
}

#[derive(PartialEq, Eq, Hash)]
pub struct Key {
    type_hash: i32,
    request_id: Uuid,
    generation_id: i32,
    context: String,
}

impl Service {
    pub fn builder(
        genotypes: Arc<genotypes::Repository>,
        indexing: Arc<indexing::Service>,
    ) -> super::ServiceBuilder {
        ServiceBuilder {
            indexing,
            genotypes,
            indexers: HashMap::new(),
        }
    }

    pub async fn index_genotypes(&self, genotype_ids: &[Uuid]) -> Result<Option<()>, super::Error> {
        let genotypes = self.genotypes.get_genotypes(genotype_ids).await?;

        // NOTE!
        // Ensure genotypes are grouped and indexed by the tagging strategy
        // An alternative approach is to first index with a permissive grouping
        // and then add tags as needed.
        let mut inputs: HashMap<Key, Vec<EncodeInput>> = HashMap::new();
        for genotype in genotypes {
            let type_hash = genotype.type_hash();
            let request_id = genotype.request_id();
            let generation_id = genotype.generation_id();

            let indexer = match self.indexers.get(&type_hash) {
                Some(indexer) => indexer,
                None => return Ok(None),
            };

            let encode_input = indexer.input(&genotype);
            let context = indexer.context_hash();

            let key = Key {
                type_hash,
                request_id: request_id,
                generation_id: generation_id,
                context,
            };

            inputs.entry(key).or_insert(Vec::new()).push(encode_input);
        }

        for (k, v) in inputs {
            let tag_names = &[
                format!("type:{}", k.type_hash),
                format!("request:{}", k.request_id),
                format!("generation:{}", k.generation_id),
                format!("context:{}", k.context),
            ];
            self.indexing.index(&v, tag_names).await?;
        }

        Ok(Some(()))
    }
}
