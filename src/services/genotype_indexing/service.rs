use crate::{
    GenotypesFilter,
    chainable::{Chain, FromTx},
    models::GenotypeIndexer,
    repositories::genotypes,
    services::{
        genotype_indexing::{ServiceBuilder, events::GroupCreatedEvent},
        indexing,
    },
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

pub struct Service {
    pub(super) indexing: Arc<indexing::Service>,
    pub(super) genotypes: Arc<genotypes::Repository>,
    pub(super) indexers: HashMap<i32, Box<dyn GenotypeIndexer>>,
}

const GENOTYPES_PAGE_LIMIT: i64 = 1_000;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupingKey {
    type_hash: i32,
    request_id: Uuid,
    generation_id: i32,
    context_hash: String,
}

impl Service {
    pub(crate) fn builder(
        genotypes: Arc<genotypes::Repository>,
        indexing: Arc<indexing::Service>,
    ) -> super::ServiceBuilder {
        ServiceBuilder {
            indexing,
            genotypes,
            indexers: HashMap::new(),
        }
    }

    pub async fn group_genotypes(&self, filter: &GenotypesFilter) -> Result<(), super::Error> {
        let mut groups = HashMap::new();

        loop {
            let genotypes = self
                .genotypes
                .search_genotypes(filter, GENOTYPES_PAGE_LIMIT)
                .await?;
            let count = genotypes.len();

            for (g, _) in genotypes {
                let type_hash = g.type_hash();
                let request_id = g.request_id();
                let generation_id = g.generation_id();

                let indexer = match self.indexers.get(&type_hash) {
                    Some(indexer) => indexer,
                    None => continue,
                };

                let key = GroupingKey {
                    type_hash,
                    request_id,
                    generation_id,
                    context_hash: indexer.context_hash(),
                };

                groups.entry(key).or_insert(Vec::new()).push(g.id());
            }
            if count < GENOTYPES_PAGE_LIMIT as usize {
                break;
            }
        }

        let events = groups
            .drain()
            .map(|(grouping_key, genotype_ids)| GroupCreatedEvent {
                grouping_key,
                genotype_ids,
            })
            .collect::<Vec<_>>();

        self.genotypes
            .chain(|tx| {
                Box::pin(async move {
                    let mut publisher = fx_event_bus::Publisher::from_tx(tx);
                    publisher.publish_many(&events).await?;
                    Ok((publisher, ()))
                })
            })
            .await?;

        Ok(())
    }

    pub async fn index_genotypes(
        &self,
        grouping_key: &GroupingKey,
        genotype_ids: &[Uuid],
    ) -> Result<Option<()>, super::Error> {
        let mut filter = GenotypesFilter::default();
        for id in genotype_ids {
            filter = filter.with_genotype_id(*id);
        }

        let genotypes = self
            .genotypes
            .search_genotypes(&filter, genotype_ids.len() as i64)
            .await?;

        let indexer = match self.indexers.get(&grouping_key.type_hash) {
            Some(indexer) => indexer,
            None => return Ok(None),
        };

        let mut inputs = Vec::with_capacity(genotype_ids.len());
        for (genotype, _) in genotypes {
            inputs.push(indexer.input(&genotype));
        }

        let tags = &[
            format!("type_hash:{}", grouping_key.type_hash),
            format!("request_id:{}", grouping_key.request_id),
            format!("generation_id:{}", grouping_key.generation_id),
            format!("context_hash:{}", grouping_key.context_hash),
        ];

        self.indexing.index(&inputs, tags).await?;

        Ok(Some(()))
    }
}
