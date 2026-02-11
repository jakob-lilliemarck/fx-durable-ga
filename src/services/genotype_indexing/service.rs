use crate::{
    GenotypesFilter,
    chainable::{Chain, ToTx},
    models::GenotypeIndexer,
    repositories::{
        encoders::{self, EncoderPairing},
        genotypes,
    },
    services::{
        genotype_indexing::{ServiceBuilder, jobs::IndexGenotypesMessage},
        indexing,
    },
};
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use sqlx::PgTransaction;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

pub struct Service {
    pub(super) indexing: Arc<indexing::Service>,
    pub(super) genotypes: Arc<genotypes::Repository>,
    pub(super) encoders: Arc<encoders::Repository>,
    pub(super) indexers: HashMap<i32, Box<dyn GenotypeIndexer>>,
    pub(super) mq_queries: Arc<Queries>,
}

const GENOTYPES_PAGE_LIMIT: i64 = 1_000;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupingKey {
    encoder_id: Uuid,
    type_hash: i32,
    type_name: String,
}

impl Service {
    pub(crate) fn builder(
        genotypes: Arc<genotypes::Repository>,
        encoders: Arc<encoders::Repository>,
        indexing: Arc<indexing::Service>,
        mq_queries: Arc<Queries>,
    ) -> super::ServiceBuilder {
        ServiceBuilder {
            indexing,
            genotypes,
            encoders,
            mq_queries,
            indexers: HashMap::new(),
        }
    }

    /// Index filtered genotypes.
    /// Groups genotypes by tags and dispatches a indexing job for each
    pub(super) async fn index_genotypes(
        &self,
        filter: &GenotypesFilter,
    ) -> Result<(), super::Error> {
        let mut groups = HashMap::new();

        loop {
            let genotypes = self
                .genotypes
                .search_genotypes(filter, GENOTYPES_PAGE_LIMIT)
                .await?;
            let count = genotypes.len();

            let type_hashes = genotypes
                .iter()
                .map(|(g, _)| g.type_hash())
                .collect::<Vec<_>>();
            let pairings = self.encoders.get_encoder_pairings(&type_hashes).await?;
            let pairings_by_type_hash = EncoderPairing::group_by_type_hash(pairings);

            for (g, _) in genotypes {
                let type_hash = g.type_hash();
                let encoder_ids = match pairings_by_type_hash.get(&type_hash) {
                    Some(encoder_ids) => encoder_ids,
                    None => {
                        // This genotype has no encoder configured
                        // Skip to next iteration
                        //
                        // NOTE!
                        // This filtering could perhaps be done in the database,
                        // But not without more coupling between encoders and genotypes.
                        //
                        // If this becomes a bottleneck an option could be to keep a table
                        // in the genotype repository tracking which genotypes has a toggled encoder
                        // This service would then register an event listener for toggling events
                        continue;
                    }
                };

                // Group genotypes by encoder pairings.
                // GenotypeIndexers are accessed by type_hash and load datasets.
                // Encoders are accessed by ID and load model weight matrices.
                // Those are expensive operations, which is why one job should target single
                // pairing such that a it will be carried out on a single machine that can
                // effectively utilize caching of models and datasets.
                for id in encoder_ids {
                    let key = GroupingKey {
                        encoder_id: *id,
                        type_hash,
                        type_name: g.type_name().to_owned(),
                    };
                    groups.entry(key).or_insert(Vec::new()).push(g.id());
                }
            }

            if count < GENOTYPES_PAGE_LIMIT as usize {
                break;
            }
        }

        let jobs = groups
            .drain()
            .map(|(grouping_key, genotype_ids)| IndexGenotypesMessage {
                grouping_key,
                genotype_ids,
            })
            .collect::<Vec<_>>();

        self.genotypes
            .chain(|tx| {
                Box::pin(async move {
                    let mut publisher =
                        fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx.tx(), &self.mq_queries);

                    for job in jobs {
                        // FIXME!
                        //
                        // This is terrible! Add a `publish_many` method to the publisher!!
                        if let Err(error) = publisher.publish(&job).await {
                            tracing::error!(
                                message = "Error encountered processing IndexGenotypes",
                                error = error.to_string()
                            );
                            break;
                        };
                    }

                    let tx: PgTransaction<'_> = publisher.into();
                    Ok((tx, ()))
                })
            })
            .await?;

        Ok(())
    }

    pub(super) async fn index_genotype_group(
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

        let inputs = genotypes.iter().fold(
            Vec::with_capacity(genotype_ids.len()),
            |mut acc, (g, ..)| {
                acc.push(indexer.input(&g));
                acc
            },
        );

        self.indexing
            .index_many(
                &grouping_key.encoder_id,
                &inputs,
                &Self::format_tags(&grouping_key, &indexer.context_hash()),
            )
            .await?;

        Ok(Some(()))
    }

    pub(super) async fn index_genotype(&self, genotype_id: &Uuid) -> Result<(), super::Error> {
        let filter = GenotypesFilter::default().with_genotype_id(*genotype_id);
        self.index_genotypes(&filter).await?;
        Ok(())
    }

    fn format_tags<'a>(grouping_key: &'a GroupingKey, context_hash: &str) -> [String; 5] {
        [
            "type:Genotype".to_string(),
            format!("encoder_id:{}", grouping_key.encoder_id),
            format!("type_name:{}", grouping_key.type_name),
            format!("type_hash:{}", grouping_key.type_hash),
            format!("context_hash:{}", context_hash),
        ]
    }
}
