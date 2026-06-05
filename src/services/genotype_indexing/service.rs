use crate::infrastructure::db;
use crate::repositories::genotypes::{Genotype, Identifiable, TypeName};
use crate::services::indexing::Digest;
use crate::services::indexing::{SearchAssociatedTagsFilter, SearchRequestedEmbeddingsFilter};
use crate::services::optimization::Request;
use crate::{
    SearchGenotypesFilter,
    repositories::genotypes,
    services::{genotype_indexing::jobs::IndexGenotypesMessage, indexing},
};
use fx_mq_building_blocks::queries::Queries;
use fx_mq_jobs::Message;
use serde::{Deserialize, Serialize, Serializer};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tracing::instrument;
use uuid::Uuid;

/// Manages indexing of genotypes into embedding representations.
pub struct Service {
    pub(super) embeddings_ro: indexing::embeddings::Read,
    pub(super) indexing: Arc<indexing::Service>,
    pub(super) genotypes_ro: genotypes::Read,
    // FIXME: genotypes_wr is only used for message queue operations
    // (publishing IndexGenotypesMessage jobs and checking pending jobs),
    // not for any genotypes storage. The dependency on genotypes::Write
    // exists solely to get a write transaction for fx_mq_jobs.
    //
    // The proper fix is to add pool-accepting methods to
    // fx_mq_jobs::Queries (e.g. publish_message, search_pending that
    // take &PgPool directly), removing the need for a transaction
    // from an unrelated repository's write pool.
    pub(super) genotypes_wr: genotypes::Write,
    pub(super) mq: Arc<Queries>,
}

const DB_PAGE_SIZE: usize = 1_000;
const LIMIT: i64 = DB_PAGE_SIZE as i64;
const INDEX_BATCH_SIZE: usize = 100;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupingKey {
    encoder_digest: Digest,
    type_name: String,
}

#[derive(Debug, Deserialize)]
struct IndexableGenotype {
    id: Uuid,
    genome: serde_json::Value,
    type_name: String,
}

impl Serialize for IndexableGenotype {
    #[instrument(level = "debug", skip_all)]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.genome.serialize(serializer)
    }
}

impl Identifiable for IndexableGenotype {
    fn id(&self) -> Uuid {
        self.id
    }
}

impl From<Genotype> for IndexableGenotype {
    #[instrument(level = "debug")]
    fn from(value: Genotype) -> Self {
        Self {
            id: value.id,
            genome: value.genome,
            type_name: value.type_name,
        }
    }
}

impl TypeName for IndexableGenotype {
    fn type_name(&self) -> &str {
        &self.type_name
    }
}

impl Service {
    /// Creates a new genotype indexing service with the given dependencies.
    pub fn new(
        embeddings_ro: indexing::embeddings::Read,
        genotypes_ro: genotypes::Read,
        genotypes_wr: genotypes::Write,
        indexing: Arc<indexing::Service>,
        mq: Arc<Queries>,
    ) -> Self {
        Self {
            embeddings_ro,
            genotypes_ro,
            genotypes_wr,
            indexing,
            mq,
        }
    }

    /// Indexes genotypes by creating embeddings for the given indexer.
    #[instrument(level = "debug", skip(self))]
    pub(super) async fn index_genotypes(
        &self,
        indexer_id: &Digest,
        genotype_ids: &[Uuid],
        request_id: Option<Uuid>,
    ) -> Result<Option<Vec<Uuid>>, super::Error> {
        let mut items = Vec::with_capacity(genotype_ids.len());
        for chunk in genotype_ids.chunks(LIMIT as usize) {
            let filter = chunk
                .iter()
                .fold(SearchGenotypesFilter::default(), |a, id| {
                    a.with_genotype_id(*id)
                });

            let genotypes = self
                .genotypes_ro
                .search_genotypes(&filter, chunk.len() as i64)
                .await?;

            for genotype in genotypes {
                let tags = Self::format_genotype_tags(indexer_id, &genotype);
                items.push((IndexableGenotype::from(genotype), tags))
            }
        }

        let metadata = request_id.map(|id| serde_json::json!({ "request_id": id }));

        self.indexing
            .index_many(indexer_id, &items, metadata)
            .await
            .map_err(Into::into)
    }

    /// Indexes genotypes that are not yet indexed, for all indexers of the matching type.
    #[instrument(level = "debug", skip(self))]
    pub async fn backfill_missing_genotypes(
        &self,
        mut filter: SearchGenotypesFilter,
    ) -> Result<(), super::Error> {
        let mut cursor: Option<Uuid> = None;

        loop {
            if let Some(cursor_id) = cursor {
                filter = filter.with_cursor(cursor_id);
            }

            let genotypes = self.genotypes_ro.search_genotypes(&filter, LIMIT).await?;

            let mut type_indexers: HashMap<String, Vec<Digest>> = HashMap::new();
            let mut counts: HashMap<(Digest, Uuid), usize> = HashMap::new();

            for genotype in &genotypes {
                let indexer_ids = match type_indexers.entry(genotype.type_name().to_string()) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let indexers = self
                            .indexing
                            .get_indexer_ids_of_type(genotype.type_name())
                            .await;
                        entry.insert(indexers)
                    }
                };

                for indexer_id in indexer_ids.iter() {
                    *counts
                        .entry((*indexer_id, genotype.request_id()))
                        .or_insert(0) += 1;
                }
            }

            let mut groups: HashMap<(Digest, Uuid), Vec<Uuid>> =
                HashMap::with_capacity(counts.len());
            for (key, count) in counts {
                groups.insert(key, Vec::with_capacity(count));
            }

            for genotype in &genotypes {
                if let Some(indexer_ids) = type_indexers.get(genotype.type_name()) {
                    for indexer_id in indexer_ids.iter() {
                        if let Some(group) = groups.get_mut(&(*indexer_id, genotype.request_id())) {
                            group.push(genotype.id());
                        }
                    }
                }
            }

            let mut jobs: Vec<IndexGenotypesMessage> = Vec::new();
            for ((indexer_id, request_id), genotype_ids) in groups {
                let lhs_tags = genotype_ids
                    .iter()
                    .map(|id| Self::fmt_tag_genotype_id(*id))
                    .collect::<Vec<_>>();
                let rhs_tags = vec![Self::fmt_tag_indexer_id(&indexer_id)];

                let pairs = self
                    .embeddings_ro
                    .get_tag_pairs_for_embeddings(&lhs_tags, &rhs_tags)
                    .await?;

                let mut indexed_set: HashSet<Uuid> = HashSet::with_capacity(pairs.len());
                for (_embedding_id, lhs_tag, _rhs_tag) in pairs {
                    let Some(id_str) = lhs_tag.strip_prefix("genotype_id:") else {
                        return Err(super::Error::InvalidGenotypeTag {
                            tag: lhs_tag,
                            source: None,
                        });
                    };

                    let id = Uuid::parse_str(id_str).map_err(|source| {
                        super::Error::InvalidGenotypeTag {
                            tag: format!("genotype_id:{id_str}"),
                            source: Some(source),
                        }
                    })?;
                    indexed_set.insert(id);
                }

                let mut pending: Vec<Uuid> = genotype_ids
                    .into_iter()
                    .filter(|id| !indexed_set.contains(id))
                    .collect();

                let jobs_to_add = (pending.len() + INDEX_BATCH_SIZE - 1) / INDEX_BATCH_SIZE;
                jobs.reserve(jobs_to_add);

                while !pending.is_empty() {
                    let batch_len = pending.len().min(INDEX_BATCH_SIZE);
                    let split_at = pending.len() - batch_len;
                    let batch: Vec<Uuid> = pending.split_off(split_at);
                    jobs.push(IndexGenotypesMessage::new(
                        indexer_id,
                        batch,
                        Some(request_id),
                    ));
                }
            }

            if !jobs.is_empty() {
                let mq = self.mq.clone();

                db::begin(self.genotypes_wr.clone(), |tx| {
                    Box::pin(async move {
                        let mut publisher = fx_mq_jobs::Publisher::new_tx(tx, &mq);
                        for chunk in jobs.chunks(DB_PAGE_SIZE) {
                            publisher.publish_many(chunk).await?;
                        }
                        Ok(())
                    })
                })
                .await?;
            }

            cursor = genotypes.last().map(|genotype| genotype.id());
            if genotypes.len() < LIMIT as usize {
                break;
            }
        }

        Ok(())
    }

    /// Retries indexing for genotypes that were deferred due to missing models.
    #[instrument(level = "debug", skip(self))]
    pub(super) async fn retry_deferred_indexation(
        &self,
        indexer_id: &Digest,
    ) -> Result<(), super::Error> {
        let mut jobs: Vec<IndexGenotypesMessage> = Vec::with_capacity(LIMIT as usize);

        let mut filter = SearchRequestedEmbeddingsFilter::default().with_indexer_id(indexer_id);
        loop {
            let deferred = self.indexing.get_deferred_items(&filter, LIMIT).await?;

            // Group deferred items by request_id so each job carries a single request context.
            let mut groups: HashMap<Option<Uuid>, Vec<Uuid>> = HashMap::new();
            for d in deferred.iter() {
                let request_id = d
                    .metadata
                    .get("request_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok());

                groups.entry(request_id).or_default().push(d.entity_id);
            }

            for (request_id, ids) in groups {
                for batch in indexing::Service::batch_iter(ids, INDEX_BATCH_SIZE) {
                    jobs.push(IndexGenotypesMessage::new(*indexer_id, batch, request_id));
                }
            }

            if deferred.len() < LIMIT as usize {
                break;
            }

            let Some(cursor) = deferred.get(LIMIT as usize - 1).map(|d| d.next_cursor()) else {
                break;
            };

            filter = filter.with_cursor(&cursor);
        }

        let mq = self.mq.clone();
        db::begin(self.genotypes_wr.clone(), |tx| {
            Box::pin(async move {
                let mut publisher = fx_mq_jobs::Publisher::new_tx(tx, &mq);
                publisher.publish_many(&jobs).await?;
                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    /// Returns the distinct indexer IDs associated with a given request.
    ///
    /// Paginates through all tags co-occurring with the `request_id` tag,
    /// restricting the search to tags containing the `"indexer_id:"` prefix.
    /// Each returned tag is parsed as a `Digest`. Returns an error if any
    /// `indexer_id:` tag contains a value that cannot be parsed — such tags
    /// are not expected and indicate corrupt data.
    #[instrument(level = "debug", skip(self))]
    pub async fn get_indexers_of_request(
        &self,
        request: &Request,
    ) -> Result<Vec<Digest>, super::Error> {
        const PAGE_SIZE: i64 = 1_000;
        const PREFIX: &str = "indexer_id:";

        let mut ids: HashSet<Digest> = HashSet::new();

        for id in self
            .indexing
            .get_indexer_ids_of_type(&request.type_name)
            .await
        {
            ids.insert(id);
        }

        let request_tag = Self::fmt_tag_request_id(request.id);
        let mut cursor: Option<String> = None;

        loop {
            let mut filter = SearchAssociatedTagsFilter::default()
                .with_tag(request_tag.clone())
                .with_substring(PREFIX.to_string());

            if let Some(c) = cursor {
                filter = filter.with_cursor(c);
            }

            let (page, next_cursor) = self
                .embeddings_ro
                .get_distinct_associated_tags(&filter, PAGE_SIZE)
                .await?;

            for tag in page {
                if let Some(hex) = tag.strip_prefix(PREFIX).map(str::to_owned) {
                    let id =
                        Digest::from_hex(&hex).map_err(|e| super::Error::InvalidIndexerTag {
                            tag: format!("indexer_id:{hex}"),
                            source: e,
                        })?;

                    ids.insert(id);
                }
            }

            cursor = next_cursor;

            if cursor.is_none() {
                break;
            }
        }

        Ok(ids.into_iter().collect::<Vec<Digest>>())
    }

    /// Returns the registered indexer IDs for the given genotype type.
    pub async fn get_registered_indexers_of_type(&self, type_name: &str) -> Vec<Digest> {
        self.indexing.get_indexer_ids_of_type(type_name).await
    }

    /// Checks if any indexing jobs are pending for the given indexer and request.
    #[instrument(level = "debug", skip(self))]
    pub async fn is_indexing_pending(
        &self,
        indexer_id: &Digest,
        request_id: Uuid,
    ) -> Result<bool, super::Error> {
        let payload = serde_json::json!({
            "indexer_id": indexer_id,
            "request_id": request_id,
        });

        let mq = self.mq.clone();
        let is_pending = db::begin(self.genotypes_wr.clone(), |mut tx| {
            Box::pin(async move {
                let count = mq
                    .search_pending(&mut tx, IndexGenotypesMessage::NAME, &payload)
                    .await
                    .map_err(anyhow::Error::new)?;
                Ok(count > 0)
            })
        })
        .await?;

        Ok(is_pending)
    }

    #[instrument(level = "debug")]
    fn format_genotype_tags<'a>(indexer_id: &'a Digest, genotype: &Genotype) -> Vec<String> {
        // Capacity set to the maximum number of tags
        let mut tags = Vec::with_capacity(6);

        tags.push(Self::fmt_tag_type_genotype());
        tags.push(Self::fmt_tag_type_name(genotype.type_name()));
        tags.push(Self::fmt_tag_genotype_id(genotype.id()));
        tags.push(Self::fmt_tag_indexer_id(indexer_id));

        tags.push(Self::fmt_tag_request_id(genotype.request_id()));

        if let Some(generation_id) = genotype.generation_id() {
            tags.push(Self::fmt_tag_generation_id(generation_id))
        }

        tags
    }

    fn fmt_tag_type_genotype() -> String {
        "type:Genotype".to_string()
    }

    fn fmt_tag_type_name(type_name: &str) -> String {
        format!("type_name:{}", type_name)
    }

    fn fmt_tag_request_id(request_id: Uuid) -> String {
        format!("request_id:{}", request_id)
    }

    fn fmt_tag_generation_id(generation_id: i32) -> String {
        format!("generation_id:{}", generation_id)
    }

    fn fmt_tag_genotype_id(genotype_id: Uuid) -> String {
        format!("genotype_id:{}", genotype_id)
    }

    fn fmt_tag_indexer_id(indexer_id: &Digest) -> String {
        format!("indexer_id:{}", indexer_id)
    }
}

#[cfg(test)]
mod tests_index_genotypes {
    use super::IndexGenotypesMessage;
    use super::Service;
    use crate::SearchGenotypesFilter;
    use crate::bootstrap::App;
    use crate::infrastructure::db;
    use crate::repositories::genotypes::{Genotype, TypeName, store_genotypes};
    use crate::services::indexing::EncodeInput;
    use crate::services::indexing::SearchEmbeddingsFilter;
    use crate::services::indexing::embeddings::{self, EmbeddingNew, RequestedEmbedding, TagNew};
    use crate::services::indexing::encoder::lstm::{self, AutoencoderConfig, AutoencoderModel};
    use crate::services::indexing::{
        Digest, Encoder,
        encoder_queries::{store_encoder, store_encoder_availability},
    };
    use crate::services::indexing::{Indexer, Registry};
    use crate::services::indexing::{
        MODEL_FORMAT, TrainModelConfig,
        encoder::{
            dataset::{SequenceDataSource, SequenceDataset, SequenceSample},
            train::AutoencoderTrainConfig,
        },
    };
    use crate::services::optimization as foreign_service;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use burn::prelude::*;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn_ndarray::NdArray;
    use chrono::Utc;
    use fx_mq_building_blocks::testing_tools::TestQueries;
    use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
    use fx_mq_jobs::Message;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    // Indexing returns embedding ids for all requested genotypes.
    async fn it_indexes_genotypes(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (_request_id, genotype_ids) = seed(&pool, &[indexer], 2).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let embedding_ids = app
            .services()
            .genotype_indexing()
            .index_genotypes(&indexer_id, &genotype_ids, None)
            .await?
            .expect("expected embeddings to be returned");

        assert_eq!(embedding_ids.len(), genotype_ids.len());

        let embeddings = app
            .repositories()
            .embeddings()
            .search_embeddings(
                &SearchEmbeddingsFilter::default().with_encoder_id(&indexer_id),
                genotype_ids.len() as i64,
            )
            .await?;

        assert_eq!(embeddings.len(), genotype_ids.len());

        let indexer_id_tag = format!("indexer_id:{}", indexer_id);
        let type_name_tag = format!("type_name:{}", TestIndexableType::TYPE_NAME);
        for embedding in &embeddings {
            assert!(
                embedding.tags.iter().any(|t| t == "type:Genotype"),
                "missing type:Genotype tag for embedding {}",
                embedding.embedding_id
            );
            assert!(
                embedding.tags.contains(&type_name_tag),
                "missing type_name tag for embedding {}",
                embedding.embedding_id
            );
            assert!(
                embedding.tags.iter().any(|t| t.starts_with("genotype_id:")),
                "missing genotype_id tag for embedding {}",
                embedding.embedding_id
            );
            assert!(
                embedding.tags.contains(&indexer_id_tag),
                "missing indexer_id tag for embedding {}",
                embedding.embedding_id
            );
        }

        Ok(())
    }

    #[ignore = "genotypes repository does not support pagination yet"]
    #[sqlx::test(migrations = false)]
    // Indexing handles a batch larger than the page limit.
    async fn it_paginates_and_indexes_large_batches(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (_request_id, genotype_ids) =
            seed(&pool, &[indexer], super::LIMIT as usize + 5).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let embedding_ids = app
            .services()
            .genotype_indexing()
            .index_genotypes(&indexer_id, &genotype_ids, None)
            .await?
            .expect("expected embeddings to be returned");

        assert_eq!(embedding_ids.len(), genotype_ids.len());

        let embeddings = app
            .repositories()
            .embeddings()
            .search_embeddings(
                &SearchEmbeddingsFilter::default().with_encoder_id(&indexer_id),
                genotype_ids.len() as i64,
            )
            .await?;

        assert_eq!(embeddings.len(), genotype_ids.len());

        let indexer_id_tag = format!("indexer_id:{}", indexer_id);
        let type_name_tag = format!("type_name:{}", TestIndexableType::TYPE_NAME);
        for embedding in &embeddings {
            assert!(
                embedding.tags.iter().any(|t| t == "type:Genotype"),
                "missing type:Genotype tag for embedding {}",
                embedding.embedding_id
            );
            assert!(
                embedding.tags.contains(&type_name_tag),
                "missing type_name tag for embedding {}",
                embedding.embedding_id
            );
            assert!(
                embedding.tags.iter().any(|t| t.starts_with("genotype_id:")),
                "missing genotype_id tag for embedding {}",
                embedding.embedding_id
            );
            assert!(
                embedding.tags.contains(&indexer_id_tag),
                "missing indexer_id tag for embedding {}",
                embedding.embedding_id
            );
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    // Backfill skips genotypes that are already indexed.
    async fn it_backfills_only_missing_genotypes(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (request_id, genotype_ids) = seed(&pool, &[indexer], 3).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;
        let first_id = genotype_ids[0];

        let now = Utc::now();

        let embedding = EmbeddingNew::new(indexer_id, now, [1_f32; 256]);

        let genotype = app
            .repositories()
            .genotypes()
            .search_genotypes(
                &SearchGenotypesFilter::default().with_genotype_id(first_id),
                1,
            )
            .await?
            .into_iter()
            .next()
            .expect("expected genotype to exist");

        let tags = Service::format_genotype_tags(&indexer_id, &genotype)
            .into_iter()
            .map(|tag| TagNew::new(&tag, *embedding.id(), now))
            .collect::<Vec<_>>();

        let embeddings_wr = embeddings::Write::new(db::WritePool { pool: pool.clone() });
        db::begin(embeddings_wr.clone(), |tx| {
            Box::pin(async move {
                let mut wr = embeddings::WriteTx::new(tx);
                wr.store_embeddings(&[embedding]).await?;
                wr.store_tags(&tags).await?;
                Ok(())
            })
        })
        .await?;

        app.services()
            .genotype_indexing()
            .backfill_missing_genotypes(
                SearchGenotypesFilter::default().with_request_id(request_id),
            )
            .await?;

        let mut tx = pool.begin().await?;
        let jobs = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME)
            .get_all_messages(&mut tx)
            .await?;

        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].name, IndexGenotypesMessage::NAME);

        let payload: IndexGenotypesMessage = serde_json::from_value(jobs[0].payload.clone())?;
        assert_eq!(payload.indexer_id, indexer_id);
        assert_eq!(payload.genotype_ids.len(), 2);
        assert!(!payload.genotype_ids.contains(&first_id));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    // Backfill emits no jobs when all genotypes are already indexed.
    async fn it_skips_when_all_indexed(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (request_id, genotype_ids) = seed(&pool, &[indexer], 3).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let embeddings_wr = embeddings::Write::new(db::WritePool { pool: pool.clone() });
        store_indexed(&app, &embeddings_wr, &indexer_id, &genotype_ids).await?;

        app.services()
            .genotype_indexing()
            .backfill_missing_genotypes(
                SearchGenotypesFilter::default().with_request_id(request_id),
            )
            .await?;

        let mut tx = pool.begin().await?;
        let jobs = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME)
            .get_all_messages(&mut tx)
            .await?;

        assert!(jobs.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    // Backfill batches jobs at INDEX_BATCH_SIZE.
    async fn it_batches_backfill_jobs(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let total = super::INDEX_BATCH_SIZE + 5;
        let (request_id, genotype_ids) = seed(&pool, &[indexer], total).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        app.services()
            .genotype_indexing()
            .backfill_missing_genotypes(
                SearchGenotypesFilter::default().with_request_id(request_id),
            )
            .await?;

        let mut tx = pool.begin().await?;
        let jobs = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME)
            .get_all_messages(&mut tx)
            .await?;

        let expected_jobs = total.div_ceil(super::INDEX_BATCH_SIZE);
        assert_eq!(jobs.len(), expected_jobs);

        let mut seen: HashSet<Uuid> = HashSet::new();
        for job in jobs.iter() {
            let payload: IndexGenotypesMessage = serde_json::from_value(job.payload.clone())?;
            assert_eq!(payload.indexer_id, indexer_id);
            assert!(payload.genotype_ids.len() <= super::INDEX_BATCH_SIZE);
            for id in payload.genotype_ids {
                seen.insert(id);
            }
        }

        let expected: HashSet<Uuid> = genotype_ids.into_iter().collect();
        assert_eq!(seen, expected);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    // Backfill processes all pages when results exceed DB_PAGE_SIZE.
    async fn it_backfills_across_pages(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let total = super::DB_PAGE_SIZE + 5;
        let (request_id, genotype_ids) = seed(&pool, &[indexer], total).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        app.services()
            .genotype_indexing()
            .backfill_missing_genotypes(
                SearchGenotypesFilter::default().with_request_id(request_id),
            )
            .await?;

        let mut tx = pool.begin().await?;
        let jobs = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME)
            .get_all_messages(&mut tx)
            .await?;

        let mut seen: HashSet<Uuid> = HashSet::new();
        for job in jobs.iter() {
            let payload: IndexGenotypesMessage = serde_json::from_value(job.payload.clone())?;
            assert_eq!(payload.indexer_id, indexer_id);
            for id in payload.genotype_ids {
                seen.insert(id);
            }
        }

        let expected: HashSet<Uuid> = genotype_ids.into_iter().collect();
        assert_eq!(seen, expected);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    // Backfill filters per indexer when multiple indexers share a type.
    async fn it_backfills_per_indexer(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer_a = TestGenotypeIndexer::default();
        let indexer_b = test_indexer_with_epochs(2);
        let indexer_id_a = Registry::get_indexer_id(&indexer_a)?;
        let indexer_id_b = Registry::get_indexer_id(&indexer_b)?;
        let (request_id, genotype_ids) = seed(&pool, &[indexer_a, indexer_b], 4).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .with_indexer(test_indexer_with_epochs(2))
            .build()
            .await?;

        let app = c.get::<Arc<App>>().await?;

        let preindexed = vec![genotype_ids[0], genotype_ids[1]];
        let embeddings_wr = embeddings::Write::new(db::WritePool { pool: pool.clone() });
        store_indexed(&app, &embeddings_wr, &indexer_id_a, &preindexed).await?;

        app.services()
            .genotype_indexing()
            .backfill_missing_genotypes(
                SearchGenotypesFilter::default().with_request_id(request_id),
            )
            .await?;

        let mut tx = pool.begin().await?;
        let jobs = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME)
            .get_all_messages(&mut tx)
            .await?;

        let mut by_indexer: HashMap<Digest, HashSet<Uuid>> = HashMap::new();
        for job in jobs.iter() {
            let payload: IndexGenotypesMessage = serde_json::from_value(job.payload.clone())?;
            by_indexer
                .entry(payload.indexer_id)
                .or_insert_with(HashSet::new)
                .extend(payload.genotype_ids);
        }

        let expected_b: HashSet<Uuid> = genotype_ids.iter().copied().collect();
        let expected_a: HashSet<Uuid> = genotype_ids
            .iter()
            .copied()
            .filter(|id| !preindexed.contains(id))
            .collect();

        let seen_a = by_indexer
            .remove(&indexer_id_a)
            .ok_or_else(|| anyhow::anyhow!("missing jobs for indexer A"))?;
        let seen_b = by_indexer
            .remove(&indexer_id_b)
            .ok_or_else(|| anyhow::anyhow!("missing jobs for indexer B"))?;

        assert_eq!(seen_a, expected_a);
        assert_eq!(seen_b, expected_b);

        Ok(())
    }

    #[derive(Serialize, Deserialize)]
    struct TestIndexableType {
        values: Vec<f32>,
    }

    impl TestIndexableType {
        const TYPE_NAME: &'static str = "test::indexable_type";
    }

    struct NoOpOptimizer;

    impl TypeName for NoOpOptimizer {
        fn type_name(&self) -> &str {
            TestIndexableType::TYPE_NAME
        }
    }

    impl foreign_service::Optimizer for NoOpOptimizer {
        type Type = TestIndexableType;

        fn random(&self) -> anyhow::Result<Self::Type> {
            Ok(TestIndexableType {
                values: vec![0.0, 0.0],
            })
        }

        fn crossover(
            &self,
            _parent1: Self::Type,
            _parent2: Self::Type,
        ) -> anyhow::Result<Self::Type> {
            Ok(TestIndexableType {
                values: vec![0.0, 0.0],
            })
        }

        fn mutate(&self, _instance: &mut Self::Type) -> anyhow::Result<()> {
            Ok(())
        }
    }

    impl TypeName for TestIndexableType {
        fn type_name(&self) -> &str {
            Self::TYPE_NAME
        }
    }

    #[derive(Deserialize)]
    struct TestGenotypeIndexer {
        train_config: TrainModelConfig,
    }

    impl TestGenotypeIndexer {
        const INPUT_SIZE: usize = 2;
        const LATENT_SIZE: usize = 2;
    }

    impl Default for TestGenotypeIndexer {
        fn default() -> Self {
            Self {
                train_config: TrainModelConfig::Lstm(AutoencoderTrainConfig {
                    input_size: Self::INPUT_SIZE,
                    hidden_size: 4,
                    latent_size: Self::LATENT_SIZE,
                    batch_size: 2,
                    epochs: 1,
                    learning_rate: 1e-3,
                }),
            }
        }
    }

    fn test_indexer_with_epochs(epochs: usize) -> TestGenotypeIndexer {
        TestGenotypeIndexer {
            train_config: TrainModelConfig::Lstm(AutoencoderTrainConfig {
                input_size: TestGenotypeIndexer::INPUT_SIZE,
                hidden_size: 4,
                latent_size: TestGenotypeIndexer::LATENT_SIZE,
                batch_size: 2,
                epochs,
                learning_rate: 1e-3,
            }),
        }
    }

    impl TypeName for TestGenotypeIndexer {
        fn type_name(&self) -> &'static str {
            TestIndexableType::TYPE_NAME
        }
    }

    impl Indexer for TestGenotypeIndexer {
        type Type = TestIndexableType;

        fn preprocess(&self, indexable_type: &Self::Type) -> EncodeInput {
            let values = indexable_type.values.clone();

            EncodeInput {
                values,
                dimensions: vec![1, Self::INPUT_SIZE],
            }
        }

        fn dataset(&self) -> Arc<dyn SequenceDataSource> {
            Arc::new(SequenceDataset::new(
                vec![
                    SequenceSample {
                        steps: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
                    },
                    SequenceSample {
                        steps: vec![vec![0.5, 0.6]],
                    },
                ],
                Self::INPUT_SIZE,
            ))
        }

        fn training_config(&self) -> &TrainModelConfig {
            &self.train_config
        }
    }

    async fn seed(
        pool: &sqlx::PgPool,
        indexers: &[TestGenotypeIndexer],
        genotype_count: usize,
    ) -> anyhow::Result<(Uuid, Vec<Uuid>)> {
        let now = Utc::now();

        let request = new_request()?;
        let request_id = request.id;
        store_request(pool, request).await?;

        let genotypes = new_genotypes(request_id, genotype_count)?;
        store_genotypes(pool, &genotypes).await?;

        for indexer in indexers.iter() {
            let indexer_id = Registry::get_indexer_id(indexer)?;
            let encoder = new_encoder(&indexer_id)?;
            store_encoder(pool, &encoder, &now).await?;
            store_encoder_availability(pool, &indexer_id, true, &now).await?;
        }

        Ok((request_id, genotypes.iter().map(|g| g.id()).collect()))
    }

    async fn store_indexed(
        app: &crate::bootstrap::App,
        embeddings_wr: &embeddings::Write,
        indexer_id: &Digest,
        genotype_ids: &[Uuid],
    ) -> anyhow::Result<()> {
        let now = Utc::now();
        let genotypes = app
            .repositories()
            .genotypes()
            .search_genotypes(
                &genotype_ids
                    .iter()
                    .fold(SearchGenotypesFilter::default(), |filter, id| {
                        filter.with_genotype_id(*id)
                    }),
                genotype_ids.len() as i64,
            )
            .await?
            .into_iter()
            .collect::<Vec<_>>();

        let mut embeddings: Vec<EmbeddingNew> = Vec::with_capacity(genotypes.len());
        let mut tags: Vec<TagNew> = Vec::new();
        for genotype in genotypes.iter() {
            let embedding = EmbeddingNew::new(*indexer_id, now, [1_f32; 256]);
            let embedding_id = *embedding.id();
            let mut embedding_tags = Service::format_genotype_tags(indexer_id, genotype)
                .into_iter()
                .map(|tag| TagNew::new(&tag, embedding_id, now))
                .collect::<Vec<_>>();
            tags.append(&mut embedding_tags);
            embeddings.push(embedding);
        }

        db::begin(embeddings_wr.clone(), |tx| {
            Box::pin(async move {
                let mut wr = crate::services::indexing::embeddings::WriteTx::new(tx);
                wr.store_embeddings(&embeddings).await?;
                wr.store_tags(&tags).await?;
                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    fn new_request() -> anyhow::Result<Request> {
        let request = Request::new(
            "genotype-indexing-test",
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        Ok(request)
    }

    fn new_genotypes(request_id: Uuid, genotype_count: usize) -> anyhow::Result<Vec<Genotype>> {
        let genotypes = (0..genotype_count)
            .map(|i| {
                Genotype::new(
                    TestIndexableType::TYPE_NAME,
                    json!({ "values": [i as i32, (i + 1) as i32] }),
                    request_id,
                    Some((i + 1) as i32),
                    None,
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(genotypes)
    }

    #[sqlx::test(migrations = false)]
    async fn get_registered_indexers_of_type_returns_indexers(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let expected = Registry::get_indexer_id(&TestGenotypeIndexer::default())?;
        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let ids = app
            .services()
            .genotype_indexing()
            .get_registered_indexers_of_type(TestIndexableType::TYPE_NAME)
            .await;

        assert_eq!(ids, vec![expected]);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn get_registered_indexers_of_type_returns_empty_for_unknown_type(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let ids = app
            .services()
            .genotype_indexing()
            .get_registered_indexers_of_type("unknown::type")
            .await;

        assert!(ids.is_empty());

        Ok(())
    }

    async fn seed_request(pool: &sqlx::PgPool) -> anyhow::Result<Uuid> {
        let request = new_request()?;
        let id = request.id;
        store_request(pool, request).await?;
        Ok(id)
    }

    #[sqlx::test(migrations = false)]
    async fn is_indexing_pending_returns_true_when_jobs_exist(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let request_id = seed_request(&pool).await?;
        let genotypes = new_genotypes(request_id, 2)?;
        let stored = store_genotypes(&pool, &genotypes).await?;
        let _genotype_ids: Vec<Uuid> = stored.iter().map(|g| g.id()).collect();

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(indexer)
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        app.services()
            .genotype_indexing()
            .backfill_missing_genotypes(
                crate::SearchGenotypesFilter::default().with_request_id(request_id),
            )
            .await?;

        let pending = app
            .services()
            .genotype_indexing()
            .is_indexing_pending(&indexer_id, request_id)
            .await?;

        assert!(pending);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn is_indexing_pending_returns_false_when_no_jobs(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let request_id = Uuid::nil();

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(indexer)
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let pending = app
            .services()
            .genotype_indexing()
            .is_indexing_pending(&indexer_id, request_id)
            .await?;

        assert!(!pending);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn get_indexers_of_request_finds_indexers_from_registry_and_tags(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (request_id, genotype_ids) = seed(&pool, &[indexer], 2).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let embeddings_wr = embeddings::Write::new(db::WritePool { pool: pool.clone() });
        store_indexed(&app, &embeddings_wr, &indexer_id, &genotype_ids).await?;

        let request = app
            .repositories()
            .requests()
            .get_request(request_id)
            .await?;
        let indexers = app
            .services()
            .genotype_indexing()
            .get_indexers_of_request(&request)
            .await?;

        assert!(indexers.contains(&indexer_id));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn get_indexers_of_request_returns_empty_when_no_indexers(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (request_id, _) = seed(&pool, &[], 2).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        let request = app
            .repositories()
            .requests()
            .get_request(request_id)
            .await?;
        let indexers = app
            .services()
            .genotype_indexing()
            .get_indexers_of_request(&request)
            .await?;

        assert!(indexers.is_empty());

        Ok(())
    }

    fn new_encoder(indexer_id: &Digest) -> anyhow::Result<Encoder> {
        type TestBackend = NdArray<f32>;

        let config = AutoencoderConfig {
            input_size: TestGenotypeIndexer::INPUT_SIZE,
            hidden_size: 4,
            latent_size: TestGenotypeIndexer::LATENT_SIZE,
        };

        let device = <TestBackend as Backend>::Device::default();
        let model = lstm::LstmAutoencoder::<TestBackend>::new(&device, config);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let model_bytes = Recorder::<TestBackend>::record(&recorder, model.into_record(), ())?;

        let encoder = Encoder::new(
            *indexer_id,
            TestIndexableType::TYPE_NAME.to_string(),
            serde_json::to_value(&config)?,
            model_bytes,
            MODEL_FORMAT.to_string(),
            vec![TestGenotypeIndexer::INPUT_SIZE as i32],
            TestGenotypeIndexer::LATENT_SIZE as i32,
        );

        Ok(encoder)
    }

    // --- retry_deferred_indexation tests ---

    async fn seed_deferred(
        pool: &sqlx::PgPool,
        indexer_id: &Digest,
        genotype_ids: &[Uuid],
        request_id: Option<Uuid>,
    ) -> anyhow::Result<()> {
        let now = Utc::now();
        let metadata = request_id
            .map(|id| serde_json::json!({ "request_id": id.to_string() }))
            .unwrap_or(serde_json::json!({}));

        let deferred: Vec<RequestedEmbedding> = genotype_ids
            .iter()
            .map(|entity_id| {
                RequestedEmbedding::new(
                    *entity_id,
                    TestIndexableType::TYPE_NAME.to_string(),
                    *indexer_id,
                    metadata.clone(),
                    now,
                )
            })
            .collect();

        let embeddings_wr = embeddings::Write::new(db::WritePool { pool: pool.clone() });
        db::begin(embeddings_wr.clone(), |tx| {
            Box::pin(async move {
                let mut wr = embeddings::WriteTx::new(tx);
                wr.store_requested_embeddings(&deferred).await?;
                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_retries_deferred_items(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (request_id, genotype_ids) = seed(&pool, &[indexer], 3).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        seed_deferred(&pool, &indexer_id, &genotype_ids, Some(request_id)).await?;

        app.services()
            .genotype_indexing()
            .retry_deferred_indexation(&indexer_id)
            .await?;

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let jobs = mq.get_all_messages(&mut tx).await?;
        tx.commit().await?;

        let matching: Vec<_> = jobs
            .iter()
            .filter(|j| j.name == IndexGenotypesMessage::NAME)
            .collect();
        assert_eq!(matching.len(), 1);

        let payload: IndexGenotypesMessage = serde_json::from_value(matching[0].payload.clone())?;
        assert_eq!(payload.indexer_id, indexer_id);
        assert_eq!(payload.genotype_ids.len(), 3);
        assert_eq!(payload.request_id, Some(request_id));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_retries_nothing_when_empty(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (_request_id, _genotype_ids) = seed(&pool, &[indexer], 2).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        app.services()
            .genotype_indexing()
            .retry_deferred_indexation(&indexer_id)
            .await?;

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let jobs = mq.get_all_messages(&mut tx).await?;
        tx.commit().await?;

        let matching: Vec<_> = jobs
            .iter()
            .filter(|j| j.name == IndexGenotypesMessage::NAME)
            .collect();
        assert!(matching.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_groups_deferred_by_request_id(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let (request_id_a, genotype_ids_a) = seed(&pool, &[indexer], 2).await?;
        let (request_id_b, genotype_ids_b) = seed(&pool, &[], 1).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        seed_deferred(&pool, &indexer_id, &genotype_ids_a, Some(request_id_a)).await?;
        seed_deferred(&pool, &indexer_id, &genotype_ids_b, Some(request_id_b)).await?;

        app.services()
            .genotype_indexing()
            .retry_deferred_indexation(&indexer_id)
            .await?;

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let jobs = mq.get_all_messages(&mut tx).await?;
        tx.commit().await?;

        let matching: Vec<_> = jobs
            .iter()
            .filter(|j| j.name == IndexGenotypesMessage::NAME)
            .collect();
        assert_eq!(matching.len(), 2);

        let payload_a: IndexGenotypesMessage = serde_json::from_value(matching[0].payload.clone())?;
        let payload_b: IndexGenotypesMessage = serde_json::from_value(matching[1].payload.clone())?;

        assert_ne!(payload_a.request_id, payload_b.request_id);
        assert_eq!(
            payload_a.genotype_ids.len() + payload_b.genotype_ids.len(),
            3
        );

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_batches_large_deferred_sets(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let indexer = TestGenotypeIndexer::default();
        let indexer_id = Registry::get_indexer_id(&indexer)?;
        let total = super::INDEX_BATCH_SIZE + 5;
        let (request_id, genotype_ids) = seed(&pool, &[indexer], total).await?;

        let mut c = crate::test_tools::TestConfig::new(pool.clone())
            .with_optimizer(NoOpOptimizer)
            .with_indexer(TestGenotypeIndexer::default())
            .build()
            .await?;
        let app = c.get::<Arc<App>>().await?;

        seed_deferred(&pool, &indexer_id, &genotype_ids, Some(request_id)).await?;

        app.services()
            .genotype_indexing()
            .retry_deferred_indexation(&indexer_id)
            .await?;

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let jobs = mq.get_all_messages(&mut tx).await?;
        tx.commit().await?;

        let expected_jobs = total.div_ceil(super::INDEX_BATCH_SIZE);
        assert_eq!(jobs.len(), expected_jobs);

        let mut seen: HashSet<Uuid> = HashSet::new();
        for job in jobs.iter() {
            let payload: IndexGenotypesMessage = serde_json::from_value(job.payload.clone())?;
            assert!(payload.genotype_ids.len() <= super::INDEX_BATCH_SIZE);
            for id in payload.genotype_ids {
                seen.insert(id);
            }
        }

        let expected: HashSet<Uuid> = genotype_ids.into_iter().collect();
        assert_eq!(seen, expected);

        Ok(())
    }
}
