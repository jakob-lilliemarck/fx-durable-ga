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
const GENOTYPES_INDEXING_BATCH_SIZE: usize = 100;

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
            .flat_map(|(grouping_key, genotype_ids)| {
                genotype_ids
                    .chunks(GENOTYPES_INDEXING_BATCH_SIZE)
                    .map(|chunk| IndexGenotypesMessage {
                        grouping_key: grouping_key.clone(),
                        genotype_ids: chunk.to_vec(),
                    })
                    .collect::<Vec<_>>()
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
    ) -> Result<Option<Vec<Uuid>>, super::Error> {
        let mut filter = GenotypesFilter::default();
        for id in genotype_ids {
            filter = filter.with_genotype_id(*id);
        }

        let genotypes = self
            .genotypes
            .search_genotypes(&filter, genotype_ids.len() as i64)
            .await?;

        // FIXME!
        // consider validating that each genotype has the expected type_hash and type_name

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

        let embedding_ids = self
            .indexing
            .index_many(
                &grouping_key.encoder_id,
                &inputs,
                &Self::format_tags(&grouping_key, &indexer.context_hash()),
            )
            .await?;

        Ok(Some(embedding_ids))
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

#[cfg(test)]
mod test_support {
    use super::Service;
    use crate::bootstrap::ApplicationBuilder;
    use crate::models::{
        EncodeInput, FitnessGoal, Genotype, GenotypeIndexer, Request, Schedule, Selector, TypeName,
    };
    use crate::repositories::encoders::{Encoder, store_encoder};
    use crate::repositories::genotypes::new_genotypes;
    use crate::repositories::requests::queries::new_request;
    use crate::services::indexing;
    use crate::services::indexing::ModelConfig;
    use crate::services::indexing::encoder::lstm::{
        AutoencoderConfig, AutoencoderModel, LstmAutoencoder,
    };
    use anyhow::Context;
    use burn::prelude::*;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn_ndarray::NdArray;
    use chrono::Utc;
    use serde_json::{Value, json};
    use sqlx::PgPool;
    use std::sync::Arc;
    use uuid::Uuid;

    #[derive(Clone)]
    pub(crate) struct BuiltServices {
        pub(crate) indexing: Arc<indexing::Service>,
        pub(crate) genotype_indexing: Arc<Service>,
    }

    pub(crate) async fn build_services(
        pool: &PgPool,
        register_indexer: bool,
    ) -> anyhow::Result<BuiltServices> {
        let app_builder = ApplicationBuilder::default().with_pool(pool.clone());
        let indexing = Arc::new(app_builder.indexing_service().build());
        let builder = app_builder.genotype_indexing_service(indexing.clone());
        let builder = if register_indexer {
            builder.with_indexable(TestIndexer)
        } else {
            builder
        };
        let genotype_indexing = Arc::new(builder.build());

        Ok(BuiltServices {
            indexing,
            genotype_indexing,
        })
    }

    #[derive(Clone, Copy)]
    pub(crate) struct TestIndexer;

    impl TestIndexer {
        pub(crate) const TYPE_NAME: &'static str = "TestIndexer";
        pub(crate) const TYPE_HASH: i32 = 9_001;
        pub(crate) const CONTEXT_HASH: &'static str = "test-context";
    }

    impl TypeName for TestIndexer {
        fn type_name(&self) -> &'static str {
            Self::TYPE_NAME
        }

        fn type_hash(&self) -> i32 {
            Self::TYPE_HASH
        }
    }

    impl GenotypeIndexer for TestIndexer {
        fn input(&self, genotype: &Genotype) -> EncodeInput {
            let values = match genotype.genome() {
                Value::Array(items) => items
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .map(|v| v as f32)
                    .collect::<Vec<f32>>(),
                _ => vec![0.0, 0.0],
            };
            let dimensions = vec![1, values.len().max(1)];
            EncodeInput { values, dimensions }
        }

        fn context_hash(&self) -> String {
            Self::CONTEXT_HASH.to_string()
        }
    }

    pub(crate) async fn seed_encoder(pool: &PgPool) -> anyhow::Result<Uuid> {
        type TestBackend = NdArray<f32>;

        let config = AutoencoderConfig {
            input_size: 2,
            hidden_size: 4,
            latent_size: 2,
        };
        let model_config = ModelConfig::Lstm(config);

        let device = <TestBackend as Backend>::Device::default();
        let model = LstmAutoencoder::<TestBackend>::new(&device, config);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let model_bytes = Recorder::<TestBackend>::record(&recorder, model.into_record(), ())?;
        let trained_at = Utc::now();
        let encoder = Encoder {
            id: Uuid::now_v7(),
            model_type: "lstm".to_string(),
            model_config: serde_json::to_value(&model_config).context("serialize config")?,
            model_weights: model_bytes,
            model_format: "burn-bin-f32".to_string(),
            shape_in: vec![2],
            shape_out: 2,
            trained_at,
            trained_on_checksum: vec![1_u8, 2, 3, 4],
        };

        let stored = store_encoder(pool, &encoder).await?;
        Ok(stored.id())
    }

    pub(crate) async fn seed_request(
        pool: &PgPool,
        type_name: &str,
        type_hash: i32,
    ) -> anyhow::Result<Uuid> {
        let goal = FitnessGoal::maximize(0.9)?;
        let selector = Selector::tournament(2);
        let schedule = Schedule::generational(10, 2);
        let user_defined = json!({ "Uniform": { "probability": 1.0 } });
        let request = Request::new(
            type_name,
            type_hash,
            goal,
            selector,
            schedule,
            user_defined,
            None::<()>,
        )?;

        let inserted = new_request(pool, request).await?;
        Ok(inserted.id)
    }

    pub(crate) async fn insert_genotypes(
        pool: &PgPool,
        request_id: Uuid,
        type_name: &str,
        type_hash: i32,
        count: usize,
    ) -> anyhow::Result<Vec<Uuid>> {
        let mut ids = Vec::with_capacity(count);
        for n in 0..count {
            let genome = json!([n as f64, (n + 1) as f64]);
            let id = insert_genotype(pool, request_id, type_name, type_hash, genome, 1).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    pub(crate) async fn insert_genotype(
        pool: &PgPool,
        request_id: Uuid,
        type_name: &str,
        type_hash: i32,
        genome: Value,
        generation_id: i32,
    ) -> anyhow::Result<Uuid> {
        let genotype = Genotype::new(
            type_name,
            type_hash,
            genome,
            request_id,
            generation_id,
            None::<&Uuid>,
            None::<&Uuid>,
        );
        let genotype_id = genotype.id();

        new_genotypes(pool, vec![genotype]).await?;
        Ok(genotype_id)
    }
}

#[cfg(test)]
mod tests_index_genotypes {
    use super::GENOTYPES_INDEXING_BATCH_SIZE;
    use super::test_support::{
        TestIndexer, build_services, insert_genotypes, seed_encoder, seed_request,
    };
    use crate::migrations;
    use crate::repositories::genotypes::GenotypesFilter;
    use crate::services::genotype_indexing::jobs::IndexGenotypesMessage;
    use anyhow::Context;
    use fx_mq_jobs::Message;
    use sqlx::Row;
    use std::collections::HashSet;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_creates_indexing_jobs(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let services = build_services(&pool, true).await?;
        let encoder_id = seed_encoder(&pool).await?;
        services
            .indexing
            .enable_encoder_pairing(&encoder_id, TestIndexer::TYPE_HASH)
            .await?;

        let request_id =
            seed_request(&pool, TestIndexer::TYPE_NAME, TestIndexer::TYPE_HASH).await?;
        let genotype_ids = insert_genotypes(
            &pool,
            request_id,
            TestIndexer::TYPE_NAME,
            TestIndexer::TYPE_HASH,
            GENOTYPES_INDEXING_BATCH_SIZE + 1,
        )
        .await?;

        let filter = GenotypesFilter::default().with_request_id(request_id);
        services.genotype_indexing.index_genotypes(&filter).await?;

        let messages = fetch_index_jobs(&pool).await?;
        assert_eq!(messages.len(), 2, "expected batching to split payloads");

        let expected_ids: HashSet<Uuid> = genotype_ids.iter().copied().collect();
        let mut seen = HashSet::new();
        for message in &messages {
            assert_eq!(message.grouping_key.encoder_id, encoder_id);
            assert_eq!(message.grouping_key.type_hash, TestIndexer::TYPE_HASH);
            assert_eq!(message.grouping_key.type_name, TestIndexer::TYPE_NAME);
            assert!(
                message.genotype_ids.len() <= GENOTYPES_INDEXING_BATCH_SIZE,
                "batch size exceeded"
            );
            for id in &message.genotype_ids {
                seen.insert(*id);
            }
        }

        assert_eq!(
            seen, expected_ids,
            "all genotypes should be scheduled exactly once"
        );

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_skips_unpaired_genotypes(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let services = build_services(&pool, true).await?;
        let encoder_id = seed_encoder(&pool).await?;
        services
            .indexing
            .enable_encoder_pairing(&encoder_id, TestIndexer::TYPE_HASH)
            .await?;

        let paired_request =
            seed_request(&pool, TestIndexer::TYPE_NAME, TestIndexer::TYPE_HASH).await?;
        let paired_ids = insert_genotypes(
            &pool,
            paired_request,
            TestIndexer::TYPE_NAME,
            TestIndexer::TYPE_HASH,
            3,
        )
        .await?;

        let unpaired_hash = TestIndexer::TYPE_HASH + 1;
        let unpaired_request = seed_request(&pool, "Unpaired", unpaired_hash).await?;
        let unpaired_ids =
            insert_genotypes(&pool, unpaired_request, "Unpaired", unpaired_hash, 2).await?;

        let filter = GenotypesFilter::default()
            .with_request_id(paired_request)
            .with_request_id(unpaired_request);
        services.genotype_indexing.index_genotypes(&filter).await?;

        let messages = fetch_index_jobs(&pool).await?;
        assert!(!messages.is_empty());

        let paired: HashSet<Uuid> = paired_ids.iter().copied().collect();
        let skipped: HashSet<Uuid> = unpaired_ids.iter().copied().collect();
        for message in messages {
            for id in message.genotype_ids {
                assert!(paired.contains(&id));
                assert!(!skipped.contains(&id));
            }
        }

        Ok(())
    }

    async fn fetch_index_jobs(pool: &sqlx::PgPool) -> anyhow::Result<Vec<IndexGenotypesMessage>> {
        let rows = sqlx::query(
            r#"
            SELECT payload
            FROM fx_mq_jobs.messages_unattempted
            WHERE name = $1
            ORDER BY published_at ASC, id ASC
            "#,
        )
        .bind(IndexGenotypesMessage::NAME)
        .fetch_all(pool)
        .await?;

        let mut messages = Vec::with_capacity(rows.len());
        for row in rows {
            let payload: serde_json::Value = row.try_get("payload")?;
            let message: IndexGenotypesMessage =
                serde_json::from_value(payload).context("decode IndexGenotypesMessage")?;
            messages.push(message);
        }

        Ok(messages)
    }
}

#[cfg(test)]
mod tests_index_genotype_group {
    use super::test_support::{
        TestIndexer, build_services, insert_genotype, seed_encoder, seed_request,
    };
    use crate::migrations;
    use anyhow::Context;
    use serde_json::json;
    use sqlx::Row;
    use std::collections::{HashMap, HashSet};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_indexes_a_genotype_group(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let services = build_services(&pool, true).await?;
        let encoder_id = seed_encoder(&pool).await?;
        services
            .indexing
            .enable_encoder_pairing(&encoder_id, TestIndexer::TYPE_HASH)
            .await?;

        let request_id =
            seed_request(&pool, TestIndexer::TYPE_NAME, TestIndexer::TYPE_HASH).await?;
        let g1 = insert_genotype(
            &pool,
            request_id,
            TestIndexer::TYPE_NAME,
            TestIndexer::TYPE_HASH,
            json!([1.0, 2.0]),
            1,
        )
        .await?;
        let g2 = insert_genotype(
            &pool,
            request_id,
            TestIndexer::TYPE_NAME,
            TestIndexer::TYPE_HASH,
            json!([3.0, 4.0]),
            1,
        )
        .await?;
        let genotype_ids = vec![g1, g2];

        let grouping_key = super::GroupingKey {
            encoder_id,
            type_hash: TestIndexer::TYPE_HASH,
            type_name: TestIndexer::TYPE_NAME.to_string(),
        };

        let result = services
            .genotype_indexing
            .index_genotype_group(&grouping_key, &genotype_ids)
            .await?;

        assert!(result.is_some());

        let embedding_rows = sqlx::query(
            r#"
            SELECT id, encoded_with
            FROM fx_durable_ga.embeddings
            ORDER BY encoded_at ASC
            "#,
        )
        .fetch_all(&pool)
        .await?;

        assert_eq!(embedding_rows.len(), genotype_ids.len());
        let embedding_ids: Vec<Uuid> = embedding_rows
            .into_iter()
            .map(|row| {
                let encoded_with: Uuid = row.try_get("encoded_with")?;
                assert_eq!(encoded_with, encoder_id);
                row.try_get("id")
            })
            .collect::<Result<_, sqlx::Error>>()?;

        let tag_rows = sqlx::query(
            r#"
            SELECT embedding_id, tag_name
            FROM fx_durable_ga.embedding_tags
            ORDER BY embedding_id, tag_name
            "#,
        )
        .fetch_all(&pool)
        .await?;

        let mut tags_by_embedding: HashMap<Uuid, Vec<String>> = HashMap::new();
        for row in tag_rows {
            let embedding_id: Uuid = row.try_get("embedding_id")?;
            let tag_name: String = row.try_get("tag_name")?;
            tags_by_embedding
                .entry(embedding_id)
                .or_default()
                .push(tag_name);
        }

        let context_hash = TestIndexer::CONTEXT_HASH.to_string();
        let expected: HashSet<String> = super::Service::format_tags(&grouping_key, &context_hash)
            .into_iter()
            .collect();

        for embedding_id in embedding_ids {
            let stored = tags_by_embedding
                .get(&embedding_id)
                .cloned()
                .context("missing tags for embedding")?;
            let actual: HashSet<String> = stored.into_iter().collect();
            assert_eq!(actual, expected);
        }

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_none_when_indexer_missing(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let services = build_services(&pool, false).await?;
        let encoder_id = seed_encoder(&pool).await?;
        services
            .indexing
            .enable_encoder_pairing(&encoder_id, TestIndexer::TYPE_HASH)
            .await?;

        let request_id =
            seed_request(&pool, TestIndexer::TYPE_NAME, TestIndexer::TYPE_HASH).await?;
        let genotype_id = insert_genotype(
            &pool,
            request_id,
            TestIndexer::TYPE_NAME,
            TestIndexer::TYPE_HASH,
            json!([5.0, 6.0]),
            1,
        )
        .await?;

        let grouping_key = super::GroupingKey {
            encoder_id,
            type_hash: TestIndexer::TYPE_HASH,
            type_name: TestIndexer::TYPE_NAME.to_string(),
        };

        let result = services
            .genotype_indexing
            .index_genotype_group(&grouping_key, &[genotype_id])
            .await?;

        assert!(result.is_none());

        let embeddings_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM fx_durable_ga.embeddings")
                .fetch_one(&pool)
                .await?;
        assert_eq!(embeddings_count, 0);

        let tags_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM fx_durable_ga.embedding_tags")
                .fetch_one(&pool)
                .await?;
        assert_eq!(tags_count, 0);

        Ok(())
    }
}
