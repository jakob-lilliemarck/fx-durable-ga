use super::encoder::dataset::SequenceDataSource;
use super::encoder::lstm::{self, AutoencoderModel};
use super::encoder::train::{AutoencoderTrainConfig, train_autoencoder};
use crate::chainable::FromTx;
use crate::models::EncodeInput;
use crate::repositories::chainable::Chain;
use crate::repositories::embeddings::{Embedding, Similar, Tag};
use crate::repositories::encoders::Encoder;
use crate::repositories::{self, embeddings, encoders};
use crate::services::indexing::events::{EmbeddingCreatedEvent, EncoderPairingToggledEvent};
use burn::backend::Autodiff;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_ndarray::NdArray;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::slice;
use std::sync::Arc;
use uuid::Uuid;

type CpuBackend = NdArray<f32>;
type TrainingBackend = Autodiff<CpuBackend>;
type InferenceBackend = CpuBackend;

const MODEL_FORMAT: &str = "burn-bin-f32";

pub struct Service {
    pub(super) embeddings: Arc<embeddings::Repository>,
    pub(super) encoders: Arc<encoders::Repository>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelConfig {
    Lstm(lstm::AutoencoderConfig),
}

// Improve construction of these - make constructors accessible on the service?
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TrainModelConfig {
    Lstm(AutoencoderTrainConfig),
}

impl ModelConfig {
    pub fn model_type<'a>(&self) -> &'a str {
        match self {
            Self::Lstm(_) => "lstm",
        }
    }
}

impl Service {
    pub(crate) fn builder(
        embeddings: Arc<embeddings::Repository>,
        encoders: Arc<encoders::Repository>,
    ) -> super::ServiceBuilder {
        super::ServiceBuilder {
            embeddings,
            encoders,
        }
    }

    pub async fn index_many(
        &self,
        encoder_id: &Uuid,
        inputs: &[EncodeInput],
        tag_names: &[String],
    ) -> Result<Vec<Uuid>, super::Error> {
        let timestamp = Utc::now();
        let encoder = self.encoders.get_encoder(encoder_id).await?;

        let mut embeddings = Vec::with_capacity(inputs.len());
        for i in inputs {
            let embedding = self.encode(i, &encoder)?;
            embeddings.push(Embedding::new(encoder.id(), timestamp, embedding))
        }

        let mut tags = Vec::with_capacity(embeddings.len() * tag_names.len());
        for ref t in tag_names {
            for e in embeddings.iter() {
                tags.push(Tag::new(t.to_string(), *e.id(), timestamp));
            }
        }

        let embedding_ids = self
            .embeddings
            .chain(|mut tx| {
                Box::pin(async move {
                    // Store embeddings
                    let embeddings = tx.store_embeddings(&embeddings).await?;

                    // Store tags
                    let tags = tx.store_tags(&tags).await?;

                    // Iterate over tags and group by embedding id
                    let grouped: HashMap<Uuid, Vec<String>> =
                        tags.iter().fold(HashMap::new(), |mut acc, t| {
                            acc.entry(t.embedding_id)
                                .or_default()
                                .push(t.tag_name.clone());
                            acc
                        });

                    // Publish events for each embedding created
                    let events = grouped
                        .into_iter()
                        .map(|(embedding_id, tags)| EmbeddingCreatedEvent { embedding_id, tags })
                        .collect::<Vec<_>>();

                    let mut publisher = fx_event_bus::Publisher::from_tx(tx);
                    publisher.publish_many(&events).await?;

                    Ok((
                        publisher,
                        embeddings.into_iter().map(|e| *e.id()).collect::<Vec<_>>(),
                    ))
                })
            })
            .await?;

        Ok(embedding_ids)
    }

    pub async fn index_one(
        &self,
        encoder_id: &Uuid,
        input: &EncodeInput,
        tag_names: &[String],
    ) -> Result<(), super::Error> {
        self.index_many(encoder_id, slice::from_ref(input), tag_names)
            .await?;
        Ok(())
    }

    fn encode(
        &self,
        input: &EncodeInput,
        encoder: &Arc<Encoder>,
    ) -> Result<repositories::embeddings::Value, super::Error> {
        if encoder.model_type != "lstm" {
            return Err(super::Error::UnsupportedModel(encoder.model_type.clone()));
        }

        let config = serde_json::from_value(encoder.model_config.clone())?;
        let ModelConfig::Lstm(model_cfg) = config;

        let seq_len = input.dimensions.get(0).copied().unwrap_or(0);
        let input_size = input
            .dimensions
            .get(1)
            .copied()
            .unwrap_or(model_cfg.input_size);

        if input_size != model_cfg.input_size || seq_len == 0 {
            return Err(super::Error::UnsupportedModel("invalid_input".to_string()));
        }

        if seq_len * input_size != input.values.len() {
            return Err(super::Error::UnsupportedModel("invalid_input".to_string()));
        }

        let device = <InferenceBackend as Backend>::Device::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let record =
            Recorder::<InferenceBackend>::load(&recorder, encoder.model_weights.clone(), &device)?;
        let model =
            lstm::LstmAutoencoder::<InferenceBackend>::new(&device, model_cfg).load_record(record);

        let tensor = Tensor::<InferenceBackend, 3>::from_data(
            TensorData::new(input.values.clone(), [1, seq_len, input_size]),
            &device,
        );
        let output = model.forward(tensor);
        let latent = output.latent.into_data().into_vec::<f32>()?;

        let mut embedding = [0.0f32; 256];
        let copy_len = latent.len().min(embedding.len());
        embedding[..copy_len].copy_from_slice(&latent[..copy_len]);

        Ok(embedding)
    }

    pub async fn add_tag<'a>(
        &self,
        embedding_ids: &'a [Uuid],
        tag: &'a str,
    ) -> Result<Vec<Tag>, super::Error> {
        let tagged_at = Utc::now();
        let tags = embedding_ids
            .iter()
            .map(|id| Tag::new(tag.to_owned(), id.clone(), tagged_at))
            .collect::<Vec<Tag>>();

        let tags = self
            .embeddings
            .chain(|mut tx| {
                Box::pin(async move {
                    let res = tx.store_tags(&tags).await?;
                    Ok((tx, res))
                })
            })
            .await?;

        Ok(tags)
    }

    pub async fn enable_encoder_pairing(
        &self,
        encoder_id: &Uuid,
        type_hash: i32,
    ) -> Result<bool, super::Error> {
        let is_enabled = self
            .toggle_encoder_pairing(encoder_id, type_hash, true)
            .await?;
        Ok(is_enabled)
    }

    pub async fn disable_encoder_pairing(
        &self,
        encoder_id: &Uuid,
        type_hash: i32,
    ) -> Result<bool, super::Error> {
        let is_enabled = self
            .toggle_encoder_pairing(encoder_id, type_hash, false)
            .await?;
        Ok(is_enabled)
    }

    async fn toggle_encoder_pairing(
        &self,
        encoder_id: &Uuid,
        type_hash: i32,
        is_enabled: bool,
    ) -> Result<bool, super::Error> {
        let is_enabled = self
            .encoders
            .chain(|mut tx| {
                Box::pin(async move {
                    let result = tx
                        .toggle_encoder_pairing(type_hash, encoder_id, is_enabled)
                        .await?;

                    let mut publisher = fx_event_bus::Publisher::from_tx(tx);

                    // Only publish an event if the state changed
                    if result.was_changed {
                        let event = EncoderPairingToggledEvent {
                            encoder_id: *encoder_id,
                            type_hash,
                            is_enabled: result.is_enabled,
                        };
                        publisher.publish(event).await?;
                    }

                    Ok((publisher, is_enabled))
                })
            })
            .await?;

        Ok(is_enabled)
    }

    pub async fn find_similar(
        &self,
        embedding_id: &Uuid,
        tag_name: &str,
        limit: i64,
    ) -> Result<Vec<Similar>, super::Error> {
        let similar = self
            .embeddings
            .find_similar(embedding_id, tag_name, limit)
            .await?;

        Ok(similar)
    }

    // Get or train an encoder
    pub async fn train_encoder<D>(
        &self,
        encoder_id: Uuid,
        train_config: TrainModelConfig,
        dataset: D,
    ) -> Result<Encoder, super::Error>
    where
        D: SequenceDataSource,
    {
        let (weights, model_config) = self.train_model(&train_config, &dataset)?;

        let (shape_in, shape_out) = match model_config {
            ModelConfig::Lstm(cfg) => (vec![cfg.input_size as i32], cfg.latent_size as i32),
        };

        let encoder = encoders::Encoder {
            id: encoder_id,
            model_type: model_config.model_type().to_string(),
            model_config: serde_json::to_value(model_config)?,
            model_weights: weights,
            model_format: MODEL_FORMAT.to_string(),
            shape_in,
            shape_out,
            trained_at: Utc::now(),
            trained_on_checksum: dataset.checksum(),
        };

        let encoder = self
            .encoders
            .chain(|mut tx| {
                let encoder = encoder;
                Box::pin(async move {
                    let encoder = tx.store_encoder(&encoder).await?;
                    Ok((tx, encoder))
                })
            })
            .await?;

        Ok(encoder)
    }

    fn train_model<D>(
        &self,
        train_config: &TrainModelConfig,
        dataset: &D,
    ) -> Result<(Vec<u8>, ModelConfig), super::Error>
    where
        D: SequenceDataSource,
    {
        match train_config {
            TrainModelConfig::Lstm(cfg) => {
                let device = <TrainingBackend as Backend>::Device::default();

                let (model, _report) =
                    train_autoencoder::<TrainingBackend>(&device, dataset, cfg.clone());

                let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();

                let bytes =
                    Recorder::<InferenceBackend>::record(&recorder, model.into_record(), ())?;

                let model_config = ModelConfig::Lstm(cfg.autoencoder_config());

                Ok((bytes, model_config))
            }
        }
    }
}

#[cfg(test)]
mod test_support {
    use super::super::encoder::dataset::{SequenceDataset, SequenceSample};
    use super::super::encoder::lstm::AutoencoderConfig;
    use super::*;
    use crate::bootstrap::ApplicationBuilder;
    use crate::services::indexing::TrainModelConfig;
    use anyhow::Context;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn_ndarray::NdArray;
    use fx_event_bus::test_tools::get_unacknowledged_events;
    use sqlx::{PgPool, Row};
    use std::sync::Arc;
    use std::time::Duration;

    pub(crate) struct TestContext {
        pub(crate) service: Arc<Service>,
        pub(crate) embeddings: Arc<embeddings::Repository>,
        pub(crate) encoders: Arc<encoders::Repository>,
    }

    pub(crate) async fn build_context(pool: &PgPool) -> anyhow::Result<TestContext> {
        let app_builder = ApplicationBuilder::default().with_pool(pool.clone());
        let service = Arc::new(app_builder.indexing_service().build());

        let embeddings_repo = Arc::new(embeddings::Repository::new(pool.clone()));
        let ttl = Duration::from_secs(60 * 10);
        let capacity = 20;
        let encoders_repo = Arc::new(encoders::Repository::new(pool.clone(), ttl, capacity));

        Ok(TestContext {
            service,
            embeddings: embeddings_repo,
            encoders: encoders_repo,
        })
    }

    pub(crate) async fn seed_encoder(pool: &PgPool) -> anyhow::Result<Uuid> {
        type TestBackend = NdArray<f32>;

        let config = AutoencoderConfig {
            input_size: 2,
            hidden_size: 4,
            latent_size: 2,
        };

        let device = <TestBackend as Backend>::Device::default();
        let model = lstm::LstmAutoencoder::<TestBackend>::new(&device, config);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let model_bytes = Recorder::<TestBackend>::record(&recorder, model.into_record(), ())?;

        let encoder = encoders::Encoder {
            id: Uuid::now_v7(),
            model_type: "lstm".to_string(),
            model_config: serde_json::to_value(ModelConfig::Lstm(config))?,
            model_weights: model_bytes,
            model_format: MODEL_FORMAT.to_string(),
            shape_in: vec![2],
            shape_out: 2,
            trained_at: Utc::now(),
            trained_on_checksum: vec![1, 2, 3, 4],
        };

        let stored = encoders::store_encoder(pool, &encoder)
            .await
            .context("store encoder")?;

        Ok(stored.id())
    }

    pub(crate) fn encode_input(values: (f32, f32)) -> EncodeInput {
        EncodeInput {
            values: vec![values.0, values.1],
            dimensions: vec![1, 2],
        }
    }

    pub(crate) fn dataset_for_training() -> SequenceDataset {
        SequenceDataset::new(
            vec![
                SequenceSample {
                    steps: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
                },
                SequenceSample {
                    steps: vec![vec![0.5, 0.6]],
                },
            ],
            2,
        )
    }

    pub(crate) fn training_config() -> TrainModelConfig {
        TrainModelConfig::Lstm(AutoencoderTrainConfig {
            input_size: 2,
            hidden_size: 4,
            latent_size: 2,
            batch_size: 2,
            epochs: 1,
            learning_rate: 1e-3,
        })
    }

    pub(crate) async fn event_count(pool: &PgPool) -> anyhow::Result<i64> {
        let count = get_unacknowledged_events(pool).await?;
        Ok(count)
    }

    pub(crate) async fn latest_pairing_state(
        pool: &PgPool,
        encoder_id: &Uuid,
        type_hash: i32,
    ) -> anyhow::Result<Option<bool>> {
        let row = sqlx::query(
            r#"
            SELECT is_enabled
            FROM fx_durable_ga.encoder_toggles
            WHERE encoder_id = $1 AND type_hash = $2
            ORDER BY timestamp DESC
            LIMIT 1
            "#,
        )
        .bind(encoder_id)
        .bind(type_hash)
        .fetch_optional(pool)
        .await?;

        let state = match row {
            Some(row) => Some(row.try_get("is_enabled")?),
            None => None,
        };

        Ok(state)
    }
}

#[cfg(test)]
mod tests_index_many {
    use super::test_support;
    use super::*;
    use crate::migrations;
    use sqlx::Row;

    #[sqlx::test(migrations = false)]
    async fn it_indexes_many(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let encoder_id = test_support::seed_encoder(&pool).await?;

        let inputs = vec![
            test_support::encode_input((1.0, 2.0)),
            test_support::encode_input((3.0, 4.0)),
        ];
        let tags = vec!["type:Genotype".to_string(), "context:test".to_string()];

        let before_events = test_support::event_count(&pool).await?;

        let embedding_ids = ctx.service.index_many(&encoder_id, &inputs, &tags).await?;

        assert_eq!(embedding_ids.len(), inputs.len());

        let rows = sqlx::query(
            r#"
            SELECT id, encoded_with
            FROM fx_durable_ga.embeddings
            WHERE id = ANY($1)
            ORDER BY encoded_at
            "#,
        )
        .bind(&embedding_ids)
        .fetch_all(&pool)
        .await?;

        assert_eq!(rows.len(), inputs.len());
        for row in rows {
            let encoded_with: Uuid = row.try_get("encoded_with")?;
            assert_eq!(encoded_with, encoder_id);
        }

        let tag_rows = sqlx::query(
            r#"
            SELECT embedding_id, tag_name
            FROM fx_durable_ga.embedding_tags
            WHERE embedding_id = ANY($1)
            ORDER BY embedding_id, tag_name
            "#,
        )
        .bind(&embedding_ids)
        .fetch_all(&pool)
        .await?;

        assert_eq!(tag_rows.len(), embedding_ids.len() * tags.len());

        let after_events = test_support::event_count(&pool).await?;
        assert_eq!(after_events - before_events, embedding_ids.len() as i64);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_missing_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let missing_encoder = Uuid::now_v7();
        let inputs = vec![test_support::encode_input((5.0, 6.0))];

        let err = ctx
            .service
            .index_many(&missing_encoder, &inputs, &[])
            .await
            .expect_err("expected missing encoder error");

        match err {
            super::super::Error::EncodersRepository(encoders::Error::NotFound(id)) => {
                assert_eq!(id, missing_encoder)
            }
            other => panic!("unexpected error: {other:?}"),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests_add_tag {
    use super::test_support;
    use super::*;
    use crate::migrations;
    use sqlx::Row;

    #[sqlx::test(migrations = false)]
    async fn it_adds_tag(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let embedding = Embedding::new(Uuid::now_v7(), Utc::now(), [0.0; 256]);
        let embedding_id = *embedding.id();

        ctx.embeddings
            .chain(|mut tx| {
                let embedding = embedding.clone();
                Box::pin(async move {
                    tx.store_embeddings(&[embedding]).await?;
                    Ok((tx, ()))
                })
            })
            .await?;

        let tags = ctx.service.add_tag(&[embedding_id], "favorite").await?;

        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].tag_name, "favorite");
        assert_eq!(tags[0].embedding_id, embedding_id);

        let stored = sqlx::query(
            r#"
            SELECT tag_name
            FROM fx_durable_ga.embedding_tags
            WHERE embedding_id = $1
            "#,
        )
        .bind(embedding_id)
        .fetch_all(&pool)
        .await?;

        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].try_get::<String, _>("tag_name")?, "favorite");

        Ok(())
    }
}

#[cfg(test)]
mod tests_find_similar {
    use super::test_support;
    use super::*;
    use crate::migrations;

    #[sqlx::test(migrations = false)]
    async fn it_finds_similar(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let encoder_id = Uuid::now_v7();
        let now = Utc::now();
        let tag = "cluster".to_string();

        let build_embedding = |value: f32| {
            let mut vec = [0.0f32; 256];
            vec[0] = value;
            Embedding::new(encoder_id, now, vec)
        };

        let anchor = build_embedding(0.5);
        let close = build_embedding(0.55);
        let far = build_embedding(0.9);

        let embeddings = vec![anchor.clone(), close.clone(), far.clone()];
        let tags: Vec<Tag> = embeddings
            .iter()
            .map(|embedding| Tag::new(tag.clone(), *embedding.id(), now))
            .collect();

        ctx.embeddings
            .chain(|mut tx| {
                let embeddings = embeddings.clone();
                let tags = tags.clone();
                Box::pin(async move {
                    tx.store_embeddings(&embeddings).await?;
                    tx.store_tags(&tags).await?;
                    Ok((tx, ()))
                })
            })
            .await?;

        let similar = ctx.service.find_similar(anchor.id(), &tag, 2).await?;

        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].embedding_id(), close.id());
        assert_eq!(similar[1].embedding_id(), far.id());

        Ok(())
    }
}

#[cfg(test)]
mod tests_train_encoder {
    use super::test_support;
    use super::*;
    use crate::migrations;

    #[sqlx::test(migrations = false)]
    async fn it_trains_an_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let dataset = test_support::dataset_for_training();
        let config = test_support::training_config();
        let encoder_id = Uuid::now_v7();

        let checksum = dataset.checksum();
        let encoder = ctx
            .service
            .train_encoder(encoder_id, config.clone(), dataset)
            .await?;

        assert_eq!(encoder.id(), encoder_id);
        assert_eq!(encoder.model_type, "lstm");
        assert_eq!(encoder.shape_in, vec![2]);
        assert_eq!(encoder.shape_out, 2);
        assert_eq!(encoder.trained_on_checksum, checksum);

        let fetched = ctx.encoders.get_encoder(&encoder_id).await?;
        assert_eq!(fetched.id(), encoder_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests_toggling_encoder_pairings {
    use super::test_support;
    use crate::migrations;

    #[sqlx::test(migrations = false)]
    async fn it_enables_encoder_pairing(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let encoder_id = test_support::seed_encoder(&pool).await?;
        let type_hash = 10_001;
        let before_events = test_support::event_count(&pool).await?;

        let enabled = ctx
            .service
            .enable_encoder_pairing(&encoder_id, type_hash)
            .await?;

        assert!(enabled);

        let state = test_support::latest_pairing_state(&pool, &encoder_id, type_hash).await?;
        assert_eq!(state, Some(true));

        let after_events = test_support::event_count(&pool).await?;
        assert_eq!(after_events, before_events + 1);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_disabled_encoder_pairing(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let encoder_id = test_support::seed_encoder(&pool).await?;
        let type_hash = 10_002;

        ctx.service
            .enable_encoder_pairing(&encoder_id, type_hash)
            .await?;

        let before_disable_events = test_support::event_count(&pool).await?;

        let disabled = ctx
            .service
            .disable_encoder_pairing(&encoder_id, type_hash)
            .await?;

        assert!(!disabled);

        let state = test_support::latest_pairing_state(&pool, &encoder_id, type_hash).await?;
        assert_eq!(state, Some(false));

        let active_pairings = ctx.encoders.get_encoder_pairings(&[type_hash]).await?;
        assert!(active_pairings.is_empty());

        let after_disable_events = test_support::event_count(&pool).await?;
        assert_eq!(after_disable_events, before_disable_events + 1);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_only_fires_events_on_state_changed(pool: sqlx::PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;

        let ctx = test_support::build_context(&pool).await?;
        let encoder_id = test_support::seed_encoder(&pool).await?;
        let type_hash = 10_003;

        let before = test_support::event_count(&pool).await?;
        let first = ctx
            .service
            .disable_encoder_pairing(&encoder_id, type_hash)
            .await?;
        assert!(!first);
        let after_first = test_support::event_count(&pool).await?;
        assert_eq!(after_first, before + 1);

        let second = ctx
            .service
            .disable_encoder_pairing(&encoder_id, type_hash)
            .await?;
        assert!(!second);
        let after_second = test_support::event_count(&pool).await?;
        assert_eq!(after_second, after_first);

        Ok(())
    }
}
