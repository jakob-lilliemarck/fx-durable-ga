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
mod tests_index_many {
    #[sqlx::test(migrations = false)]
    async fn it_indexes_many(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // index many
        // assert embeddings were created
        // assert tags were created
        // assert events were dispatched
        todo!("it_indexes_many")
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_missing_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // assert that it errors on missing encoder
        todo!("it_errors_on_missing_encoder")
    }
}

#[cfg(test)]
mod tests_add_tag {
    #[sqlx::test(migrations = false)]
    async fn it_adds_tag(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // Add a tag to an embedding
        // Then just assert the returned value is as expected
        todo!("it_adds_tag")
    }
}

#[cfg(test)]
mod tests_find_similar {
    #[sqlx::test(migrations = false)]
    async fn it_finds_similar(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // store three embeddings with values that are easily human readable and assessable
        // call find_similar for one of the embeddings
        // assert the results are in the expected order
        // maybe assert approx similarity
        todo!("it_finds_similar")
    }
}

#[cfg(test)]
mod tests_train_encoder {
    #[sqlx::test(migrations = false)]
    async fn it_trains_an_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // train an encoder
        // We're not asserting the quality of the encoded values here
        // This is just a smoke-test to make sure training works
        todo!("it_trains_an_encoder")
    }
}

#[cfg(test)]
mod tests_toggling_encoder_pairings {
    #[sqlx::test(migrations = false)]
    async fn it_enables_encoder_pairing(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // enable an encoder pairing
        // assert the returned value
        // assert an event was created with the expected values
        todo!("it_enables_encoder_pairing")
    }

    #[sqlx::test(migrations = false)]
    async fn it_disabled_encoder_pairing(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // disable an encoder pairing
        // assert the returned value
        // assert an event was created with the expected values
        todo!("it_disabled_encoder_pairing")
    }

    #[sqlx::test(migrations = false)]
    async fn it_only_fires_events_on_state_changed(pool: sqlx::PgPool) -> anyhow::Result<()> {
        // disable an encoder pairing twice
        // assert the returned values
        // assert that only one events were created
        todo!("it_only_fires_events_on_state_changed")
    }
}
