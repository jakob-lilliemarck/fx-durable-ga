use super::encoder::dataset::SequenceDataSource;
use super::encoder::lstm::{AutoencoderConfig, AutoencoderModel, LstmAutoencoder};
use super::encoder::train::{AutoencoderTrainConfig, train_autoencoder};
use crate::repositories::chainable::Chain;
use crate::repositories::embeddings::{Embedding, Similar, Tag};
use crate::repositories::encoders::Encoder;
use crate::repositories::{self, embeddings, encoders};
use burn::backend::Autodiff;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_ndarray::NdArray;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use uuid::Uuid;

type CpuBackend = NdArray<f32>;
type TrainingBackend = Autodiff<CpuBackend>;
type InferenceBackend = CpuBackend;
const MODEL_FORMAT: &str = "burn-bin-f32";

pub struct Service {
    embeddings: Arc<embeddings::Repository>,
    encoders: Arc<encoders::Repository>,
    loaded: Option<Encoder>,
}

pub struct EncodeInput {
    pub values: Vec<f32>,
    pub dimensions: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelConfig {
    Lstm(AutoencoderConfig),
}

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
    pub fn new(
        embeddings: Arc<embeddings::Repository>,
        encoders: Arc<encoders::Repository>,
    ) -> Self {
        Self {
            embeddings,
            encoders,
            loaded: None,
        }
    }

    // Load a model into the service cache
    // Models must be loaded prior to calling encode
    pub async fn load(&mut self, encoder_id: &Uuid) -> Result<(), super::Error> {
        if self
            .loaded
            .as_ref()
            .map_or(true, |e| return &e.id != encoder_id)
        {
            self.loaded = match self.encoders.get_encoder(encoder_id).await? {
                Some(encoder) => Some(encoder),
                None => {
                    return Err(super::Error::NotFoundEncoder(encoder_id.clone()));
                }
            };
        }

        Ok(())
    }

    // Encodes inputs to an embeddings using the specified encoder and store it with tags.
    // Returns ids of stored embeddings
    pub async fn index(
        &self,
        inputs: &[EncodeInput],
        tag_names: &[&str],
    ) -> Result<Vec<Uuid>, super::Error> {
        let now = Utc::now();
        let mut embeddings = Vec::with_capacity(inputs.len());
        for i in inputs {
            let (encoded_with, value) = self.encode(i)?;
            embeddings.push(Embedding::new(encoded_with, now, value))
        }

        let mut tags = Vec::with_capacity(embeddings.len() * tag_names.len());

        for ref t in tag_names {
            for e in embeddings.iter() {
                tags.push(Tag::new(t.to_string(), *e.id(), now));
            }
        }

        let embedding_ids = self
            .embeddings
            .chain(|mut tx| {
                Box::pin(async move {
                    let embeddings = tx.store_embeddings(&embeddings).await?;
                    tx.store_tags(&tags).await?;
                    let ids = embeddings.iter().map(|e| *e.id()).collect();
                    Ok((tx, ids))
                })
            })
            .await?;

        Ok(embedding_ids)
    }

    // Encodes input to an embedding using the specified encoder
    // Returns the raw embedding
    pub fn encode(
        &self,
        input: &EncodeInput,
    ) -> Result<(Uuid, repositories::embeddings::Value), super::Error> {
        let encoder = match self.loaded {
            Some(ref encoder) => encoder,
            None => return Err(super::Error::NoEncoder),
        };

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
            LstmAutoencoder::<InferenceBackend>::new(&device, model_cfg).load_record(record);

        let tensor = Tensor::<InferenceBackend, 3>::from_data(
            TensorData::new(input.values.clone(), [1, seq_len, input_size]),
            &device,
        );
        let output = model.forward(tensor);
        let latent = output.latent.into_data().into_vec::<f32>()?;

        let mut embedding = [0.0f32; 256];
        let copy_len = latent.len().min(embedding.len());
        embedding[..copy_len].copy_from_slice(&latent[..copy_len]);

        Ok((encoder.id, embedding))
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

    pub async fn get_encoder(&self, encoder_id: Uuid) -> Result<Option<Encoder>, super::Error> {
        let encoder = self.encoders.get_encoder(&encoder_id).await?;
        Ok(encoder)
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
        if let Some(encoder) = self.encoders.get_encoder(&encoder_id).await? {
            return Ok(encoder);
        };

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
