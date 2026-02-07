use super::encoder::dataset::SequenceDataSource;
use super::encoder::lstm::{AutoencoderConfig, AutoencoderModel, LstmAutoencoder};
use super::encoder::train::{AutoencoderTrainConfig, train_autoencoder};
use crate::repositories::chainable::Chain;
use crate::repositories::embeddings::{EmbeddingsRepository, Tag};
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
use uuid::{NoContext, Timestamp, Uuid};

type CpuBackend = NdArray<f32>;
type TrainingBackend = Autodiff<CpuBackend>;
type InferenceBackend = CpuBackend;
const MODEL_FORMAT: &str = "burn-bin-f32";

pub struct Service {
    embeddings: Arc<embeddings::Repository>,
    encoders: Arc<encoders::Repository>,
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

pub struct Filter {
    encoder_id: Option<Uuid>,
    tags: Option<Vec<String>>,
    ids: Option<Vec<Uuid>>,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            encoder_id: None,
            tags: None,
            ids: None,
        }
    }
}

impl Filter {
    pub fn with_encoder_id(&mut self, encoder_id: Uuid) {
        self.encoder_id = Some(encoder_id)
    }

    pub fn with_tag(&mut self, tag: String) {
        if let Some(ref mut tags) = self.tags {
            tags.push(tag);
        } else {
            self.tags = Some(vec![tag]);
        }
    }

    pub fn with_id(&mut self, id: Uuid) {
        if let Some(ref mut ids) = self.ids {
            ids.push(id);
        } else {
            self.ids = Some(vec![id]);
        }
    }
}

pub struct SimilarityResult {
    pub embedding_id: Uuid,
    pub distance: f32,
}

impl Service {
    pub fn new(
        embeddings: Arc<embeddings::Repository>,
        encoders: Arc<encoders::Repository>,
    ) -> Self {
        Self {
            embeddings,
            encoders,
        }
    }

    // Encodes input to an embedding using the specified encoder and stores it.
    // Returns ids of stored embeddings
    pub fn index(
        &self,
        encoder_id: &Uuid,
        input: &EncodeInput,
        tags: &[&str],
    ) -> Result<Vec<Uuid>, super::Error> {
        // Outputs
        unimplemented!()
    }

    // Encodes input to an embedding using the specified encoder
    // Returns the raw embedding
    pub async fn encode(
        &self,
        encoder_id: &Uuid,
        input: &EncodeInput,
    ) -> Result<repositories::embeddings::Value, super::Error> {
        let encoder = self.encoders.get(encoder_id).await?;

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
        let weights = encoder.model_weights.clone();
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

        Ok(embedding)
    }

    pub async fn add_tag<'a>(
        &self,
        embedding_ids: &'a [Uuid],
        tag: &'a str,
    ) -> Result<Vec<Tag>, super::Error> {
        let tag_hash: i64 = 1;
        let tagged_at = Utc::now();
        let tagged = embedding_ids
            .iter()
            .map(|id| (tag_hash, tag, id, &tagged_at));
        let tags = self.embeddings.clone().store_tags(tagged).await?;
        Ok(tags)
    }

    pub fn find_similar(&self, filter: &Filter) -> Result<Vec<Uuid>, super::Error> {
        let _ = filter;
        unimplemented!()
    }

    pub async fn train_encoder<D>(
        &self,
        train_config: TrainModelConfig,
        dataset: D,
    ) -> Result<Encoder, super::Error>
    where
        D: SequenceDataSource,
    {
        let (weights, model_config) = self.train_model(&train_config, &dataset)?;
        let encoder = self.build_encoder(&model_config, weights)?;

        let encoder = self
            .encoders
            .chain(|mut tx| {
                let encoder = encoder;
                Box::pin(async move {
                    let encoder = tx.store(&encoder).await?;
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

    fn build_encoder(
        &self,
        model_config: &ModelConfig,
        model_weights: Vec<u8>,
    ) -> Result<encoders::Encoder, super::Error> {
        let (shape_in, shape_out) = match model_config {
            ModelConfig::Lstm(cfg) => (vec![cfg.input_size as i32], cfg.latent_size as i32),
        };

        Ok(encoders::Encoder {
            id: Self::next_encoder_id(),
            model_type: model_config.model_type().to_string(),
            model_config: serde_json::to_value(model_config)?,
            model_weights,
            model_format: MODEL_FORMAT.to_string(),
            shape_in,
            shape_out,
            trained_at: Utc::now(),
            trained_on_checksum: Vec::new(),
        })
    }

    fn next_encoder_id() -> Uuid {
        let ts = Timestamp::now(NoContext);
        Uuid::new_v7(ts)
    }
}
