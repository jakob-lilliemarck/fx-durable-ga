use super::encoder::dataset::SequenceDataSource;
use super::encoder::lstm::{self, AutoencoderModel};
use super::encoder::train::{AutoencoderTrainConfig, train_autoencoder};
use super::repositories::embeddings;
use super::repositories::embeddings::{
    EmbeddingNew, RequestedEmbedding, SearchRequestedEmbeddingsFilter, TagNew,
};
use super::{
    Digest,
    repositories::encoders::{self, Encoder, EncoderAvailability},
};
use crate::infrastructure::db;
use crate::repositories::genotypes::{Identifiable, TypeName};
use crate::services::indexing::EncodeInput;
use crate::services::indexing::IndexerErased;
use crate::services::indexing::events::{EmbeddingCreatedEvent, EncoderAvailableEvent};
use crate::services::indexing::indexable;
use crate::services::indexing::jobs::TrainEncoderMessage;
use crate::services::locking;
use burn::backend::Autodiff;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_ndarray::NdArray;
use chrono::Utc;
use futures::lock::Mutex;
use fx_mq_jobs::Queries;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

type CpuBackend = NdArray<f32>;
type TrainingBackend = Autodiff<CpuBackend>;
type InferenceBackend = CpuBackend;

pub(crate) const MODEL_FORMAT: &str = "burn-bin-f32";

pub struct Service {
    pub(super) embeddings_ro: embeddings::Read,
    pub(super) embeddings_wr: embeddings::Write,
    pub(super) encoders_ro: encoders::Read,
    pub(super) encoders_wr: encoders::Write,
    pub(super) registry: Arc<Mutex<indexable::Registry>>,
    pub(super) locking: Arc<locking::Service>,
    pub(super) mq: Arc<Queries>,
}

// Improve construction of these - make constructors accessible on the service?
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TrainModelConfig {
    Lstm(AutoencoderTrainConfig),
}

impl Service {
    /// Batches an iterator into fixed-size chunks for streaming processing.
    ///
    /// This is useful when you want to handle large inputs in smaller groups
    /// (e.g., dispatching jobs or database writes) without allocating all
    /// batches up front. The iterator yields `Vec<T>` batches as it consumes
    /// the input, with a final partial batch if items remain.
    pub(crate) fn batch_iter<T, I>(iter: I, batch_size: usize) -> impl Iterator<Item = Vec<T>>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut current: Vec<T> = Vec::with_capacity(batch_size);

        std::iter::from_fn(move || {
            while current.len() < batch_size {
                let Some(item) = iter.next() else {
                    break;
                };
                current.push(item);
            }

            if current.is_empty() {
                // `from_fn` expects `None` to signal the iterator is finished.
                None
            } else {
                // Move the accumulated batch out without cloning; replace `current` with a
                // pre-sized buffer for the next batch.
                let batch = std::mem::replace(&mut current, Vec::with_capacity(batch_size));
                Some(batch)
            }
        })
    }

    /// Creates a new indexing service with the given dependencies.
    pub fn new(
        embeddings_ro: embeddings::Read,
        embeddings_wr: embeddings::Write,
        encoders_ro: encoders::Read,
        encoders_wr: encoders::Write,
        registry: Arc<Mutex<indexable::Registry>>,
        locking: Arc<locking::Service>,
        mq: Arc<fx_mq_jobs::Queries>,
    ) -> Self {
        Self {
            embeddings_ro,
            embeddings_wr,
            encoders_ro,
            encoders_wr,
            registry,
            locking,
            mq,
        }
    }

    /// Try to get an encoder from the repository, if no encoder could be found
    /// fallback to training an encoder.
    ///
    /// This method uses a lock to de-race the application from training multiple models with the
    /// same digest
    #[instrument(level = "info", skip(self))]
    async fn try_get_encoder(
        &self,
        indexer_id: &Digest,
    ) -> Result<Option<Arc<Encoder>>, super::Error> {
        let result = self.encoders_ro.get_encoder(indexer_id).await;

        let Err(err) = result else {
            return result.map(|encoder| Some(encoder)).map_err(Into::into);
        };

        let encoders::Error::NotFound(_) = err else {
            return Err(super::Error::from(err));
        };

        let key = format!("request_encoder_training:{}", indexer_id.as_str());
        self.locking.lock_while(&key, || async {
            let availability = self.encoders_ro.get_encoder_availability(indexer_id).await?;

            if matches!(availability, EncoderAvailability::Available) {
                tracing::warn!(message="the encoder was not found but should be available. Retrying..", indexer_id=%indexer_id);
                let encoder = self.encoders_ro.get_encoder(indexer_id).await?;
                return Ok(Some(encoder));
            }

            if matches!(availability, EncoderAvailability::NotFound) {
                tracing::info!("encoder not found");
                self.request_encoder(indexer_id).await?;
                return Ok(None)
            }

            Ok::<Option<Arc<Encoder>>, super::Error>(None)
        }).await?
    }

    /// Atomically dispatches a training job and sets the encoder state from "NotFound" to "NotReady"
    #[instrument(level = "info", skip(self))]
    async fn request_encoder(&self, indexer_id: &Digest) -> Result<(), super::Error> {
        tracing::info!("encoder requested");
        let training_job = TrainEncoderMessage::new(*indexer_id);
        let indexer_id = *indexer_id;
        let mq = self.mq.clone();

        db::begin(self.encoders_wr.clone(), |tx| {
            Box::pin(async move {
                encoders::WriteTx::new(tx)
                    .store_encoder_availability(&indexer_id, false)
                    .await?;

                let mut publisher = fx_mq_jobs::Publisher::new_tx(tx, &mq);
                publisher.publish(&training_job).await?;

                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn get_deferred_items(
        &self,
        filter: &SearchRequestedEmbeddingsFilter,
        limit: i64,
    ) -> Result<Vec<RequestedEmbedding>, super::Error> {
        let deferred = self
            .embeddings_ro
            .search_requested_embeddings(filter, limit)
            .await?;

        Ok(deferred)
    }

    /// Atomically defers indexation of a collection of indexable items.
    #[instrument(level = "debug", skip(self, items))]
    async fn defer_items<I>(
        &self,
        indexer_id: &Digest,
        items: &[(I, Vec<String>)],
        metadata: Option<serde_json::Value>,
    ) -> Result<(), super::Error>
    where
        I: TypeName + Identifiable + 'static,
    {
        let timestamp = Utc::now();

        let metadata = metadata.unwrap_or(serde_json::json!({}));

        let deferred = items
            .iter()
            .map(|(item, _)| {
                RequestedEmbedding::new(
                    item.id(),
                    item.type_name().to_string(),
                    *indexer_id,
                    metadata.clone(),
                    timestamp,
                )
            })
            .collect::<Vec<RequestedEmbedding>>();

        db::begin(self.embeddings_wr.clone(), |tx| {
            Box::pin(async move {
                embeddings::WriteTx::new(tx)
                    .store_requested_embeddings(&deferred)
                    .await?;

                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    // FIXME!
    //
    // Make this method pub(crate)
    // The indexing service is intended to be wrapped by an adapter and not exposed directly
    //
    /// Index many items
    /// Returns Some(embedding_ids) if the indexer was available and indexes could
    /// be computed at the time of the call.
    ///
    /// Returns None if the requested indexer was unavailable at the time time of the call.
    /// Callers of this method should listen for `EncoderTrainedEvent` and to check if
    /// there's deferred indexing work to re-schedule at that time.
    #[instrument(level = "info", skip(self, items, metadata), fields(indexer_id, item_count = %items.len()))]
    pub async fn index_many<I>(
        &self,
        indexer_id: &Digest,
        items: &[(I, Vec<String>)],
        metadata: Option<serde_json::Value>,
    ) -> Result<Option<Vec<Uuid>>, super::Error>
    where
        I: Serialize + TypeName + Identifiable + 'static,
    {
        let timestamp = Utc::now();

        if items.is_empty() {
            return Ok(None);
        }

        let indexer = match self.get_indexer(indexer_id).await {
            Ok(indexer) => indexer,
            Err(indexable::Error::NotFound { id }) => {
                tracing::warn!(message="index_many called with an unknown indexer id", requested_id=%indexer_id, not_found_id=%id);
                return Ok(None);
            }
            Err(err) => return Err(err.into()),
        };

        let encoder = match self.try_get_encoder(indexer_id).await? {
            Some(encoder) => encoder,
            None => {
                self.defer_items(indexer_id, items, metadata).await?;
                return Ok(None);
            }
        };

        let count = items.len();
        let mut tags = Vec::with_capacity(count);
        let mut embeddings = Vec::with_capacity(count);
        let mut handled = Vec::with_capacity(count);

        for (item, t) in items {
            let json = serde_json::to_value(item)?;

            let preprocessed = indexer.preprocess(json)?;

            let embedding_raw = Self::encode(&preprocessed, &encoder)?;

            let embedding = EmbeddingNew::new(encoder.digest, timestamp, embedding_raw);

            for tag in t.iter() {
                tags.push(TagNew::new(tag, *embedding.id(), timestamp));
            }

            embeddings.push(embedding);

            handled.push((item.id(), indexer_id.clone()))
        }

        let embedding_ids = db::begin(self.embeddings_wr.clone(), |tx| {
            Box::pin(async move {
                let mut embeddings_wr = embeddings::WriteTx::new(tx);
                let embedding_ids = embeddings_wr.store_embeddings(&embeddings).await?;

                // Mark all as handled - only mutates found records, ignores any unknown ids
                embeddings_wr
                    .set_requested_embeddings_handled_at(&handled, timestamp)
                    .await?;

                let tags = embeddings_wr.store_tags(&tags).await?;

                let events = tags
                    .iter()
                    .fold(
                        HashMap::with_capacity(embedding_ids.len()),
                        |mut acc, (embedding_id, tag_name)| {
                            acc.entry(*embedding_id)
                                .or_insert_with(|| EmbeddingCreatedEvent {
                                    embedding_id: *embedding_id,
                                    tags: vec![],
                                })
                                .tags
                                .push(tag_name.to_string());
                            acc
                        },
                    )
                    .into_values()
                    .collect::<Vec<_>>();

                let mut publisher = fx_event_bus::Publisher::new_tx(tx);
                publisher.publish_many(&events).await?;

                Ok(embedding_ids)
            })
        })
        .await?;

        Ok(Some(embedding_ids))
    }

    #[instrument(level = "debug", skip_all)]
    fn encode(
        input: &EncodeInput,
        encoder: &Arc<Encoder>,
    ) -> Result<embeddings::EmbeddingValue, super::Error> {
        let model_cfg: lstm::AutoencoderConfig =
            serde_json::from_value(encoder.model_config.clone())?;

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

    /// Tags the given embeddings with the specified tag.
    #[instrument(level = "debug", skip(self))]
    pub async fn tag_embeddings<'a>(
        &self,
        embedding_ids: &'a [Uuid],
        tag: &'a str,
    ) -> Result<(), super::Error> {
        let tagged_at = Utc::now();

        let tags = embedding_ids
            .iter()
            .map(|id| TagNew::new(tag, id.clone(), tagged_at))
            .collect::<Vec<TagNew>>();

        db::begin(self.embeddings_wr.clone(), |tx| {
            Box::pin(async move {
                let res = embeddings::WriteTx::new(tx).store_tags(&tags).await?;
                Ok(res)
            })
        })
        .await?;

        Ok(())
    }

    /// Enables an encoder, making it available for indexing.
    #[instrument(level = "debug", skip(self))]
    pub async fn enable_encoder(&self, encoder_digest: Digest) -> Result<bool, super::Error> {
        let is_enabled = self.toggle_encoder(encoder_digest, true).await?;
        Ok(is_enabled)
    }

    /// Disables an encoder, preventing it from being used for indexing.
    #[instrument(level = "debug", skip(self))]
    pub async fn disable_encoder(&self, encoder_digest: Digest) -> Result<bool, super::Error> {
        let is_enabled = self.toggle_encoder(encoder_digest, false).await?;
        Ok(is_enabled)
    }

    #[instrument(level = "info", skip(self))]
    async fn toggle_encoder(
        &self,
        encoder_digest: Digest,
        is_enabled: bool,
    ) -> Result<bool, super::Error> {
        let is_enabled = db::begin(self.encoders_wr.clone(), |tx| {
            Box::pin(async move {
                let (availability, was_changed) = encoders::WriteTx::new(tx)
                    .store_encoder_availability(&encoder_digest, is_enabled)
                    .await?;

                let mut publisher = fx_event_bus::Publisher::new_tx(tx);

                if was_changed && availability == EncoderAvailability::Available {
                    let event = EncoderAvailableEvent {
                        indexer_id: encoder_digest,
                    };
                    tracing::info!("encoder available");
                    publisher.publish(event).await?;
                }

                Ok(is_enabled)
            })
        })
        .await?;

        Ok(is_enabled)
    }

    #[instrument(level = "info", skip(self))]
    pub(crate) async fn train_encoder(&self, indexer_id: &Digest) -> Result<Digest, super::Error> {
        let indexer = self.get_indexer(indexer_id).await?;

        let dataset = indexer.dataset();

        let train_config = indexer.train_config();

        let (weights, autoencoder_cfg) = self.train_model(&train_config, &dataset)?;

        let (shape_in, shape_out) = (
            vec![autoencoder_cfg.input_size as i32],
            autoencoder_cfg.latent_size as i32,
        );

        let encoder = encoders::Encoder {
            digest: *indexer_id,
            encodable_type_name: indexer.encodable_type_name().to_string(),
            model_config: serde_json::to_value(&autoencoder_cfg)?,
            model_weights: weights,
            model_format: MODEL_FORMAT.to_string(),
            shape_in,
            shape_out,
        };

        let encoder = db::begin(self.encoders_wr.clone(), |tx| {
            Box::pin(async move {
                let encoder = encoders::WriteTx::new(tx)
                    .store_encoder(&encoder, &Utc::now())
                    .await?;

                tracing::info!("encoder stored");

                Ok(encoder)
            })
        })
        .await?;

        Ok(encoder.digest)
    }

    #[instrument(level = "debug", skip(self, dataset))]
    fn train_model<D>(
        &self,
        train_config: &TrainModelConfig,
        dataset: &D,
    ) -> Result<(Vec<u8>, lstm::AutoencoderConfig), super::Error>
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

                Ok((bytes, cfg.autoencoder_config()))
            }
        }
    }

    /// Returns all indexer IDs registered for the given type name.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn get_indexer_ids_of_type(&self, encodable_type_name: &str) -> Vec<Digest> {
        let lock = self.registry.lock().await;
        lock.get_indexers_of_type(encodable_type_name)
            .map(|indexers| indexers.iter().copied().collect())
            .unwrap_or_default()
    }

    async fn get_indexer(
        &self,
        indexer_id: &Digest,
    ) -> Result<Arc<dyn IndexerErased>, indexable::Error> {
        let lock = self.registry.lock().await;
        let indexer = lock.get_indexer(indexer_id)?;
        Ok(indexer.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::Service;

    #[test]
    fn batch_iter_splits_items_into_full_and_partial_batches() {
        let items = vec![1, 2, 3, 4, 5, 6, 7];
        let batch_size = 3;

        let batches = Service::batch_iter(items, batch_size).collect::<Vec<_>>();

        assert_eq!(batches, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
    }

    #[test]
    fn batch_iter_returns_empty_when_batch_size_is_zero() {
        let items = vec![1, 2, 3];
        let batch_size = 0;

        let batches = Service::batch_iter(items, batch_size).collect::<Vec<_>>();

        assert!(batches.is_empty());
    }
}
