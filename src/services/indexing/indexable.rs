use crate::repositories::genotypes::TypeName;
use crate::services::indexing::{Digest as IndexerDigest, EncoderDigestError};
use crate::services::indexing::{TrainModelConfig, encoder::dataset::SequenceDataSource};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tracing::instrument;

#[derive(Serialize)]
struct SerializableAutoencoderTrainConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub latent_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: String,
}

#[derive(Serialize)]
enum SerializableTrainModelConfig {
    Lstm(SerializableAutoencoderTrainConfig),
}

impl From<&TrainModelConfig> for SerializableTrainModelConfig {
    fn from(config: &TrainModelConfig) -> Self {
        match config {
            TrainModelConfig::Lstm(cfg) => {
                SerializableTrainModelConfig::Lstm(SerializableAutoencoderTrainConfig {
                    input_size: cfg.input_size,
                    hidden_size: cfg.hidden_size,
                    latent_size: cfg.latent_size,
                    batch_size: cfg.batch_size,
                    epochs: cfg.epochs,
                    learning_rate: format!("{:.10e}", cfg.learning_rate),
                })
            }
        }
    }
}

pub struct EncodeInput {
    pub values: Vec<f32>,
    pub dimensions: Vec<usize>,
}

type Dataset = Arc<dyn SequenceDataSource>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("No indexer registered found for id='{id}'")]
    NotFound { id: IndexerDigest },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Type with name {type_name} can not be processed by encoder {indexer_id}")]
    IndexerMismatch {
        type_name: String,
        indexer_id: IndexerDigest,
    },

    #[error("Encoder digest error: {0}")]
    EncoderDigest(#[from] EncoderDigestError),
}

pub trait Indexer: TypeName + Send + Sync {
    type Type: DeserializeOwned;

    fn preprocess(&self, entity: &Self::Type) -> EncodeInput;
    fn dataset(&self) -> Dataset;
    fn training_config(&self) -> &TrainModelConfig;
}

pub trait IndexerErased: Send + Sync {
    fn preprocess(&self, json: serde_json::Value) -> Result<EncodeInput, Error>;
    fn dataset(&self) -> Dataset;
    fn train_config(&self) -> &TrainModelConfig;
    fn encodable_type_name(&self) -> &str;
}

impl<I> IndexerErased for I
where
    I: Indexer + 'static,
{
    fn preprocess(&self, json: serde_json::Value) -> Result<EncodeInput, Error> {
        let typed = serde_json::from_value::<I::Type>(json)?;

        Ok(Indexer::preprocess(self, &typed))
    }

    fn dataset(&self) -> Dataset {
        Indexer::dataset(self)
    }

    fn train_config(&self) -> &TrainModelConfig {
        Indexer::training_config(self)
    }

    fn encodable_type_name(&self) -> &str {
        self.type_name()
    }
}

pub struct Registry {
    relation: HashMap<String, HashSet<IndexerDigest>>,
    indexers: HashMap<IndexerDigest, Arc<dyn IndexerErased>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub(crate) fn new() -> Self {
        Self {
            relation: HashMap::new(),
            indexers: HashMap::new(),
        }
    }

    #[instrument(level = "info", skip(self, indexer))]
    pub fn register(&mut self, indexer: Arc<dyn IndexerErased>) -> Result<IndexerDigest, Error> {
        let digest = Self::get_erased_indexer_id(&indexer)?;

        let type_name = indexer.encodable_type_name().to_string();

        self.indexers.insert(digest, indexer);

        self.relation
            .entry(type_name.to_string())
            .or_insert_with(HashSet::new)
            .insert(digest);

        tracing::info!(message="registered indexer", id = %digest);
        Ok(digest)
    }

    pub(crate) fn preprocess<T>(
        &self,
        instance: &T,
        indexer_id: &IndexerDigest,
    ) -> Result<EncodeInput, Error>
    where
        T: TypeName + Serialize + 'static,
    {
        let type_name = instance.type_name();

        let indexer_ids = self
            .relation
            .get(type_name)
            .ok_or(Error::NotFound { id: *indexer_id })?;

        if !indexer_ids.contains(indexer_id) {
            return Err(Error::IndexerMismatch {
                type_name: type_name.to_string(),
                indexer_id: *indexer_id,
            });
        }

        let indexer = self.get_indexer(indexer_id)?;

        let json = serde_json::to_value(instance)?;

        let preprocessed = indexer.preprocess(json)?;

        Ok(preprocessed)
    }

    pub(crate) fn get_indexer<'a>(
        &self,
        id: &'a IndexerDigest,
    ) -> Result<&Arc<dyn IndexerErased>, Error> {
        let indexer = self.indexers.get(id).ok_or(Error::NotFound { id: *id })?;

        Ok(indexer)
    }

    pub fn get_indexer_id<I>(indexer: &I) -> Result<IndexerDigest, Error>
    where
        I: Indexer + 'static,
        I::Type: DeserializeOwned + 'static,
    {
        let mut hasher = Sha256::new();

        let type_name_bytes = indexer.type_name().as_bytes();
        hasher.update(type_name_bytes);

        let dataset_digest_bytes = indexer.dataset().checksum();
        hasher.update(dataset_digest_bytes);

        let serializable_config = SerializableTrainModelConfig::from(indexer.training_config());
        let config_bytes = serde_json::to_vec(&serializable_config)?;
        hasher.update(config_bytes);

        let result = hasher.finalize();
        let hex = format!("{:x}", result);
        let digest = IndexerDigest::from_hex(&hex)?;

        Ok(digest)
    }

    pub fn get_erased_indexer_id(indexer: &Arc<dyn IndexerErased>) -> Result<IndexerDigest, Error> {
        let mut hasher = Sha256::new();

        let type_name_bytes = indexer.encodable_type_name().as_bytes();
        hasher.update(type_name_bytes);

        let dataset_digest_bytes = indexer.dataset().checksum();
        hasher.update(dataset_digest_bytes);

        let serializable_config = SerializableTrainModelConfig::from(indexer.train_config());
        let config_bytes = serde_json::to_vec(&serializable_config)?;
        hasher.update(config_bytes);

        let result = hasher.finalize();
        let hex = format!("{:x}", result);
        let digest = IndexerDigest::from_hex(&hex)?;

        Ok(digest)
    }

    pub fn get_indexers_of_type<'a>(
        &'a self,
        encodable_type_name: &'a str,
    ) -> Option<&'a HashSet<IndexerDigest>> {
        self.relation.get(encodable_type_name)
    }
}
