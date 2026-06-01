use crate::services::indexing::{Indexer, IndexerErased};
use serde::{Serialize, de::DeserializeOwned};
use std::{collections::HashMap, sync::Arc};
use tracing::instrument;

pub struct OptimizationService<T, O>
where
    O: Optimizer<Type = T>,
    T: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    pub type_name: &'static str,
    pub optimizer: O,
    pub indexers: Vec<Arc<dyn IndexerErased>>,
}

impl<T, O> OptimizationService<T, O>
where
    O: Optimizer<Type = T>,
    T: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    pub fn new(type_name: &'static str, optimizer: O) -> Self {
        Self {
            type_name,
            optimizer,
            indexers: Vec::new(),
        }
    }

    pub fn with_indexer<I>(mut self, indexer: I) -> Self
    where
        I: Indexer<Type = T> + 'static,
    {
        self.indexers.push(Arc::new(indexer));
        self
    }
}

pub trait Optimizer: Send + Sync {
    type Type: Serialize + DeserializeOwned + Send + Sync;

    fn random(&self) -> anyhow::Result<Self::Type>;
    fn crossover(
        &self,
        parent1: Self::Type,
        parent2: Self::Type,
    ) -> anyhow::Result<Self::Type>;
    fn mutate(
        &self,
        instance: &mut Self::Type,
    ) -> anyhow::Result<()>;
}

pub(crate) trait OptimizerErased: Send + Sync {
    fn random(&self) -> anyhow::Result<serde_json::Value>;
    fn crossover(
        &self,
        parent1: serde_json::Value,
        parent2: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value>;
    fn mutate(
        &self,
        instance: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value>;
}

impl<T> OptimizerErased for T
where
    T: Optimizer + 'static,
{
    fn random(&self) -> anyhow::Result<serde_json::Value> {
        let typed = Optimizer::random(self)?;
        let json = serde_json::to_value(&typed)?;
        Ok(json)
    }

    fn crossover(
        &self,
        parent1: serde_json::Value,
        parent2: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let typed_parent1 = serde_json::from_value::<T::Type>(parent1)?;
        let typed_parent2 = serde_json::from_value::<T::Type>(parent2)?;
        let typed_child = Optimizer::crossover(self, typed_parent1, typed_parent2)?;
        let json = serde_json::to_value(typed_child)?;
        Ok(json)
    }

    fn mutate(
        &self,
        instance: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let mut typed = serde_json::from_value::<T::Type>(instance)?;
        Optimizer::mutate(self, &mut typed)?;
        let json = serde_json::to_value(typed)?;
        Ok(json)
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("No optimizer registered found for type_name='{type_name}'")]
    NotFound { type_name: String },
}

pub struct OptimizerRegistry {
    optimizers: HashMap<&'static str, Arc<dyn OptimizerErased>>,
}

impl Default for OptimizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizerRegistry {
    pub(crate) fn new() -> Self {
        Self {
            optimizers: HashMap::new(),
        }
    }

    #[instrument(level = "info", skip(self, optimizer))]
    pub fn register<T>(&mut self, type_name: &'static str, optimizer: T)
    where
        T: Optimizer + 'static,
    {
        self.optimizers.insert(type_name, Arc::new(optimizer));
        tracing::info!("registered optimizer")
    }

    pub(crate) fn has_registered(&self, type_name: &str) -> bool {
        self.optimizers.contains_key(type_name)
    }

    pub(crate) fn get(&self, type_name: &str) -> anyhow::Result<&Arc<dyn OptimizerErased>> {
        let optimizer = self.optimizers.get(type_name).ok_or(Error::NotFound {
            type_name: type_name.to_string(),
        })?;

        Ok(optimizer)
    }

    pub(crate) fn get_registered_type_names(&self) -> Vec<String> {
        self.optimizers.keys().map(|s| s.to_string()).collect()
    }
}
