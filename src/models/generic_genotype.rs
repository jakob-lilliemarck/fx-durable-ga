use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::instrument;
use uuid::Uuid;

/// A struct that represents a genotype in the database, using type erasure.
/// The genome itself is stored as a `serde_json::Value`.
#[derive(Debug, Clone)]
pub struct GenericGenotype {
    pub(crate) id: Uuid,
    pub(crate) generated_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) genome_data: Value,
    pub(crate) genome_hash: i64,
    pub(crate) request_id: Uuid,
    pub(crate) generation_id: i32,
}

impl GenericGenotype {
    /// Creates a new generic genotype from any `Evolvable` type.
    #[instrument(level = "debug", skip(genome), fields(type_name = type_name, type_hash = type_hash))]
    pub fn new<G>(
        type_name: &str,
        type_hash: i32,
        genome: &G,
        request_id: Uuid,
        generation_id: i32,
    ) -> Result<Self, serde_json::Error>
    where
        G: Serialize + Hash,
    {
        let genome_data = serde_json::to_value(genome)?;

        let mut hasher = DefaultHasher::new();
        genome.hash(&mut hasher);
        let genome_hash = hasher.finish() as i64;

        Ok(Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            genome_data,
            genome_hash,
            request_id,
            generation_id,
        })
    }

    /// Deserializes the inner `genome_data` into a concrete type.
    pub fn deserialize<G: DeserializeOwned>(self) -> Result<G, serde_json::Error> {
        serde_json::from_value(self.genome_data)
    }
}
