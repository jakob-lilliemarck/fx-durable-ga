use chrono::{DateTime, Utc};
use serde_json::{Map, Value};
use sqlx::prelude::FromRow;
use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use tracing::instrument;
use uuid::Uuid;

/// Represents an individual genotype in the genetic algorithm population.
/// Contains the genome data and metadata for tracking through generations.
#[derive(Debug, Clone, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Genotype {
    pub(crate) id: Uuid,
    pub(crate) generated_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) genome: Value,
    pub(crate) genome_hash: i64,
    #[allow(dead_code)]
    pub(crate) request_id: Uuid,
    #[allow(dead_code)]
    pub(crate) generation_id: i32,
}

impl Genotype {
    /// Creates a new genotype with the given genome and metadata.
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash))]
    pub(crate) fn new<G: serde::Serialize + std::fmt::Debug>(
        type_name: &str,
        type_hash: i32,
        genome: G,
        request_id: Uuid,
        generation_id: i32,
    ) -> Self {
        // FIXME: add a named error or a better logging here. We know its "serializable" because the compiler guarantees it, but there could still be an error during serialization.
        let genome_value = serde_json::to_value(genome).expect("genome must be serializable");
        let genome_hash = Self::compute_genome_hash(&genome_value);

        Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            genome: genome_value,
            genome_hash,
            request_id,
            generation_id,
        }
    }

    /// Computes a deterministic hash of the genome for deduplication and comparison.
    pub(crate) fn compute_genome_hash<G: serde::Serialize>(genome: &G) -> i64 {
        // FIXME: add a named error or a better logging here. We know its "serializable" because the compiler guarantees it, but there could still be an error during serialization.
        let genome_value = serde_json::to_value(genome).expect("genome must be serializable");
        let canonical = canonicalize_json(&genome_value);
        let mut hasher = DefaultHasher::new();
        canonical.to_string().hash(&mut hasher);
        hasher.finish() as i64
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn genome(&self) -> Value {
        self.genome.clone()
    }

    pub(crate) fn type_hash(&self) -> i32 {
        self.type_hash
    }

    pub(crate) fn type_name(&self) -> &str {
        &self.type_name
    }

    pub(crate) fn generated_at(&self) -> DateTime<Utc> {
        self.generated_at
    }

    pub(crate) fn genome_hash(&self) -> i64 {
        self.genome_hash
    }

    pub(crate) fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub(crate) fn generation_id(&self) -> i32 {
        self.generation_id
    }

}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut ordered = BTreeMap::new();
            for (k, v) in map {
                ordered.insert(k.clone(), canonicalize_json(v));
            }
            let mut new_map = Map::new();
            for (k, v) in ordered {
                new_map.insert(k, v);
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_json).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_hash_consistency_object_order() {
        let genome1 = serde_json::json!({"a": 1, "b": 2});
        let genome2 = serde_json::json!({"b": 2, "a": 1});

        let hash1 = Genotype::compute_genome_hash(&genome1);
        let hash2 = Genotype::compute_genome_hash(&genome2);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_genome_hash_changes_with_values() {
        let genome1 = serde_json::json!({"a": 1});
        let genome2 = serde_json::json!({"a": 2});

        let hash1 = Genotype::compute_genome_hash(&genome1);
        let hash2 = Genotype::compute_genome_hash(&genome2);

        assert_ne!(hash1, hash2);
    }
}

/// Represents a fitness evaluation result for a genotype.
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Fitness {
    pub(crate) genotype_id: Uuid,
    pub(crate) fitness: f64,
    pub(crate) evaluated_at: DateTime<Utc>,
}

impl Fitness {
    /// Creates a new fitness record for a genotype.
    #[instrument(level = "debug", fields(genotype_id = %genotype_id, fitness = fitness))]
    pub(crate) fn new(genotype_id: Uuid, fitness: f64) -> Self {
        Self {
            genotype_id,
            fitness,
            evaluated_at: Utc::now(),
        }
    }
}
