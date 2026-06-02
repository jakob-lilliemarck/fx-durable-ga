use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::{Map, Value};
use sqlx::prelude::FromRow;
use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use tracing::instrument;
use uuid::Uuid;

pub trait Identifiable {
    fn id(&self) -> Uuid;
}

pub trait TypeName {
    fn type_name(&self) -> &str;
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Represents an individual genotype in the genetic algorithm population.
#[derive(Debug, Clone, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Genotype {
    pub(crate) id: Uuid,
    pub(crate) generated_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) genome: Value,
    pub(crate) genome_hash: i64,
    #[allow(dead_code)]
    pub(crate) request_id: Option<Uuid>,
    #[allow(dead_code)]
    pub(crate) generation_id: Option<i32>,
    pub(crate) parent_a: Option<Uuid>,
    pub(crate) parent_b: Option<Uuid>,
}

impl Identifiable for Genotype {
    fn id(&self) -> Uuid {
        self.id
    }
}

impl TypeName for Genotype {
    fn type_name(&self) -> &str {
        &self.type_name
    }
}

impl Genotype {
    /// Creates a new genotype with the given genome and metadata.
    #[instrument(level = "debug", fields(type_name = type_name))]
    pub(crate) fn new<G: serde::Serialize + std::fmt::Debug>(
        type_name: &str,
        genome: G,
        request_id: Option<Uuid>,
        generation_id: Option<i32>,
        parent_a: Option<&Uuid>,
        parent_b: Option<&Uuid>,
    ) -> Result<Self, Error> {
        let genome_value = serde_json::to_value(genome)?;
        let genome_hash = Self::compute_genome_hash(&genome_value)?;

        Ok(Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            genome: genome_value,
            genome_hash,
            request_id,
            generation_id,
            parent_a: parent_a.map(Clone::clone),
            parent_b: parent_b.map(Clone::clone),
        })
    }

    /// Computes a deterministic hash of the genome for deduplication and comparison.
    pub(crate) fn compute_genome_hash<G: serde::Serialize>(genome: &G) -> Result<i64, Error> {
        let genome_value = serde_json::to_value(genome)?;
        let canonical = canonicalize_json(&genome_value);
        let mut hasher = DefaultHasher::new();
        canonical.to_string().hash(&mut hasher);
        Ok(hasher.finish() as i64)
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn genome(&self) -> Value {
        self.genome.clone()
    }

    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    pub fn generated_at(&self) -> DateTime<Utc> {
        self.generated_at
    }

    pub fn genome_hash(&self) -> i64 {
        self.genome_hash
    }

    pub fn request_id(&self) -> Option<Uuid> {
        self.request_id
    }

    pub fn generation_id(&self) -> Option<i32> {
        self.generation_id
    }

    pub fn parent_a(&self) -> &Option<Uuid> {
        &self.parent_a
    }

    pub fn parent_b(&self) -> &Option<Uuid> {
        &self.parent_b
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

/// Genotype-only population stats (no evaluation data).
#[derive(Debug, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub struct GenotypePopulation {
    pub(crate) request_id: Uuid,
    pub(crate) total_genotypes: i64,
    pub(crate) current_generation: i32,
}

impl GenotypePopulation {
    pub fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub fn total_genotypes(&self) -> i64 {
        self.total_genotypes
    }

    pub fn current_generation(&self) -> i32 {
        self.current_generation
    }
}

/// Represents the current state of a genetic algorithm population.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Population {
    #[allow(dead_code)]
    pub(crate) request_id: Uuid,
    pub(crate) evaluated_genotypes: i64,
    pub(crate) live_genotypes: i64,
    pub(crate) current_generation: i32,
    pub(crate) min_fitness: Option<f64>,
    pub(crate) max_fitness: Option<f64>,
}

impl Population {
    pub fn new(
        request_id: Uuid,
        total_genotypes: i64,
        evaluated_genotypes: i64,
        current_generation: i32,
        min_fitness: Option<f64>,
        max_fitness: Option<f64>,
    ) -> Self {
        Self {
            request_id,
            evaluated_genotypes,
            live_genotypes: total_genotypes - evaluated_genotypes,
            current_generation,
            min_fitness,
            max_fitness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_hash_consistency_object_order() -> anyhow::Result<()> {
        let genome1 = serde_json::json!({"a": 1, "b": 2});
        let genome2 = serde_json::json!({"b": 2, "a": 1});

        let hash1 = Genotype::compute_genome_hash(&genome1)?;
        let hash2 = Genotype::compute_genome_hash(&genome2)?;

        assert_eq!(hash1, hash2);

        Ok(())
    }

    #[test]
    fn test_genome_hash_changes_with_values() -> anyhow::Result<()> {
        let genome1 = serde_json::json!({"a": 1});
        let genome2 = serde_json::json!({"a": 2});

        let hash1 = Genotype::compute_genome_hash(&genome1)?;
        let hash2 = Genotype::compute_genome_hash(&genome2)?;

        assert_ne!(hash1, hash2);

        Ok(())
    }
}
