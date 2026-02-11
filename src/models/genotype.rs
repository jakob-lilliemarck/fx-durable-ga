use chrono::{DateTime, Utc};
use serde_json::{Map, Value};
use sqlx::prelude::FromRow;
use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::time::Duration;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

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
    pub(crate) parent_a: Option<Uuid>,
    pub(crate) parent_b: Option<Uuid>,
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
        parent_a: Option<&Uuid>,
        parent_b: Option<&Uuid>,
    ) -> Result<Self, Error> {
        let genome_value = serde_json::to_value(genome)?;
        let genome_hash = Self::compute_genome_hash(&genome_value)?;

        Ok(Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
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

    pub fn type_hash(&self) -> i32 {
        self.type_hash
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

    pub fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub fn generation_id(&self) -> i32 {
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

/// Represents a fitness evaluation result for a genotype.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Evaluation {
    pub(crate) genotype_id: Uuid,
    pub(crate) fitness: f64,
    pub(crate) started_at: Option<DateTime<Utc>>,
    pub(crate) completed_at: Option<DateTime<Utc>>,
    pub(crate) evaluated_by: Option<Uuid>,
    pub(crate) copied_from: Option<Uuid>,
}

impl Evaluation {
    /// Creates a new fitness record for a genotype.
    #[instrument(level = "debug", fields(
        genotype_id = %genotype_id,
        fitness = fitness,
        started_at = ?started_at,
        completed_at = ?completed_at,
        evaluated_by = ?evaluated_by
    ))]
    pub(crate) fn new(
        genotype_id: Uuid,
        fitness: f64,
        started_at: Option<DateTime<Utc>>,
        completed_at: Option<DateTime<Utc>>,
        evaluated_by: Option<Uuid>,
    ) -> Self {
        Self {
            genotype_id,
            fitness,
            started_at,
            completed_at,
            evaluated_by,
            copied_from: None,
        }
    }

    pub(crate) fn new_with_copied_from(
        genotype_id: Uuid,
        fitness: f64,
        started_at: DateTime<Utc>,
        completed_at: DateTime<Utc>,
        evaluated_by: Uuid,
        copied_from: Uuid,
    ) -> Self {
        Self {
            genotype_id,
            fitness,
            started_at: Some(started_at),
            completed_at: Some(completed_at),
            evaluated_by: Some(evaluated_by),
            copied_from: Some(copied_from),
        }
    }

    pub fn genotype_id(&self) -> &Uuid {
        &self.genotype_id
    }

    pub fn fitness(&self) -> f64 {
        self.fitness
    }

    pub fn started_at(&self) -> &Option<DateTime<Utc>> {
        &self.started_at
    }

    pub fn completed_at(&self) -> &Option<DateTime<Utc>> {
        &self.completed_at
    }

    pub fn evaluated_by(&self) -> &Option<Uuid> {
        &self.evaluated_by
    }

    pub fn copied_from(&self) -> &Option<Uuid> {
        &self.copied_from
    }
}

pub struct TimingsSummary {
    pub(crate) records: i64,
    pub(crate) min: Duration,
    pub(crate) max: Duration,
    pub(crate) avg: Duration,
    pub(crate) percentiles: Vec<Duration>,
}

impl TimingsSummary {
    pub fn records(&self) -> i64 {
        self.records
    }

    pub fn min(&self) -> &Duration {
        &self.min
    }

    pub fn max(&self) -> &Duration {
        &self.max
    }

    pub fn avg(&self) -> &Duration {
        &self.avg
    }

    pub fn percentiles(&self) -> &[Duration] {
        &self.percentiles
    }
}
