use chrono::{DateTime, Utc};
use sqlx::prelude::FromRow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::instrument;
use uuid::Uuid;

pub type Gene = i64;

#[derive(Debug, Clone, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Genotype {
    pub(crate) id: Uuid,
    pub(crate) generated_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) genome: Vec<Gene>,
    pub(crate) genome_hash: i64,
    #[allow(dead_code)]
    pub(crate) request_id: Uuid,
    #[allow(dead_code)]
    pub(crate) generation_id: i32,
}

impl Genotype {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, genome_length = genome.len()))]
    pub(crate) fn new(
        type_name: &str,
        type_hash: i32,
        genome: Vec<Gene>,
        request_id: Uuid,
        generation_id: i32,
    ) -> Self {
        let genome_hash = Self::compute_genome_hash(&genome);

        Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            genome,
            genome_hash,
            request_id,
            generation_id,
        }
    }

    // Fast hash function for Vec<Gene>
    pub(crate) fn compute_genome_hash(genome: &[Gene]) -> i64 {
        let mut hasher = DefaultHasher::new();
        genome.hash(&mut hasher);
        hasher.finish() as i64
    }
}
