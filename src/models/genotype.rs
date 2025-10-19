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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_hash_consistency() {
        let genome1 = vec![1, 2, 3, 4, 5];
        let genome2 = vec![1, 2, 3, 4, 5];
        let genome3 = vec![5, 4, 3, 2, 1];

        // Same genome should produce same hash
        let hash1 = Genotype::compute_genome_hash(&genome1);
        let hash2 = Genotype::compute_genome_hash(&genome2);

        // Same genome should produce same hash
        assert_eq!(hash1, hash2);

        let hash3 = Genotype::compute_genome_hash(&genome3);

        // Different genome should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_empty_genome_hash() {
        let empty_genome: Vec<Gene> = vec![];
        let hash1 = Genotype::compute_genome_hash(&empty_genome);
        let hash2 = Genotype::compute_genome_hash(&empty_genome);

        //Empty genome should produce consistent hash
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_zero_genome_hash() {
        let zero_genome1 = vec![0, 0, 0];
        let zero_genome2 = vec![0, 0, 0];

        let hash1 = Genotype::compute_genome_hash(&zero_genome1);
        let hash2 = Genotype::compute_genome_hash(&zero_genome2);

        // Identical zero genomes should produce same hash
        assert_eq!(hash1, hash2,);
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Fitness {
    pub(crate) genotype_id: Uuid,
    pub(crate) fitness: f64,
    pub(crate) evaluated_at: DateTime<Utc>,
}

impl Fitness {
    pub(crate) fn new(genotype_id: Uuid, fitness: f64) -> Self {
        Self {
            genotype_id,
            fitness,
            evaluated_at: Utc::now(),
        }
    }
}
