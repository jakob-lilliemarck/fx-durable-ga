use chrono::{DateTime, Utc};
use sqlx::prelude::FromRow;
use tracing::instrument;
use uuid::Uuid;

pub type Gene = i64;

#[derive(Debug, Clone, FromRow)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Genotype {
    pub(crate) id: Uuid,
    pub(crate) generated_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) genome: Vec<Gene>,
    pub(crate) request_id: Uuid,
    pub(crate) fitness: Option<f64>,
    pub(crate) generation_id: i32,
}

impl Genotype {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, genome_length = genome.len(), request_id = %request_id, generation_id = generation_id))]
    pub(crate) fn new(
        type_name: &str,
        type_hash: i32,
        genome: Vec<Gene>,
        request_id: Uuid,
        generation_id: i32,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            generated_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            genome,
            request_id,
            fitness: None,
            generation_id,
        }
    }

    /// Panics if the genotype does not have fitness
    #[instrument(level = "debug", fields(id = %self.id, has_fitness = self.fitness.is_some()))]
    pub(crate) fn must_fitness(&self) -> f64 {
        self.fitness.expect("Genotype must have a fitness value")
    }
}
