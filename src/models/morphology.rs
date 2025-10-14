use crate::models::{Gene, GeneBounds};
use chrono::{DateTime, Utc};
use tracing::instrument;

#[derive(Debug)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Morphology {
    pub(crate) revised_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) gene_bounds: Vec<GeneBounds>,
}

impl Morphology {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, gene_bounds_count = gene_bounds.len()))]
    pub fn new(type_name: &str, type_hash: i32, gene_bounds: Vec<GeneBounds>) -> Self {
        Self {
            revised_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            gene_bounds,
        }
    }

    #[instrument(level = "debug", fields(type_name = %self.type_name, type_hash = self.type_hash, gene_bounds_count = self.gene_bounds.len()))]
    pub fn random(&self) -> Vec<Gene> {
        let mut rng = rand::rng();

        self.gene_bounds
            .iter()
            .map(|gene_bound| gene_bound.random(&mut rng))
            .collect()
    }
}
