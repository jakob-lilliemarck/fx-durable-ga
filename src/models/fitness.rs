use chrono::{DateTime, Utc};
use uuid::Uuid;

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
