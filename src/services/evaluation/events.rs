use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenotypeEvaluatedEvent {
    pub evaluation_id: Uuid,
    pub group_id: Uuid,
    pub reason: String,
    pub genotype_id: Uuid,
    pub fitness: f64,
}

impl fx_event_bus::Event for GenotypeEvaluatedEvent {
    const NAME: &'static str = "GenotypeEvaluated";
}

impl GenotypeEvaluatedEvent {
    pub fn new(
        evaluation_id: Uuid,
        group_id: Uuid,
        reason: String,
        genotype_id: Uuid,
        fitness: f64,
    ) -> Self {
        Self {
            evaluation_id,
            group_id,
            reason,
            genotype_id,
            fitness,
        }
    }
}
