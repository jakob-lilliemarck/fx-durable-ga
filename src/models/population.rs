use uuid::Uuid;

/// Represents the current state of a genetic algorithm population.
/// Tracks progress and statistics for an optimization request.
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Population {
    #[allow(dead_code)]
    pub(crate) request_id: Uuid,
    pub(crate) evaluated_genotypes: i64,
    pub(crate) live_genotypes: i64,
    pub(crate) current_generation: i32,
    pub(crate) best_fitness: Option<f64>,
}
