use super::{FitnessGoal, Strategy};
use crate::models::{Crossover, mutagen::Mutagen};
use chrono::{DateTime, Utc};
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub(crate) struct Request {
    pub(crate) id: Uuid,
    pub(crate) requested_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) goal: FitnessGoal,
    pub(crate) strategy: Strategy,
    pub(crate) mutagen: Mutagen,
    pub(crate) crossover: Crossover,
}

#[derive(Debug, thiserror::Error)]
pub enum RequestValidationError {}

impl Request {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, goal = ?goal, mutagen = ?mutagen))]
    pub(crate) fn new(
        type_name: &str,
        type_hash: i32,
        goal: FitnessGoal,
        strategy: Strategy,
        mutagen: Mutagen,
        crossover: Crossover,
    ) -> Result<Self, RequestValidationError> {
        Ok(Self {
            id: Uuid::now_v7(),
            requested_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            goal,
            strategy,
            mutagen,
            crossover,
        })
    }

    #[instrument(level = "debug", fields(request_id = %self.id))]
    pub(crate) fn population_size(&self) -> u32 {
        match self.strategy {
            Strategy::Generational {
                population_size, ..
            } => population_size,
            Strategy::Rolling {
                population_size, ..
            } => population_size,
        }
    }

    #[instrument(level = "debug", fields(request_id = %self.id, fitness = fitness, goal = ?self.goal))]
    pub(crate) fn is_completed(&self, fitness: f64) -> bool {
        self.goal.is_reached(fitness)
    }
}
