use super::{Crossover, Distribution, FitnessGoal, Mutagen, Schedule, Selector};
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
    pub(crate) selector: Selector,
    pub(crate) schedule: Schedule,
    pub(crate) mutagen: Mutagen,
    pub(crate) crossover: Crossover,
    pub(crate) distribution: Distribution,
}

#[derive(Debug, thiserror::Error)]
pub enum RequestValidationError {}

impl Request {
    #[instrument(level = "debug", fields(type_name = type_name, type_hash = type_hash, goal = ?goal, mutagen = ?mutagen))]
    pub(crate) fn new(
        type_name: &str,
        type_hash: i32,
        goal: FitnessGoal,
        selector: Selector,
        schedule: Schedule,
        mutagen: Mutagen,
        crossover: Crossover,
        distribution: Distribution,
    ) -> Result<Self, RequestValidationError> {
        Ok(Self {
            id: Uuid::now_v7(),
            requested_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            goal,
            selector,
            schedule,
            mutagen,
            crossover,
            distribution,
        })
    }

    #[instrument(level = "debug", fields(request_id = %self.id, fitness = fitness, goal = ?self.goal))]
    pub(crate) fn is_completed(&self, fitness: f64) -> bool {
        self.goal.is_reached(fitness)
    }
}

#[derive(Debug, sqlx::Type, Clone, Copy, PartialEq, Eq)]
#[sqlx(type_name = "fx_durable_ga.conclusion", rename_all = "lowercase")]
pub(crate) enum Conclusion {
    Completed,
    Terminated,
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub(crate) struct RequestConclusion {
    pub(crate) request_id: Uuid,
    pub(crate) concluded_at: DateTime<Utc>,
    pub(crate) concluded_with: Conclusion,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        Crossover, Distribution, FitnessGoal, Mutagen, MutationRate, Schedule, Selector,
        Temperature,
    };

    fn create_test_request(goal: FitnessGoal) -> Request {
        Request::new(
            "TestType",
            123,
            goal,
            Selector::tournament(5, 20),
            Schedule::generational(100, 10),
            Mutagen::new(
                Temperature::constant(0.5).unwrap(),
                MutationRate::constant(0.1).unwrap(),
            ),
            Crossover::uniform(0.5).unwrap(),
            Distribution::latin_hypercube(50),
        )
        .unwrap()
    }

    #[test]
    fn test_request_creation() {
        let goal = FitnessGoal::minimize(0.9);

        assert!(goal.is_ok());
        let goal = goal.unwrap();
        let request = create_test_request(goal);

        // Verify the request was created with correct fields
        assert_eq!(request.type_name, "TestType");
        assert_eq!(request.type_hash, 123);

        // Verify UUID and timestamp were set (basic existence checks)
        assert!(!request.id.is_nil());
        assert!(request.requested_at <= Utc::now());
    }

    #[test]
    fn test_is_completed_minimize() {
        let goal = FitnessGoal::minimize(0.9);

        assert!(goal.is_ok());
        let goal = goal.unwrap();
        let request = create_test_request(goal);

        assert!(!request.is_completed(0.95)); // Above target - not completed
        assert!(!request.is_completed(0.91)); // Slightly above - not completed
        assert!(request.is_completed(0.9)); // At target - completed
        assert!(request.is_completed(0.5)); // Below target - completed
    }

    #[test]
    fn test_is_completed_maximize() {
        let goal = FitnessGoal::maximize(0.9);

        assert!(goal.is_ok());
        let goal = goal.unwrap();
        let request = create_test_request(goal);

        assert!(!request.is_completed(0.5)); // Below target - not completed
        assert!(!request.is_completed(0.89)); // Slightly below - not completed
        assert!(request.is_completed(0.9)); // At target - completed
        assert!(request.is_completed(0.95)); // Above target - completed
    }
}
