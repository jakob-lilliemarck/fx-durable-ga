use super::super::Error;
use super::{FitnessGoal, Schedule, Selector};
use chrono::{DateTime, Utc};
use tracing::instrument;
use uuid::Uuid;

/// Represents an optimization request with all genetic algorithm parameters.
#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub struct Request {
    pub(crate) id: Uuid,
    pub(crate) requested_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) goal: FitnessGoal,
    pub(crate) selector: Selector,
    pub(crate) schedule: Schedule,
    pub(crate) account_id: Uuid,
}

impl Request {
    /// Creates a new optimization request with the given parameters.
    #[instrument(level = "debug", fields(type_name = type_name, goal = ?goal))]
    pub(crate) fn new(
        type_name: &str,
        goal: FitnessGoal,
        selector: Selector,
        schedule: Schedule,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            requested_at: Utc::now(),
            type_name: type_name.to_string(),
            goal,
            selector,
            schedule,
            account_id: Uuid::now_v7(),
        }
    }

    /// Checks if the optimization request is completed based on the given fitness value.
    #[instrument(level = "debug", fields(request_id = %self.id, fitness = fitness, goal = ?self.goal))]
    pub(crate) fn is_completed(&self, fitness: f64) -> bool {
        self.goal.is_reached(fitness)
    }
}

/// Database representation of a request with JSON-serialized configuration fields.
#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub struct DbRequest {
    pub id: Uuid,
    pub requested_at: DateTime<Utc>,
    pub type_name: String,
    pub goal: serde_json::Value,
    pub schedule: serde_json::Value,
    pub selector: serde_json::Value,
    pub account_id: Uuid,
}

impl TryFrom<Request> for DbRequest {
    type Error = Error;

    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name))]
    fn try_from(request: Request) -> Result<Self, Self::Error> {
        let schedule_json = serde_json::to_value(request.schedule)?;
        let selector_json = serde_json::to_value(request.selector)?;
        let goal_json = serde_json::to_value(request.goal)?;

        Ok(DbRequest {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            goal: goal_json,
            schedule: schedule_json,
            selector: selector_json,
            account_id: request.account_id,
        })
    }
}

impl TryFrom<DbRequest> for Request {
    type Error = Error;

    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name))]
    fn try_from(request: DbRequest) -> Result<Self, Self::Error> {
        let schedule = serde_json::from_value(request.schedule)?;
        let selector = serde_json::from_value(request.selector)?;
        let goal = serde_json::from_value(request.goal)?;

        Ok(Request {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            goal,
            schedule,
            selector,
            account_id: request.account_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_request() -> Request {
        Request::new(
            "TestType",
            FitnessGoal::minimize(0.9).unwrap(),
            Selector::tournament(5),
            Schedule::generational(100, 10),
        )
    }

    #[test]
    fn test_request_creation() {
        let goal = FitnessGoal::minimize(0.9).unwrap();
        let request = Request::new(
            "TestType",
            goal,
            Selector::tournament(5),
            Schedule::generational(100, 10),
        );

        assert_eq!(request.type_name, "TestType");

        assert!(!request.id.is_nil());
        assert!(request.requested_at <= Utc::now());
    }

    #[test]
    fn test_request_to_db_request_conversion() {
        let request = create_test_request();
        let db_request = DbRequest::try_from(request.clone()).unwrap();

        assert_eq!(db_request.id, request.id);
        assert_eq!(db_request.requested_at, request.requested_at);
        assert_eq!(db_request.type_name, request.type_name);

        assert_eq!(db_request.goal, json!({"Minimize": {"threshold": 0.9}}));
        assert_eq!(
            db_request.schedule,
            json!({"max_evaluations": 1000, "population_size": 10, "selection_interval": 10})
        );
        assert_eq!(
            db_request.selector,
            json!({"method": {"Tournament": {"size": 5}}})
        );
    }

    #[test]
    fn test_db_request_to_request_conversion() {
        let original_request = create_test_request();
        let db_request = DbRequest::try_from(original_request).unwrap();

        let request = Request::try_from(db_request).unwrap();

        assert_eq!(request.type_name, "TestType");
    }

    #[test]
    fn test_invalid_goal_json_fails() {
        let original_request = create_test_request();
        let mut db_request = DbRequest::try_from(original_request).unwrap();

        db_request.goal = json!({"invalid": "goal"});
        let result = Request::try_from(db_request);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Serde(_)));
    }

    #[test]
    fn test_invalid_schedule_json_fails() {
        let original_request = create_test_request();
        let mut db_request = DbRequest::try_from(original_request).unwrap();

        db_request.schedule = json!({"invalid": "schedule"});
        let result = Request::try_from(db_request);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Serde(_)));
    }

    #[test]
    fn test_invalid_selector_json_fails() {
        let original_request = create_test_request();
        let mut db_request = DbRequest::try_from(original_request).unwrap();

        db_request.selector = json!({"invalid": "selector"});
        let result = Request::try_from(db_request);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::Serde(_)));
    }
}
