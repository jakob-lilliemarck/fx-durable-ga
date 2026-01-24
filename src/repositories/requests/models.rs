use super::Error;
use crate::models::Request;
use chrono::{DateTime, Utc};
use tracing::instrument;
use uuid::Uuid;

/// Database representation of a request with JSON-serialized configuration fields.
#[derive(Debug)]
#[cfg_attr(test, derive(Clone))]
pub(super) struct DbRequest {
    pub(super) id: Uuid,
    pub(super) requested_at: DateTime<Utc>,
    pub(super) type_name: String,
    pub(super) type_hash: i32,
    pub(super) goal: serde_json::Value,
    pub(super) schedule: serde_json::Value,
    pub(super) selector: serde_json::Value,
    pub(super) user_defined: serde_json::Value,
    pub(super) data: Option<serde_json::Value>,
}

impl TryFrom<Request> for DbRequest {
    type Error = Error;

    /// Converts domain request to database model by serializing configuration to JSON.
    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash))]
    fn try_from(request: Request) -> Result<Self, Self::Error> {
        let schedule_json = serde_json::to_value(request.schedule)?;
        let selector_json = serde_json::to_value(request.selector)?;
        let user_defined_json = request.user_defined;
        let goal_json = serde_json::to_value(request.goal)?;

        Ok(DbRequest {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal: goal_json,
            schedule: schedule_json,
            selector: selector_json,
            user_defined: user_defined_json,
            data: request.data,
        })
    }
}

impl TryFrom<DbRequest> for Request {
    type Error = Error;

    /// Converts database request to domain model by deserializing configuration from JSON.
    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash))]
    fn try_from(request: DbRequest) -> Result<Self, Self::Error> {
        let schedule = serde_json::from_value(request.schedule)?;
        let selector = serde_json::from_value(request.selector)?;
        let user_defined = request.user_defined;
        let goal = serde_json::from_value(request.goal)?;

        Ok(Request {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal,
            schedule,
            selector,
            user_defined,
            data: request.data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FitnessGoal, Schedule, Selector};
    use serde_json::json;

    fn create_test_request() -> Request {
        Request::new(
            "TestType",
            123,
            FitnessGoal::minimize(0.9).unwrap(),
            Selector::tournament(5, 20).expect("is valid"),
            Schedule::generational(100, 10),
            json!({"Uniform":{"probability":0.5}}),
            None::<()>,
        )
        .unwrap()
    }

    #[test]
    fn test_request_to_db_request_conversion() {
        let request = create_test_request();
        let db_request = DbRequest::try_from(request.clone()).unwrap();

        assert_eq!(db_request.id, request.id);
        assert_eq!(db_request.requested_at, request.requested_at);
        assert_eq!(db_request.type_name, request.type_name);
        assert_eq!(db_request.type_hash, request.type_hash);

        // Verify JSON serialization matches expected format
        assert_eq!(db_request.goal, json!({"Minimize": {"threshold": 0.9}}));
        assert_eq!(
            db_request.schedule,
            json!({"max_evaluations": 1000, "population_size": 10, "selection_interval": 10})
        );
        assert_eq!(
            db_request.selector,
            json!({"method": {"Tournament": {"size": 5}}, "sample_size": 20})
        );
        assert_eq!(
            db_request.user_defined,
            json!({"Uniform":{"probability":0.5}})
        );
    }

    #[test]
    fn test_db_request_to_request_conversion() {
        // Create a real request and serialize it to get valid JSON
        let original_request = create_test_request();
        let db_request = DbRequest::try_from(original_request).unwrap();

        // Now convert back
        let request = Request::try_from(db_request).unwrap();

        assert_eq!(request.type_name, "TestType");
        assert_eq!(request.type_hash, 123);
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
