use super::Error;
use crate::models::{FitnessGoal, Request};
use chrono::{DateTime, Utc};
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug)]
pub(super) struct DbRequest {
    pub(super) id: Uuid,
    pub(super) requested_at: DateTime<Utc>,
    pub(super) type_name: String,
    pub(super) type_hash: i32,
    pub(super) goal: FitnessGoal,
    pub(super) threshold: f64,
    pub(super) strategy: serde_json::Value,
    pub(super) temperature: f64,
    pub(super) mutation_rate: f64,
}

impl TryFrom<Request> for DbRequest {
    type Error = Error;

    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash))]
    fn try_from(request: Request) -> Result<Self, Self::Error> {
        let strategy_json = serde_json::to_value(request.strategy)?;

        Ok(DbRequest {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal: request.goal,
            threshold: request.threshold,
            strategy: strategy_json,
            temperature: request.temperature,
            mutation_rate: request.mutation_rate,
        })
    }
}

impl TryFrom<DbRequest> for Request {
    type Error = Error;

    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash))]
    fn try_from(request: DbRequest) -> Result<Self, Self::Error> {
        let strategy_json = serde_json::from_value(request.strategy)?;

        Ok(Request {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal: request.goal,
            threshold: request.threshold,
            strategy: strategy_json,
            temperature: request.temperature,
            mutation_rate: request.mutation_rate,
        })
    }
}
