use super::Error;
use crate::models::Request;
use chrono::{DateTime, Utc};
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug)]
pub(super) struct DbRequest {
    pub(super) id: Uuid,
    pub(super) requested_at: DateTime<Utc>,
    pub(super) type_name: String,
    pub(super) type_hash: i32,
    pub(super) goal: serde_json::Value,
    pub(super) strategy: serde_json::Value,
    pub(super) mutagen: serde_json::Value,
    pub(super) crossover: serde_json::Value,
}

impl TryFrom<Request> for DbRequest {
    type Error = Error;

    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash))]
    fn try_from(request: Request) -> Result<Self, Self::Error> {
        let strategy_json = serde_json::to_value(request.strategy)?;
        let mutagen_json = serde_json::to_value(request.mutagen)?;
        let crossover_json = serde_json::to_value(request.crossover)?;
        let goal_json = serde_json::to_value(request.goal)?;

        Ok(DbRequest {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal: goal_json,
            strategy: strategy_json,
            mutagen: mutagen_json,
            crossover: crossover_json,
        })
    }
}

impl TryFrom<DbRequest> for Request {
    type Error = Error;

    #[instrument(level = "debug", fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash))]
    fn try_from(request: DbRequest) -> Result<Self, Self::Error> {
        let strategy = serde_json::from_value(request.strategy)?;
        let mutagen = serde_json::from_value(request.mutagen)?;
        let crossover = serde_json::from_value(request.crossover)?;
        let goal = serde_json::from_value(request.goal)?;

        Ok(Request {
            id: request.id,
            requested_at: request.requested_at,
            type_name: request.type_name,
            type_hash: request.type_hash,
            goal,
            strategy,
            mutagen,
            crossover,
        })
    }
}
