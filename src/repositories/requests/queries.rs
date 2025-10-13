use super::{Error, FitnessGoal, Request};
use chrono::{DateTime, Utc};
use sqlx::PgExecutor;
use uuid::Uuid;

/// A request for an optimization
#[derive(Debug)]
struct DbRequest {
    id: Uuid,
    requested_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
    goal: FitnessGoal,
    threshold: f64,
    strategy: serde_json::Value,
    temperature: f64,
    mutation_rate: f64,
}

impl TryFrom<Request> for DbRequest {
    type Error = Error;

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

pub(crate) async fn new_request<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request: Request,
) -> Result<Request, Error> {
    let db_request: DbRequest = request.try_into()?;
    let db_request = sqlx::query_as!(
        DbRequest,
        r#"
            INSERT INTO fx_durable_ga.requests (
                id,
                requested_at,
                type_name,
                type_hash,
                goal,
                threshold,
                strategy,
                temperature,
                mutation_rate
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING
                id,
                requested_at,
                type_name,
                type_hash,
                goal "goal:FitnessGoal",
                threshold,
                strategy,
                temperature,
                mutation_rate
            "#,
        db_request.id,
        db_request.requested_at,
        db_request.type_name,
        db_request.type_hash,
        db_request.goal as FitnessGoal,
        db_request.threshold,
        db_request.strategy,
        db_request.temperature,
        db_request.mutation_rate
    )
    .fetch_one(tx)
    .await?;

    let request: Request = db_request.try_into()?;
    Ok(request)
}

pub(crate) async fn get_request<'tx, E: PgExecutor<'tx>>(
    tx: E,
    id: Uuid,
) -> Result<Request, Error> {
    let db_request = sqlx::query_as!(
        DbRequest,
        r#"
        SELECT
            id,
            requested_at,
            type_name,
            type_hash,
            goal "goal!:FitnessGoal",
            threshold,
            strategy,
            temperature,
            mutation_rate
        FROM fx_durable_ga.requests
        WHERE id = $1
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    let request: Request = db_request.try_into()?;
    Ok(request)
}
