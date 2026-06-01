use crate::bootstrap::App;
use aide::axum::{ApiRouter, routing::post_with};
use axum::{Json, extract::State, http::StatusCode};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

#[derive(Serialize, JsonSchema)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct CreateRequestPayload {
    pub type_name: String,
    pub goal: serde_json::Value,
    pub schedule: serde_json::Value,
    pub selector: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CreateRequestResponse {
    pub request_id: Uuid,
}

pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .api_route(
            "/requests",
            post_with(create_request, |op| {
                op.summary("Create a new optimization request")
                    .response::<201, Json<CreateRequestResponse>>()
                    .response::<400, Json<ErrorResponse>>()
                    .response::<500, Json<ErrorResponse>>()
            }),
        )
        .with_state(app)
}

async fn create_request(
    State(app): State<Arc<App>>,
    Json(payload): Json<CreateRequestPayload>,
) -> Result<(StatusCode, Json<CreateRequestResponse>), (StatusCode, Json<ErrorResponse>)> {
    let goal = serde_json::from_value(payload.goal).map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid goal: {err}"),
            }),
        )
    })?;
    let schedule = serde_json::from_value(payload.schedule).map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid schedule: {err}"),
            }),
        )
    })?;
    let selector = serde_json::from_value(payload.selector).map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid selector: {err}"),
            }),
        )
    })?;

    let request_id = app
        .services()
        .optimization()
        .request_new(
            payload.type_name,
            goal,
            schedule,
            selector,
        )
        .await
        .map_err(|err| {
            error!(error = %err, "failed to create request");
            let status = match err {
                crate::services::optimization::Error::UnknownTypeError { .. } => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            )
        })?;

    Ok((
        StatusCode::CREATED,
        Json(CreateRequestResponse { request_id }),
    ))
}
