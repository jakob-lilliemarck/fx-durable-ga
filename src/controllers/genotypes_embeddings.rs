use crate::{SearchGenotypesFilter, bootstrap::App};
use aide::axum::{ApiRouter, routing::post_with};
use axum::{Json, http::StatusCode};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct BackfillGenotypeEmbeddingsRequest {
    pub request_id: Option<Uuid>,
    pub generation_ids: Option<Vec<String>>,
    pub genotype_ids: Option<Vec<Uuid>>,
    pub has_evaluation: Option<bool>,
}

#[derive(Serialize, JsonSchema)]
struct ErrorResponse {
    error: String,
}

pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .api_route(
            "/genotypes/embeddings",
            post_with(backfill_genotype_embeddings, |op| {
                op.summary("Backfill missing genotype embeddings")
                    .response::<202, ()>()
                    .response::<400, Json<ErrorResponse>>()
                    .response::<500, Json<ErrorResponse>>()
            }),
        )
        .with_state(app)
}

async fn backfill_genotype_embeddings(
    axum::extract::State(app): axum::extract::State<Arc<App>>,
    Json(payload): Json<BackfillGenotypeEmbeddingsRequest>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let mut filter = SearchGenotypesFilter::default();

    if let Some(request_id) = payload.request_id {
        filter = filter.with_request_id(request_id);
    }

    if let Some(generation_ids) = payload.generation_ids {
        for generation_id in generation_ids {
            let parsed = generation_id.parse::<i32>().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("invalid generation_id: {generation_id}"),
                    }),
                )
            })?;
            filter = filter.with_generation_id(parsed);
        }
    }

    if let Some(genotype_ids) = payload.genotype_ids {
        for genotype_id in genotype_ids {
            filter = filter.with_genotype_id(genotype_id);
        }
    }

    app.services()
        .genotype_indexing()
        .backfill_missing_genotypes(filter)
        .await
        .map_err(|err| {
            error!(error = %err, "failed to backfill missing genotypes");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            )
        })?;

    Ok(StatusCode::ACCEPTED)
}
