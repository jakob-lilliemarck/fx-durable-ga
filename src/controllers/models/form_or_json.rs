use crate::bootstrap::App;
use axum::{
    Form, Json,
    extract::{FromRequest, Request},
    response::{IntoResponse, Response},
};
use reqwest::header::CONTENT_TYPE;
use serde::de::DeserializeOwned;
use std::sync::Arc;

pub(crate) enum FormOrJson<T> {
    Form(T),
    Json(T),
}

impl<T> FormOrJson<T> {
    pub(crate) fn into_inner(self) -> T {
        match self {
            Self::Form(v) => v,
            Self::Json(v) => v,
        }
    }
}

impl<T: schemars::JsonSchema> aide::operation::OperationInput for FormOrJson<T> {
    fn operation_input(ctx: &mut aide::generate::GenContext, op: &mut aide::openapi::Operation) {
        // delegate to Json since that's the API contract
        axum::Json::<T>::operation_input(ctx, op);
    }
}

impl<T: DeserializeOwned> FromRequest<Arc<App>> for FormOrJson<T> {
    type Rejection = Response;

    async fn from_request(req: Request, state: &Arc<App>) -> Result<Self, Self::Rejection> {
        let content_type = req
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if content_type.contains("application/json") {
            let Json(value) = Json::<T>::from_request(req, state)
                .await
                .map_err(|e| e.into_response())?;
            Ok(Self::Json(value))
        } else {
            let Form(value) = Form::<T>::from_request(req, state)
                .await
                .map_err(|e| e.into_response())?;
            Ok(Self::Form(value))
        }
    }
}
