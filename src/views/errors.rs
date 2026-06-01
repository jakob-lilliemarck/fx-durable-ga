use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
pub struct ErrorResponse {
    error: String,
}

impl ErrorResponse {
    pub fn new(message: String) -> Self {
        Self { error: message }
    }
}
