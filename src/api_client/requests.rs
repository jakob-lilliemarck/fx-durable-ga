//! Request endpoint methods for the API client

use super::Client;
use crate::controllers::{CreateRequestPayload, CreateRequestResponse};
use crate::services::optimization::{FitnessGoal, Schedule, Selector};
use anyhow::{Context, Result};
use uuid::Uuid;

impl Client {
    /// POST /requests - Create a new optimization request
    ///
    /// # Arguments
    ///
    /// * `type_name` - The optimization type name registered on the server
    /// * `goal` - Fitness goal for the optimization
    /// * `schedule` - Breeding schedule configuration
    /// * `selector` - Parent selection strategy
    /// * `user_defined` - User-defined configuration as JSON
    /// * `data` - Optional additional data payload
    ///
    /// # Returns
    ///
    /// Returns the created request ID on success
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The server returns a non-201 status code
    /// - The response cannot be parsed
    pub async fn requests_new(
        &self,
        type_name: String,
        goal: FitnessGoal,
        schedule: Schedule,
        selector: Selector,
        user_defined: Option<serde_json::Value>,
        data: Option<serde_json::Value>,
    ) -> Result<CreateRequestResponse> {
        let payload = CreateRequestPayload {
            type_name,
            goal: serde_json::to_value(goal).context("Failed to serialize goal")?,
            schedule: serde_json::to_value(schedule).context("Failed to serialize schedule")?,
            selector: serde_json::to_value(selector).context("Failed to serialize selector")?,
            user_defined,
            data,
        };

        let url = format!("{}/requests", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .post(&url)
            .json(&payload)
            .send()
            .await
            .context("Failed to send request")?;

        if response.status() != reqwest::StatusCode::CREATED {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Request failed ({}): {}", status, body);
        }

        response.json().await.context("Failed to parse response")
    }

    /// POST /requests/{request_id}/interrupt - Interrupt a running request
    ///
    /// # Arguments
    ///
    /// * `request_id` - The UUID of the request to interrupt
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The server returns a non-200 status code
    pub async fn requests_interrupt(&self, request_id: Uuid) -> Result<()> {
        let url = format!(
            "{}/requests/{}/interrupt",
            self.base_url.trim_end_matches('/'),
            request_id
        );

        let response = self
            .http
            .post(&url)
            .send()
            .await
            .context("Failed to send request")?;

        if response.status() != reqwest::StatusCode::OK {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Request failed ({}): {}", status, body);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::optimization::{FitnessGoal, Schedule, Selector};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, header, method, path},
    };

    #[tokio::test]
    async fn test_requests_new_sends_correct_request() {
        // Arrange: Start mock server
        let mock_server = MockServer::start().await;

        let expected_request_id = uuid::Uuid::new_v4();

        // Set up mock expectation
        Mock::given(method("POST"))
            .and(path("/requests"))
            .and(header(
                "user-agent",
                format!("ga/{}", env!("CARGO_PKG_VERSION")).as_str(),
            ))
            .and(header("content-type", "application/json"))
            .and(body_json(serde_json::json!({
                "type_name": "test_optimization",
                "goal": {"Minimize": {"threshold": 0.05}},
                "schedule": {
                    "max_evaluations": 100,
                    "population_size": 20,
                    "selection_interval": 20
                },
                "selector": {"method": "Roulette"},
                "user_defined": {"test": "value"},
                "data": null
            })))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "request_id": expected_request_id
            })))
            .expect(1) // Verify this is called exactly once
            .mount(&mock_server)
            .await;

        // Act: Call the client
        let client = Client::new(mock_server.uri()).unwrap();
        let result = client
            .requests_new(
                "test_optimization".to_string(),
                FitnessGoal::minimize(0.05).unwrap(),
                Schedule::generational(5, 20),
                Selector::roulette(),
                Some(serde_json::json!({"test": "value"})),
                None,
            )
            .await;

        // Assert: Verify response parsed correctly
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.request_id, expected_request_id);

        // Mock server automatically verifies expectations were met
    }

    #[tokio::test]
    async fn test_requests_new_handles_error_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/requests"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": "Invalid type_name"
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(mock_server.uri()).unwrap();
        let result = client
            .requests_new(
                "invalid".to_string(),
                FitnessGoal::minimize(0.05).unwrap(),
                Schedule::generational(5, 20),
                Selector::roulette(),
                Some(serde_json::json!({})),
                None,
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("400"));
    }

    #[tokio::test]
    async fn test_requests_interrupt_sends_correct_request() {
        let mock_server = MockServer::start().await;
        let request_id = uuid::Uuid::new_v4();

        Mock::given(method("POST"))
            .and(path(format!("/requests/{}/interrupt", request_id)))
            .and(header(
                "user-agent",
                format!("ga/{}", env!("CARGO_PKG_VERSION")).as_str(),
            ))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = Client::new(mock_server.uri()).unwrap();
        let result = client.requests_interrupt(request_id).await;

        assert!(result.is_ok());
    }
}
