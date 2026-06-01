//! Genotype endpoint methods for the API client

use super::Client;
use crate::controllers::BackfillGenotypeEmbeddingsRequest;
use anyhow::{Context, Result};
use uuid::Uuid;

impl Client {
    /// POST /genotypes/embeddings - Backfill missing genotype embeddings
    ///
    /// # Arguments
    ///
    /// * `request_id` - Optional request ID filter
    /// * `generation_ids` - Optional generation ID filters
    /// * `genotype_ids` - Optional genotype ID filters
    /// * `has_evaluation` - Optional evaluation status filter
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The server returns a non-202 status code
    pub async fn genotypes_backfill_embeddings(
        &self,
        request_id: Option<Uuid>,
        generation_ids: Vec<i32>,
        genotype_ids: Vec<Uuid>,
        has_evaluation: Option<bool>,
    ) -> Result<()> {
        let payload = BackfillGenotypeEmbeddingsRequest {
            request_id,
            generation_ids: if generation_ids.is_empty() {
                None
            } else {
                Some(generation_ids.iter().map(|id| id.to_string()).collect())
            },
            genotype_ids: if genotype_ids.is_empty() {
                None
            } else {
                Some(genotype_ids)
            },
            has_evaluation,
        };

        let url = format!(
            "{}/genotypes/embeddings",
            self.base_url.trim_end_matches('/')
        );
        let response = self
            .http
            .post(&url)
            .json(&payload)
            .send()
            .await
            .context("Failed to send request")?;

        if response.status() != reqwest::StatusCode::ACCEPTED {
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
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, method, path},
    };

    #[tokio::test]
    async fn test_genotypes_backfill_embeddings_sends_correct_request() {
        let mock_server = MockServer::start().await;
        let request_id = uuid::Uuid::new_v4();
        let genotype_id = uuid::Uuid::new_v4();

        Mock::given(method("POST"))
            .and(path("/genotypes/embeddings"))
            .and(body_json(serde_json::json!({
                "request_id": request_id,
                "generation_ids": ["1", "2"],
                "genotype_ids": [genotype_id],
                "has_evaluation": true
            })))
            .respond_with(ResponseTemplate::new(202))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = Client::new(mock_server.uri()).unwrap();
        let result = client
            .genotypes_backfill_embeddings(
                Some(request_id),
                vec![1, 2],
                vec![genotype_id],
                Some(true),
            )
            .await;

        assert!(result.is_ok());
    }
}
