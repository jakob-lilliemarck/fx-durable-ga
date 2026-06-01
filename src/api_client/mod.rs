//! HTTP client for the FX Durable GA API
//!
//! This module provides a typed client for making requests to the FX Durable GA
//! HTTP API. It handles HTTP connection management, serialization, and provides
//! type-safe methods for each API endpoint.
//!
//! # Example
//!
//! ```rust,no_run
//! use fx_durable_ga::api_client::Client;
//! use fx_durable_ga::services::optimization::{FitnessGoal, Schedule, Selector};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = Client::new("http://localhost:3000".to_string())?;
//!
//!     let goal = FitnessGoal::minimize(0.05)?;
//!     let schedule = Schedule::generational(10, 50);
//!     let selector = Selector::tournament(3);
//!
//!     let response = client.requests_new(
//!         "my_optimization".to_string(),
//!         goal,
//!         schedule,
//!         selector,
//!     ).await?;
//!
//!     println!("Created request: {}", response.request_id);
//!     Ok(())
//! }
//! ```

mod genotypes;
mod requests;

use anyhow::{Context, Result};
use std::time::Duration;

/// HTTP client for the FX Durable GA API
pub struct Client {
    pub(crate) http: reqwest::Client,
    pub(crate) base_url: String,
}

impl Client {
    /// Creates a new API client with preconfigured HTTP settings
    ///
    /// The client is configured with:
    /// - 2 second connect timeout
    /// - 5 second total request timeout
    /// - User-Agent header with CLI version
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the API server (e.g., "http://localhost:3000")
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be built
    pub fn new(base_url: String) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(5))
            .user_agent(format!("ga/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self { http, base_url })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_construction_succeeds() {
        // Verify the client constructs successfully with our config
        let client = Client::new("http://localhost:3000".to_string()).unwrap();
        assert_eq!(client.base_url, "http://localhost:3000");
    }
}
