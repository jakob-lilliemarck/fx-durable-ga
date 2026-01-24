use crate::{models::Terminated, repositories::requests};
use futures::future::BoxFuture;
use tracing::instrument;
use uuid::Uuid;

/// Checks if an optimization request has been terminated by querying for request conclusions.
pub(crate) struct Terminator {
    request_id: Uuid,
    requests: requests::Repository,
}

impl Terminator {
    /// Creates a new terminator for the given request.
    #[instrument(level = "debug", skip(requests), fields(request_id = %request_id))]
    pub(crate) fn new(requests: requests::Repository, request_id: Uuid) -> Self {
        Terminator {
            request_id,
            requests,
        }
    }
}

impl Terminated for Terminator {
    #[instrument(level = "debug", skip(self), fields(request_id = %self.request_id))]
    fn is_terminated(&self) -> BoxFuture<'_, bool> {
        let requests = self.requests.clone();

        Box::pin(async move {
            match requests.get_request_conclusion(&self.request_id).await {
                Ok(Some(_)) => true, // Any conclusion means terminate
                Err(err) => {
                    tracing::warn!(message = "Failed to check request conclusion", err = ?err);
                    false
                }
                _ => false, // No conclusion yet, keep going
            }
        })
    }
}

// Evaluator pipeline removed; GenotypeManager now owns evaluation.
