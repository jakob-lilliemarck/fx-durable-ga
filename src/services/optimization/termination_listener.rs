use std::{sync::Arc, time::Duration};
use tokio::{
    sync::broadcast::{self, Sender},
    task::{JoinError, JoinHandle},
};
use tracing::instrument;
use uuid::Uuid;

use crate::repositories;

const DEFAULT_BUFFER: usize = 64;

/// Errors that can occur while forwarding termination notifications.
#[derive(Debug, thiserror::Error)]
pub enum TerminationListenerError {
    #[error("Invalid request_id payload: {payload}")]
    InvalidPayload {
        payload: String,
        #[source]
        source: uuid::Error,
    },
    #[error("RequestsRepository error: {0}")]
    RequestsError(#[from] repositories::requests::Error),
    #[error("Listener error: {0}")]
    Listener(#[from] sqlx::Error),
    #[error("Listener join error: {0}")]
    Join(#[from] JoinError),
    #[error("Termination listener channel closed")]
    ChannelClosed,
}

/// Fan-out listener that relays request termination notifications to subscribers.
pub struct TerminationListener {
    tx: Sender<Uuid>,
    requests: Arc<repositories::requests::Repository>,
    handle: Option<JoinHandle<Result<(), TerminationListenerError>>>,
}

impl TerminationListener {
    /// Spawns a background listener with the default buffer size.
    #[instrument(level = "debug", skip(requests))]
    pub fn new(requests: &Arc<repositories::requests::Repository>) -> Self {
        let (tx, _) = broadcast::channel(DEFAULT_BUFFER);
        let task_tx = tx.clone();
        let task_requests = requests.clone();

        let handle = tokio::spawn(async move {
            let mut listener = task_requests.listen_request_conclusions().await?;

            loop {
                let notification = listener.recv().await?;
                let payload = notification.payload().to_string();
                let request_id = Uuid::parse_str(&payload).map_err(|source| {
                    TerminationListenerError::InvalidPayload { payload, source }
                })?;

                let _ = task_tx.send(request_id);
            }
        });

        Self {
            tx,
            requests: requests.clone(),
            handle: Some(handle),
        }
    }

    /// Waits until the provided request ID is broadcast via the termination channel.
    pub async fn wait_for(&self, request_id: Uuid) -> Result<(), TerminationListenerError> {
        let mut rx = self.tx.subscribe();
        let mut poll = tokio::time::interval(Duration::from_millis(1000));
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                res = rx.recv() => match res {
                    Ok(id) if id == request_id => return Ok(()),
                    Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) =>
                        return Err(TerminationListenerError::ChannelClosed),
                },
                _ = poll.tick() => {
                    if self.requests.get_request_conclusion(&request_id).await?.is_some() {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Stops the background task and waits for it to finish.
    pub async fn stop(mut self) -> Result<(), TerminationListenerError> {
        if let Some(handle) = self.handle.take() {
            if handle.is_finished() {
                return match handle.await {
                    Ok(res) => res,
                    Err(err) => Err(TerminationListenerError::Join(err)),
                };
            }

            handle.abort();
            return match handle.await {
                Ok(_) => Ok(()),
                Err(err) if err.is_cancelled() => Ok(()),
                Err(err) => Err(TerminationListenerError::Join(err)),
            };
        }

        Ok(())
    }
}

impl Drop for TerminationListener {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}
