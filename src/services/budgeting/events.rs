use super::repositories::transactions::Transaction;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Event published when a transaction is created.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionCreatedEvent {
    pub amount: i64,
    pub balance: i64,
    pub account_id: Uuid,
    pub account_type: String,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
}

impl fx_event_bus::Event for TransactionCreatedEvent {
    const NAME: &'static str = "TransactionCreated";
}

impl TransactionCreatedEvent {
    /// Creates a new transaction created event.
    pub fn new(t: &Transaction) -> Self {
        Self {
            amount: t.amount,
            balance: t.balance,
            account_id: t.account_id,
            account_type: t.account_type.clone(),
            timestamp: t.timestamp,
            reason: t.reason.clone(),
        }
    }
}
