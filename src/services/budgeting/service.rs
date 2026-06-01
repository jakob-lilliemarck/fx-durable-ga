use crate::infrastructure::db;
use crate::services::budgeting::TransactionCreatedEvent;
use crate::services::budgeting::repositories::transactions;
use crate::services::budgeting::repositories::transactions::Transaction;
use tracing::instrument;
use uuid::Uuid;

/// Manages account balances and transaction history.
pub struct Service {
    transactions_wr: transactions::Write,
}

impl Service {
    /// Creates a new budgeting service with the given dependencies.
    pub fn new(transactions_wr: transactions::Write) -> Self {
        Self { transactions_wr }
    }

    /// Append a transaction to an account and emit an event
    #[instrument(level = "debug", skip(self))]
    pub async fn append(
        &self,
        account_id: Uuid,
        account_type: String,
        amount: i64,
        reason: String,
    ) -> Result<Transaction, super::Error> {
        let transaction = db::begin(self.transactions_wr.clone(), |tx| {
            Box::pin(async move {
                let transaction = transactions::WriteTx::new(tx)
                    .append(&account_id, &account_type, amount, &reason)
                    .await?;

                let mut publisher = fx_event_bus::Publisher::new_tx(tx);

                publisher
                    .publish(TransactionCreatedEvent::new(&transaction))
                    .await?;

                Ok(transaction)
            })
        })
        .await?;

        Ok(transaction)
    }
}
