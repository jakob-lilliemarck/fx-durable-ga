use super::queries::Filter;
use crate::infrastructure::db;
use sqlx::PgTransaction;
use tracing::instrument;
use uuid::Uuid;

/// Read-only repository for querying transactions.
#[derive(Debug, Clone)]
pub struct Read {
    ro: db::ReadPool,
}

/// Write repository for appending transactions.
#[derive(Debug, Clone)]
pub struct Write {
    wr: db::WritePool,
}

impl Write {
    pub fn new(wr: db::WritePool) -> Self {
        Self { wr }
    }
}

/// Transaction-bound write repository for appending transactions.
pub struct WriteTx<'tx> {
    tx: &'tx mut PgTransaction<'static>,
}

impl db::Tx for Write {
    type Error = super::Error;

    fn tx(self) -> db::TxFut<Self::Error> {
        let pool = self.wr.pool.clone();
        Box::pin(async move {
            let tx = pool.begin().await?;
            Ok(tx)
        })
    }
}

impl Read {
    pub fn new(ro: db::ReadPool) -> Self {
        Self { ro }
    }

    /// Returns the current balance for the given account.
    #[instrument(level = "debug", skip(self))]
    pub async fn balance(&self, account_id: &Uuid) -> Result<i64, super::Error> {
        super::queries::balance(&self.ro.pool, account_id).await
    }

    /// Returns the transaction history matching the given filter.
    #[instrument(level = "debug", skip(self))]
    pub async fn history(
        &self,
        filter: &Filter,
        limit: i64,
    ) -> Result<Vec<super::Transaction>, super::Error> {
        super::queries::history(&self.ro.pool, filter, limit).await
    }
}

impl<'tx> WriteTx<'tx> {
    #[instrument(level = "debug", skip(tx))]
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }

    /// Appends a transaction to an account and returns the updated balance.
    #[instrument(level = "debug", skip(self))]
    pub async fn append(
        &mut self,
        account_id: &Uuid,
        account_type: &str,
        amount: i64,
        reason: &str,
    ) -> Result<super::Transaction, super::Error> {
        super::queries::append(&mut **self.tx, account_id, account_type, amount, reason).await
    }
}
