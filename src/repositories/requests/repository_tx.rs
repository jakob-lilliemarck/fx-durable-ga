use super::Error;
use crate::{models::Request, repositories::chainable::ToTx};
use sqlx::PgTransaction;
use tracing::instrument;

/// Transactional repository for managing genetic algorithm requests within a database transaction.
pub(crate) struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    /// Creates a new TxRepository instance with the given database transaction.
    pub(crate) fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

    /// Creates a new request within the current transaction.
    #[instrument(level = "debug", skip(self), fields(request_id = %request.id, type_name = %request.type_name, type_hash = request.type_hash, goal = ?request.goal))]
    pub(crate) async fn new_request(&mut self, request: Request) -> Result<Request, Error> {
        super::queries::new_request(&mut *self.tx, request).await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
