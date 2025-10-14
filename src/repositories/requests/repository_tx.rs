use super::Error;
use crate::{models::Request, repositories::chainable::ToTx};
use sqlx::PgTransaction;
use tracing::instrument;

pub(crate) struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    pub(crate) fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

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
