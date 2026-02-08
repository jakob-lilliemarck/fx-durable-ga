use crate::{chainable::ToTx, repositories::encoders::Encoder};
use sqlx::PgTransaction;

pub struct TxRepository<'tx> {
    pub(super) tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    pub async fn store_encoder(&mut self, encoder: &Encoder) -> Result<Encoder, super::Error> {
        super::queries::store_encoder(&mut *self.tx, encoder).await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    /// Extracts the underlying database transaction.
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
