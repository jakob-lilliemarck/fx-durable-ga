use crate::{chainable::ToTx, repositories::encoders::Encoder};
use sqlx::PgTransaction;
use uuid::Uuid;

pub struct TxRepository<'tx> {
    pub(super) tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    pub async fn store_encoder(&mut self, encoder: &Encoder) -> Result<Encoder, super::Error> {
        super::queries::store_encoder(&mut *self.tx, encoder).await
    }

    pub(crate) async fn toggle_encoder_pairing(
        &mut self,
        type_hash: i32,
        encoder_id: &Uuid,
        is_enabled: bool,
    ) -> Result<super::queries::TogglingResult, super::Error> {
        super::queries::toggle_encoder_pairing(&mut *self.tx, type_hash, encoder_id, is_enabled)
            .await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    /// Extracts the underlying database transaction.
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
