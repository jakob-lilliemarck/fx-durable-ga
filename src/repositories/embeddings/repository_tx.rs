use sqlx::PgTransaction;

use crate::{
    chainable::ToTx,
    repositories::embeddings::{Embedding, Tag},
};

pub struct TxRepository<'tx> {
    pub(super) tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    pub async fn store_embeddings<'a, I>(
        &mut self,
        embeddings: I,
    ) -> Result<Vec<Embedding>, super::Error>
    where
        I: IntoIterator<Item = &'a Embedding>,
    {
        super::queries::store_embeddings(&mut *self.tx, embeddings).await
    }

    pub async fn store_tags<'a, I>(&mut self, tags: I) -> Result<Vec<Tag>, super::Error>
    where
        I: IntoIterator<Item = &'a Tag>,
    {
        super::queries::store_tags(&mut *self.tx, tags).await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    /// Extracts the underlying database transaction.
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
