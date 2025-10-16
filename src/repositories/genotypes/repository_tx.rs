use super::Error;
use crate::models::Genotype;
use crate::repositories::chainable::ToTx;
use sqlx::PgTransaction;
use tracing::instrument;
use uuid::Uuid;

pub(crate) struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    pub(crate) fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

    #[instrument(level = "debug", skip(self, genotypes), fields(request_id = %request_id, generation_id = generation_id, genotypes_count = genotypes.len()))]
    pub(crate) async fn create_generation_if_empty(
        &mut self,
        request_id: Uuid,
        generation_id: i32,
        genotypes: Vec<Genotype>,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::create_generation_if_empty(
            &mut *self.tx,
            request_id,
            generation_id,
            genotypes,
        )
        .await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
