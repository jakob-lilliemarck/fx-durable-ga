use super::Error;
use crate::repositories::chainable::{FromTx, ToTx};
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

    #[instrument(level = "debug", skip(self), fields(individuals_count = individuals.len()))]
    pub(crate) fn add_to_population(
        &mut self,
        individuals: &[(Uuid, Uuid)],
    ) -> impl Future<Output = Result<(), Error>> {
        super::queries::add_to_population(&mut *self.tx, individuals)
    }

    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, genotype_id = %genotype_id))]
    pub(crate) fn remove_from_population(
        &mut self,
        request_id: &Uuid,
        genotype_id: &Uuid,
    ) -> impl Future<Output = Result<(), Error>> {
        super::queries::remove_from_population(&mut *self.tx, request_id, genotype_id)
    }
}

impl<'tx> FromTx<'tx> for TxRepository<'tx> {
    fn from_tx(other: impl ToTx<'tx>) -> Self {
        TxRepository { tx: other.tx() }
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
