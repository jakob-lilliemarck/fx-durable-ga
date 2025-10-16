use super::Error;
use crate::{
    models::{Fitness, Individual},
    repositories::chainable::{FromTx, ToTx},
};
use sqlx::PgTransaction;
use tracing::instrument;

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
        individuals: &[Individual],
    ) -> impl Future<Output = Result<Vec<Individual>, Error>> {
        super::queries::add_to_population(&mut *self.tx, individuals)
    }

    #[instrument(level = "debug", skip(self), fields(fitness = ?fitness))]
    pub(crate) fn record_fitness(
        &mut self,
        fitness: &Fitness,
    ) -> impl Future<Output = Result<Fitness, Error>> {
        super::queries::record_fitness(&mut *self.tx, fitness)
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
