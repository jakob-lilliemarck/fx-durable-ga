use super::Error;
use crate::models::{Fitness, Genotype};
use crate::repositories::chainable::ToTx;
use sqlx::PgTransaction;
use tracing::instrument;

pub(crate) struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    pub(crate) fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

    #[instrument(level = "debug", skip(self), fields(fitness = ?fitness))]
    pub(crate) fn record_fitness(
        &mut self,
        fitness: &Fitness,
    ) -> impl Future<Output = Result<Fitness, Error>> {
        super::queries::record_fitness(&mut *self.tx, fitness)
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn new_genotypes(
        &mut self,
        genotypes: Vec<Genotype>,
    ) -> impl Future<Output = Result<Vec<Genotype>, Error>> {
        super::queries::new_genotypes(&mut *self.tx, genotypes)
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
