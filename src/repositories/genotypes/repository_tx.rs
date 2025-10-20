use super::Error;
use crate::models::{Fitness, Genotype};
use crate::repositories::chainable::ToTx;
use futures::Future;
use sqlx::PgTransaction;
use tracing::instrument;

/// Transaction-scoped repository for genotype operations within a database transaction.
pub(crate) struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl<'tx> TxRepository<'tx> {
    /// Creates a new transaction repository with the given database transaction.
    pub(crate) fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

    /// Records a fitness evaluation result within the current transaction.
    #[instrument(level = "debug", skip(self), fields(fitness = ?fitness))]
    pub(crate) fn record_fitness(
        &mut self,
        fitness: &Fitness,
    ) -> impl Future<Output = Result<Fitness, Error>> {
        super::queries::record_fitness(&mut *self.tx, fitness)
    }

    /// Inserts multiple genotypes within the current transaction.
    #[instrument(level = "debug", skip(self), fields(genotypes_count = genotypes.len()))]
    pub(crate) fn new_genotypes(
        &mut self,
        genotypes: Vec<Genotype>,
    ) -> impl Future<Output = Result<Vec<Genotype>, Error>> {
        super::queries::new_genotypes(&mut *self.tx, genotypes)
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    /// Extracts the underlying database transaction.
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
