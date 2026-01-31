use super::Error;
use crate::models::{Evaluation, Genotype};
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

    /// Records a evaluation result within the current transaction.
    #[instrument(level = "debug", skip(self), fields(evaluation = ?evaluation))]
    pub(crate) fn record_evaluation(
        &mut self,
        evaluation: &Evaluation,
    ) -> impl Future<Output = Result<Evaluation, Error>> {
        super::queries::record_evaluation(&mut *self.tx, evaluation)
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
