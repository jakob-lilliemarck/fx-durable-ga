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

    #[instrument(level = "debug", skip(self), fields(genotypes_count = genotypes.len()))]
    pub(crate) async fn new_genotypes(
        &mut self,
        genotypes: Vec<Genotype>,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::new_genotypes(&mut *self.tx, genotypes).await
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = %id, fitness = fitness))]
    pub(crate) async fn set_fitness(&mut self, id: Uuid, fitness: f64) -> Result<(), Error> {
        super::queries::set_fitness(&mut *self.tx, id, fitness).await
    }
}

impl<'tx> ToTx<'tx> for TxRepository<'tx> {
    fn tx(self) -> PgTransaction<'tx> {
        self.tx
    }
}
