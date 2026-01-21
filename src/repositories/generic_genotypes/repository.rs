use super::Error;
use crate::models::GenericGenotype;
use sqlx::PgPool;
use uuid::Uuid;

/// Repository for type-erased genotype data operations using the `generic_genotypes` table.
pub(crate) struct Repository {
    pool: PgPool,
}

impl Repository {
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub(crate) async fn get_genotype(&self, id: &Uuid) -> Result<GenericGenotype, Error> {
        super::queries::get_genotype(&self.pool, id).await
    }

    pub(crate) async fn new_genotypes(
        &self,
        genotypes: Vec<GenericGenotype>,
    ) -> Result<Vec<GenericGenotype>, Error> {
        super::queries::new_genotypes(&self.pool, genotypes).await
    }
}
