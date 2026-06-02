use super::errors::Error;
use crate::infrastructure::db;
use crate::repositories::genotypes::{Genotype, GenotypePopulation};
use sqlx::PgTransaction;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Read {
    ro: db::ReadPool,
}

#[derive(Debug, Clone)]
pub struct Write {
    wr: db::WritePool,
}

pub struct WriteTx<'tx> {
    tx: &'tx mut PgTransaction<'static>,
}

impl db::Tx for Write {
    type Error = super::Error;

    fn tx(self) -> db::TxFut<Self::Error> {
        let pool = self.wr.pool.clone();
        Box::pin(async move {
            let tx = pool.begin().await?;
            Ok(tx)
        })
    }
}

impl Read {
    pub fn new(ro: db::ReadPool) -> Self {
        Self { ro }
    }

    /// Retrieves a genotype by its ID.
    #[instrument(level = "debug", skip(self), fields(genotype_id = %id))]
    pub(crate) async fn get_genotype(&self, id: &Uuid) -> Result<Genotype, Error> {
        let mut genotypes = super::queries::search_genotypes(
            &self.ro.pool,
            &super::queries::SearchGenotypesFilter::default().with_genotype_id(*id),
            1,
        )
        .await?;

        if genotypes.is_empty() {
            return Err(Error::NotFound(*id));
        }

        let genotype = genotypes.remove(0);

        Ok(genotype)
    }

    /// Gets population statistics for an optimization request (genotypes only).
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) async fn get_population(
        &self,
        request_id: &Uuid,
    ) -> Result<GenotypePopulation, Error> {
        super::queries::get_population(&self.ro.pool, request_id).await
    }

    /// Searches genotypes with filtering and ordering options.
    #[instrument(level = "debug", skip(self), fields(filter = ?filter))]
    pub async fn search_genotypes(
        &self,
        filter: &super::queries::SearchGenotypesFilter,
        limit: i64,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::search_genotypes(&self.ro.pool, filter, limit).await
    }

    /// Checks if any genotypes exist for the given request and generation.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, generation_id = generation_id))]
    pub(crate) async fn check_if_generation_exists(
        &self,
        request_id: &Uuid,
        generation_id: i32,
    ) -> Result<bool, Error> {
        super::queries::check_if_generation_exists(&self.ro.pool, &request_id, generation_id).await
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = %genotype_id, degree = degree))]
    pub(crate) async fn get_ancestors(
        &self,
        genotype_id: &Uuid,
        degree: i32,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::get_ancestors(&self.ro.pool, genotype_id, degree).await
    }

    #[instrument(level = "debug", skip(self), fields(genotype_id = %genotype_id, degree = degree))]
    pub(crate) async fn get_descendants(
        &self,
        genotype_id: &Uuid,
        degree: i32,
    ) -> Result<Vec<Genotype>, Error> {
        super::queries::get_descendants(&self.ro.pool, genotype_id, degree).await
    }
}

impl Write {
    pub fn new(wr: db::WritePool) -> Self {
        Self { wr }
    }
}

impl<'tx> WriteTx<'tx> {
    /// Creates a new transaction repository with the given database transaction.
    #[instrument(level = "debug", skip(tx))]
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }

    /// Inserts multiple genotypes within the current transaction.
    #[instrument(level = "debug", skip(self, genotypes))]
    pub(crate) async fn store_genotypes<'a, I>(
        &mut self,
        genotypes: I,
    ) -> Result<Vec<Genotype>, Error>
    where
        I: IntoIterator<Item = &'a Genotype>,
    {
        super::queries::store_genotypes(&mut **self.tx, genotypes).await
    }
}
