use super::{Error, TxRepository};
use crate::models::{Genotype, Population};
use crate::repositories::chainable::{Chain, ToTx, TxType};
use futures::{Future, future::BoxFuture};
use sqlx::{PgPool, PgTransaction};
use tracing::instrument;
use uuid::Uuid;

/// Repository for genotype and fitness data operations.
pub(crate) struct Repository {
    pool: PgPool,
}

impl Repository {
    /// Creates a new genotypes repository with the given database pool.
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Retrieves a genotype by its ID.
    #[instrument(level = "debug", skip(self), fields(genotype_id = %id))]
    pub(crate) async fn get_genotype(&self, id: &Uuid) -> Result<Genotype, Error> {
        super::queries::get_genotype(&self.pool, id).await
    }

    /// Gets population statistics for an optimization request.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub(crate) fn get_population(
        &self,
        request_id: &Uuid,
    ) -> impl Future<Output = Result<Population, Error>> {
        super::queries::get_population(&self.pool, request_id)
    }

    /// Searches genotypes with filtering and ordering options.
    #[instrument(level = "debug", skip(self), fields(filter = ?filter))]
    pub(crate) fn search_genotypes(
        &self,
        filter: &super::queries::Filter,
        limit: i64,
    ) -> impl Future<Output = Result<Vec<(Genotype, Option<f64>)>, Error>> {
        super::queries::search_genotypes(&self.pool, filter, limit)
    }

    /// Finds which genome hashes already exist for deduplication.
    #[instrument(level = "debug", skip(self), fields(filter = %request_id))]
    pub(crate) fn get_intersection(
        &self,
        request_id: Uuid,
        hashes: &[i64],
    ) -> impl Future<Output = Result<Vec<i64>, Error>> {
        super::queries::get_intersection(&self.pool, request_id, hashes)
    }

    /// Checks if any genotypes exist for the given request and generation.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id, generation_id = generation_id))]
    pub(crate) fn check_if_generation_exists(
        &self,
        request_id: Uuid,
        generation_id: i32,
    ) -> impl Future<Output = Result<bool, Error>> {
        super::queries::check_if_generation_exists(&self.pool, request_id, generation_id)
    }
}

impl<'tx> TxType<'tx> for Repository {
    type TxType = TxRepository<'tx>;
    type TxError = Error;
}

impl<'tx> Chain<'tx> for Repository {
    /// Executes a function within a database transaction.
    #[instrument(level = "debug", skip(self, f))]
    fn chain<F, R, T>(&'tx self, f: F) -> BoxFuture<'tx, Result<T, Self::TxError>>
    where
        R: ToTx<'tx>,
        F: FnOnce(Self::TxType) -> BoxFuture<'tx, Result<(R, T), anyhow::Error>>
            + Send
            + Sync
            + 'tx,
        T: Send + Sync + 'tx,
    {
        Box::pin(async move {
            let pool = self.pool.clone();
            let tx = pool.begin().await?;

            let (tx, ret) = f(TxRepository::new(tx))
                .await
                .map_err(|err| Error::Tx(err))?;

            let tx: PgTransaction<'_> = tx.tx();
            tx.commit()
                .await
                .map_err(|err| Error::Tx(anyhow::Error::new(err)))?;

            Ok(ret)
        })
    }
}
