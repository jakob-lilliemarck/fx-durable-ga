use super::Error;
use crate::models::Morphology;
use sqlx::PgPool;
use tracing::instrument;

/// Repository for morphology data operations.
pub(crate) struct Repository {
    pool: PgPool,
}

impl Repository {
    /// Creates a new morphologies repository with the given database pool.
    pub(crate) fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Inserts a new morphology into the database.
    #[instrument(level = "debug", skip(self), fields(type_name = %morphology.type_name, type_hash = morphology.type_hash))]
    pub(crate) async fn new_morphology(&self, morphology: Morphology) -> Result<Morphology, Error> {
        super::queries::new_morphology(&self.pool, morphology).await
    }

    /// Retrieves a morphology by its type hash.
    #[instrument(level = "debug", skip(self), fields(type_hash = type_hash))]
    pub(crate) async fn get_morphology(&self, type_hash: i32) -> Result<Morphology, Error> {
        super::queries::get_morphology(&self.pool, type_hash).await
    }
}
