use super::errors::Error;
use super::queries::{
    AggregatedFitness, GetAggregatedFitnessFilter, GetEvaluationStatsFilter,
    GetEvaluationTimingsFilter, SearchEvaluationsFilter,
};
use super::{Evaluation, EvaluationPopulation, TimingsSummary};
use crate::infrastructure::db;
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
    type Error = Error;

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

    /// Aggregates fitness values into bins for a given optimization request.
    ///
    /// Each bin represents a fixed number of evaluations ordered by `generated_at`.
    /// Supports cursor-based pagination and optional percentile calculations.
    #[instrument(level = "debug", skip(self), fields(filter = ?filter))]
    pub async fn get_aggregated_fitness(
        &self,
        request_id: &Uuid,
        bin_size: i64,
        limit: i64,
        filter: &GetAggregatedFitnessFilter,
    ) -> Result<Vec<AggregatedFitness>, Error> {
        super::queries::get_aggregated_fitness(&self.ro.pool, request_id, bin_size, limit, filter)
            .await
    }

    /// Returns timing statistics (min/max/avg/percentiles) for evaluations matching the filter.
    #[allow(dead_code)]
    #[instrument(level = "debug", skip(self), fields(filter = ?filter))]
    pub async fn get_evaluation_timings(
        &self,
        filter: &GetEvaluationTimingsFilter,
        percentiles: &[f64],
    ) -> Result<TimingsSummary, Error> {
        super::queries::get_evaluation_timings(&self.ro.pool, filter, percentiles).await
    }

    /// Search evaluations with flexible filtering and ordering.
    #[instrument(level = "debug", skip(self), fields(filter = ?filter))]
    pub async fn search_evaluations(
        &self,
        filter: &SearchEvaluationsFilter,
    ) -> Result<Vec<Evaluation>, Error> {
        super::queries::search_evaluations(&self.ro.pool, filter).await
    }

    /// Returns aggregate statistics (COUNT, MIN, MAX) for evaluations matching the given filter.
    ///
    /// Only considers the most recent `limit` evaluations ordered by `(generated_at DESC, id DESC)`.
    /// Supports cursor-based offset pagination.
    #[instrument(level = "debug", skip(self), fields(filter = ?filter, limit = limit))]
    pub async fn get_evaluation_stats(
        &self,
        filter: &GetEvaluationStatsFilter,
        limit: i64,
    ) -> Result<EvaluationPopulation, Error> {
        super::queries::get_evaluation_stats(&self.ro.pool, filter, limit).await
    }
}

impl Write {
    pub fn new(wr: db::WritePool) -> Self {
        Self { wr }
    }
}

impl<'tx> WriteTx<'tx> {
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }

    /// Inserts evaluation records within the current transaction.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn store_evaluations(
        &mut self,
        evaluations: &[Evaluation],
    ) -> Result<Vec<Evaluation>, Error> {
        super::queries::store_evaluations(&mut **self.tx, evaluations).await
    }
}
