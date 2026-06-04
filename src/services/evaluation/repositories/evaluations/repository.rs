use super::errors::Error;
use super::queries::{
    AggregatedFitness, EvaluationAggregates, GetAggregatedFitnessFilter,
    GetEvaluationAggregatesFilter, GetEvaluationStatsFilter, GetEvaluationTimingsFilter,
    SearchEvaluationsFilter,
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

    /// Returns the minimum fitness for a given optimization request.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn get_min_fitness(&self, request_id: Uuid) -> Result<Option<f64>, Error> {
        let mut evals = self
            .search_evaluations(
                &SearchEvaluationsFilter::default()
                    .with_request_ids(vec![request_id])
                    .with_order_fitness_asc()
                    .with_limit(1),
            )
            .await?;
        Ok(evals.pop().map(|e| e.fitness()))
    }

    /// Returns the maximum fitness for a given optimization request.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn get_max_fitness(&self, request_id: Uuid) -> Result<Option<f64>, Error> {
        let mut evals = self
            .search_evaluations(
                &SearchEvaluationsFilter::default()
                    .with_request_ids(vec![request_id])
                    .with_order_fitness_desc()
                    .with_limit(1),
            )
            .await?;
        Ok(evals.pop().map(|e| e.fitness()))
    }

    /// Returns aggregate statistics (COUNT, AVG, STDDEV_POP, VAR_POP) for a set of evaluations.
    #[instrument(level = "debug", skip(self), fields(filter = ?filter))]
    pub async fn get_evaluation_aggregates(
        &self,
        filter: &GetEvaluationAggregatesFilter,
    ) -> Result<EvaluationAggregates, Error> {
        super::queries::get_evaluation_aggregates(&self.ro.pool, filter).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migrations;
    use crate::repositories::genotypes::{self, Genotype};
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector, store_request};
    use chrono::Utc;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn create_request(pool: &PgPool) -> Uuid {
        let request = Request::new(
            "test",
            FitnessGoal::maximize(1.0).unwrap(),
            Selector::tournament(3),
            Schedule::generational(10, 2),
        );
        let id = request.id;
        store_request(pool, request).await.unwrap();
        id
    }

    #[sqlx::test(migrations = false)]
    async fn get_min_fitness_returns_none_when_no_evaluations(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;
        let ro = Read::new(db::ReadPool { pool });
        let result = ro.get_min_fitness(Uuid::now_v7()).await?;
        assert_eq!(result, None);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn get_max_fitness_returns_none_when_no_evaluations(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;
        let ro = Read::new(db::ReadPool { pool });
        let result = ro.get_max_fitness(Uuid::now_v7()).await?;
        assert_eq!(result, None);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn get_min_fitness_returns_lowest_fitness(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;
        let ro = Read::new(db::ReadPool { pool: pool.clone() });
        let request_id = create_request(&pool).await;
        let genotype = Genotype::new(
            "test",
            serde_json::json!([1, 2]),
            Some(request_id),
            None,
            None,
            None,
        )?;
        let stored = crate::repositories::genotypes::store_genotypes(&pool, &[genotype]).await?;
        let genotype = stored.into_iter().next().unwrap();

        let started_at = Utc::now();
        let completed_at = Utc::now();
        let evaluation =
            Evaluation::new(genotype.id, 0.5, Some(started_at), Some(completed_at), None)
                .with_request_id(request_id)
                .with_generated_at(started_at);
        let mut tx = pool.begin().await?;
        {
            let mut wr = WriteTx::new(&mut tx);
            wr.store_evaluations(&[evaluation]).await?;
        }
        tx.commit().await?;

        let result = ro.get_min_fitness(request_id).await?;
        assert_eq!(result, Some(0.5));
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn get_max_fitness_returns_highest_fitness(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;
        let ro = Read::new(db::ReadPool { pool: pool.clone() });
        let request_id = create_request(&pool).await;
        let genotype = Genotype::new(
            "test",
            serde_json::json!([1, 2]),
            Some(request_id),
            None,
            None,
            None,
        )?;
        let stored = crate::repositories::genotypes::store_genotypes(&pool, &[genotype]).await?;
        let genotype = stored.into_iter().next().unwrap();

        let started_at = Utc::now();
        let completed_at = Utc::now();
        let evaluation =
            Evaluation::new(genotype.id, 0.5, Some(started_at), Some(completed_at), None)
                .with_request_id(request_id)
                .with_generated_at(started_at);
        let mut tx = pool.begin().await?;
        {
            let mut wr = WriteTx::new(&mut tx);
            wr.store_evaluations(&[evaluation]).await?;
        }
        tx.commit().await?;

        let result = ro.get_max_fitness(request_id).await?;
        assert_eq!(result, Some(0.5));
        Ok(())
    }
}
