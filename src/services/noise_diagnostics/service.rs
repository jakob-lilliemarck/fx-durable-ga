use super::jobs::EvaluateNoiseProbeGenotypeMessage;
use super::repositories::probes::{self, NoiseProbe};
use crate::infrastructure::db;
use crate::repositories::genotypes;
use crate::services::evaluation::{self, SHUTDOWN_SEMAPHORE as EVAL_SHUTDOWN};
use fx_mq_jobs::Queries;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

/// Estimates evaluation noise by repeatedly evaluating probe genotypes.
pub struct Service {
    probe_wr: probes::Write,
    genotypes_ro: genotypes::Read,
    evaluation: Arc<evaluation::Service>,
    mq: Arc<Queries>,
}

impl Service {
    pub fn new(
        probe_wr: probes::Write,
        genotypes_ro: genotypes::Read,
        evaluation: Arc<evaluation::Service>,
        mq: Arc<Queries>,
    ) -> Self {
        Self {
            probe_wr,
            genotypes_ro,
            evaluation,
            mq,
        }
    }

    /// Creates a new noise probe and dispatches evaluation jobs.
    ///
    /// The caller provides a probe `genotype_id` (already stored in the genotypes table)
    /// and a grouping `request_id` for later retrieval. Returns the `probe_id` for tracking.
    ///
    /// The probe genotype must be created with `request_id: None` so its evaluations
    /// do not appear in optimization-specific queries (population stats, min/max fitness).
    ///
    /// Dispatches `evaluation_count` `EvaluateNoiseProbeGenotypeMessage` jobs atomically
    /// alongside storing the probe record. Each job evaluates the genotype exactly once.
    #[instrument(level = "info", skip(self))]
    pub async fn new_noise_probe(
        &self,
        genotype_id: Uuid,
        request_id: Uuid,
        evaluation_count: i32,
    ) -> Result<Uuid, super::Error> {
        if evaluation_count <= 0 {
            return Err(super::Error::InvalidEvaluationCount(evaluation_count));
        }

        let probe = NoiseProbe::new(genotype_id, request_id, evaluation_count);
        let probe_id = probe.id();

        let jobs: Vec<EvaluateNoiseProbeGenotypeMessage> = (0..evaluation_count)
            .map(|_| EvaluateNoiseProbeGenotypeMessage::new(probe_id, genotype_id))
            .collect();

        let mq = self.mq.clone();
        db::begin(self.probe_wr.clone(), |tx| {
            Box::pin(async move {
                let mut wr = probes::WriteTx::new(tx);
                wr.store_noise_probe(&probe).await?;

                let mut publisher = fx_mq_jobs::Publisher::new_tx(wr.tx(), &mq);
                publisher.publish_many(&jobs).await?;

                Ok(())
            })
        })
        .await?;

        Ok(probe_id)
    }

    /// Evaluates a probe genotype (called by the job handler).
    #[instrument(level = "debug", skip(self))]
    pub(super) async fn evaluate_probe(
        &self,
        _probe_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), super::Error> {
        let genotype = self
            .genotypes_ro
            .get_genotype(&genotype_id)
            .await
            .map_err(|e| super::Error::Internal(anyhow::Error::from(e)))?;

        self.evaluation
            .evaluate_genotype(genotype, vec![], vec![EVAL_SHUTDOWN])
            .await
            .map_err(|e| super::Error::Internal(anyhow::Error::from(e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repositories::genotypes::{Genotype, store_genotypes};
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector, store_request};
    use crate::test_tools::TestConfig;
    use fx_mq_building_blocks::testing_tools::TestQueries;
    use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
    use sqlx::PgPool;
    use std::sync::Arc;

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

    async fn setup(pool: PgPool) -> anyhow::Result<(Arc<Service>, PgPool)> {
        let mut c = TestConfig::new(pool.clone()).build().await?;
        let svc = c.get::<Arc<Service>>().await?;
        Ok((svc, pool))
    }

    #[sqlx::test(migrations = false)]
    async fn new_noise_probe_returns_error_for_invalid_count(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (svc, _) = setup(pool.clone()).await?;

        let result = svc.new_noise_probe(Uuid::now_v7(), Uuid::now_v7(), 0).await;

        assert!(result.is_err());
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn new_noise_probe_dispatches_correct_number_of_jobs(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (svc, pool) = setup(pool.clone()).await?;

        let request_id = create_request(&pool).await;
        let genotype = Genotype::new("test", serde_json::json!([1]), None, None, None, None)?;
        let stored = store_genotypes(&pool, &[genotype]).await?;
        let genotype_id = stored[0].id();

        let count = 3;
        let probe_id = svc.new_noise_probe(genotype_id, request_id, count).await?;

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let jobs = mq.get_all_messages(&mut tx).await?;
        tx.commit().await?;

        let probe_jobs: Vec<_> = jobs
            .iter()
            .filter(|j| j.name == "EvaluateNoiseProbeGenotype")
            .collect();
        assert_eq!(probe_jobs.len(), count as usize);

        for job in &probe_jobs {
            let payload: serde_json::Value = serde_json::from_value(job.payload.clone())?;
            assert_eq!(payload["probe_id"], serde_json::json!(probe_id));
            assert_eq!(payload["genotype_id"], serde_json::json!(genotype_id));
        }

        Ok(())
    }
}
