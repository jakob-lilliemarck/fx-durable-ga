use crate::services::evaluation::repositories::evaluations;
use crate::{
    configuration::HostId, infrastructure::db, repositories::genotypes, services::synchronization,
};
use chrono::{DateTime, Utc};
use evaluations::Evaluation;
use futures::{
    FutureExt,
    future::{self, BoxFuture},
};
use genotypes::Genotype;
use serde::{Serialize, de::DeserializeOwned};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use tracing::instrument;
use uuid::Uuid;

/// Semaphore name used for graceful shutdown. When raised, all in-progress
/// evaluations abort with `Aborted` and their jobs are requeued for retry.
pub(crate) const SHUTDOWN_SEMAPHORE: &str = "shutdown";

/// Evaluates genotypes and manages evaluation lifecycle.
pub struct Service {
    host_id: HostId,
    evaluators: Arc<RwLock<HashMap<&'static str, Arc<dyn EvaluatorErased>>>>,
    synchronization: Arc<synchronization::Service>,
    genotypes_ro: genotypes::Read,
    evaluations_ro: evaluations::Read,
    evaluations_wr: evaluations::Write,
}

impl Service {
    pub(crate) fn new(
        host_id: HostId,
        synchronization: Arc<synchronization::Service>,
        genotypes_ro: genotypes::Read,
        evaluations_ro: evaluations::Read,
        evaluations_wr: evaluations::Write,
    ) -> Self {
        Self {
            host_id,
            evaluators: Arc::new(RwLock::new(HashMap::new())),
            synchronization,
            genotypes_ro,
            evaluations_ro,
            evaluations_wr,
        }
    }

    /// Registers an evaluator for a genotype type.
    #[instrument(level = "info", skip(self, evaluator))]
    pub async fn register<T>(&self, type_name: &'static str, evaluator: T)
    where
        T: Evaluator + 'static,
    {
        let mut write_lock = self.evaluators.write().await;

        write_lock.insert(type_name, Arc::new(evaluator));

        tracing::info!("Evaluator registered");
    }

    #[instrument(level = "info", skip(self))]
    async fn get(&self, type_name: &str) -> Result<Arc<dyn EvaluatorErased>, super::Error> {
        let read_lock = self.evaluators.read().await;

        let evaluator = read_lock.get(type_name).ok_or(super::Error::NotFound {
            type_name: type_name.to_string(),
        })?;

        Ok(evaluator.clone())
    }

    #[instrument(level = "info", skip(self))]
    async fn evaluate(
        &self,
        type_name: &str,
        genome: serde_json::Value,
    ) -> Result<(f64, DateTime<Utc>), super::Error> {
        let evaluator = self.get(type_name).await?;

        let fitness = evaluator.evaluate(genome).await?;

        let evaluated_at = Utc::now();

        Ok((fitness, evaluated_at))
    }

    /// Evaluates a genotype and stores the result.
    ///
    /// `drop_on` semaphores cause the evaluation to abort silently — the job
    /// returns `Ok(())` and will not be retried. Use when the evaluation is
    /// no longer needed (e.g., optimization goal reached).
    ///
    /// `retry_on` semaphores cause the evaluation to abort with an error —
    /// the job will be retried later. Use for transient conditions like
    /// application shutdown.
    #[instrument(level = "info", skip(self))]
    pub(crate) async fn evaluate_genotype(
        &self,
        genotype: Genotype,
        group_id: Uuid,
        reason: &str,
        drop_on: Vec<&str>,
        retry_on: Vec<&str>,
    ) -> Result<(), super::Error> {
        let started_at = Utc::now();

        // Evaluate, with optional semaphore-based abort/retry
        let (fitness, completed_at) = if drop_on.is_empty() && retry_on.is_empty() {
            self.evaluate(&genotype.type_name, genotype.genome).await?
        } else {
            // Merge both lists into one, tracking the boundary for dispatch
            let drop_count = drop_on.len();
            let all_names: Vec<&str> = drop_on.into_iter().chain(retry_on).collect();

            let futures: Vec<_> = all_names
                .iter()
                .map(|s| {
                    self.synchronization.wait_for(s, &started_at).boxed() as future::BoxFuture<_>
                })
                .collect();

            let semaphores = future::select_all(futures);

            tokio::select! {
                result = self.evaluate(&genotype.type_name, genotype.genome) => result?,
                (raised, index, _) = semaphores => {
                    if index < drop_count {
                        return Ok(raised?);
                    }
                    return Err(super::Error::Aborted(all_names[index].to_string()));
                }
            }
        };

        let host_id = self.host_id.clone();
        let reason = reason.to_string();
        db::begin(self.evaluations_wr.clone(), |tx| {
            Box::pin(async move {
                let evaluation = Evaluation::new(
                    genotype.id,
                    group_id,
                    reason.clone(),
                    fitness,
                    Some(started_at),
                    Some(completed_at),
                    Some(host_id.value),
                );
                let evaluation = evaluation.with_generated_at(genotype.generated_at);
                let evaluation_id = evaluation.id();
                evaluations::WriteTx::new(tx)
                    .store_evaluations(&[evaluation])
                    .await?;

                let mut publisher = fx_event_bus::Publisher::new_tx(tx);

                publisher
                    .publish(super::GenotypeEvaluatedEvent::new(
                        evaluation_id,
                        group_id,
                        reason.clone(),
                        genotype.id,
                        fitness,
                    ))
                    .await?;

                Ok(())
            })
        })
        .await?;

        Ok(())
    }
}

/// Evaluates a typed genome and returns a fitness score.
pub trait Evaluator: Send + Sync {
    type Type: Serialize + DeserializeOwned + Send + Sync;

    fn evaluate<'a>(
        &'a self,
        instance: &'a Self::Type,
    ) -> BoxFuture<'a, Result<f64, Box<dyn std::error::Error + Send + Sync>>>;
}

trait EvaluatorErased: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        instance: serde_json::Value,
    ) -> BoxFuture<'a, Result<f64, super::Error>>;
}

impl<T> EvaluatorErased for T
where
    T: Evaluator + 'static,
{
    fn evaluate<'a>(
        &'a self,
        instance: serde_json::Value,
    ) -> BoxFuture<'a, Result<f64, super::Error>> {
        Box::pin(async move {
            let typed = serde_json::from_value::<T::Type>(instance)?;

            let fitness = Evaluator::evaluate(self, &typed).await?;

            Ok(fitness)
        })
    }
}

#[cfg(test)]
mod tests_evaluate_genotype {
    use super::*;
    use crate::services::evaluation;
    use crate::services::optimization::{self, store_request};
    use crate::test_tools::TestConfig;
    use futures::future::BoxFuture;
    use sqlx::PgPool;
    use std::sync::Arc;
    use std::time::Duration;

    async fn create_request(pool: &PgPool) -> Uuid {
        let request = optimization::Request::new(
            "test",
            optimization::FitnessGoal::maximize(1.0).unwrap(),
            optimization::Selector::tournament(3),
            optimization::Schedule::generational(10, 2),
        );
        let id = request.id;
        store_request(pool, request).await.unwrap();
        id
    }

    struct SlowEvaluator {
        started: Arc<tokio::sync::Notify>,
    }

    impl SlowEvaluator {
        fn new() -> Self {
            Self {
                started: Arc::new(tokio::sync::Notify::new()),
            }
        }
    }

    impl Evaluator for SlowEvaluator {
        type Type = serde_json::Value;

        fn evaluate<'a>(
            &'a self,
            _instance: &'a Self::Type,
        ) -> BoxFuture<'a, Result<f64, Box<dyn std::error::Error + Send + Sync>>> {
            let started = self.started.clone();
            Box::pin(async move {
                started.notify_one();
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok(0.5)
            })
        }
    }

    struct TestContext {
        evaluation: Arc<evaluation::Service>,
        synchronization: Arc<crate::services::synchronization::Service>,
    }

    async fn setup(pool: PgPool) -> anyhow::Result<TestContext> {
        let mut c = TestConfig::new(pool).build().await?;

        let synchronization = c
            .get::<Arc<crate::services::synchronization::Service>>()
            .await?;
        let evaluation = c.get::<Arc<evaluation::Service>>().await?;
        evaluation.register("test", SlowEvaluator::new()).await;

        Ok(TestContext {
            evaluation,
            synchronization,
        })
    }

    async fn setup_with_listeners_and_started(
        pool: PgPool,
    ) -> anyhow::Result<(TestContext, Arc<tokio::sync::Notify>)> {
        let started = Arc::new(tokio::sync::Notify::new());
        let mut c = TestConfig::new(pool).with_listeners().build().await?;

        let synchronization = c
            .get::<Arc<crate::services::synchronization::Service>>()
            .await?;
        let evaluation = c.get::<Arc<evaluation::Service>>().await?;
        evaluation
            .register(
                "test",
                SlowEvaluator {
                    started: started.clone(),
                },
            )
            .await;

        Ok((
            TestContext {
                evaluation,
                synchronization,
            },
            started,
        ))
    }

    #[sqlx::test(migrations = false)]
    async fn it_evaluates_genotype_normally(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let ctx = setup(pool.clone()).await?;

        let request_id = create_request(&pool).await;
        let genotype = Genotype::new(
            "test",
            serde_json::json!([1, 2]),
            request_id,
            None,
            None,
            None,
        )?;
        let stored = crate::repositories::genotypes::store_genotypes(&pool, &[genotype]).await?;
        let genotype = stored.into_iter().next().unwrap();

        ctx.evaluation
            .evaluate_genotype(genotype, Uuid::nil(), "test", vec![], vec![])
            .await
            .unwrap();

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_ok_on_drop_semaphore(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (ctx, started) = setup_with_listeners_and_started(pool.clone()).await?;

        let request_id = create_request(&pool).await;
        let genotype = Genotype::new(
            "test",
            serde_json::json!([1, 2]),
            request_id,
            None,
            None,
            None,
        )?;
        let stored = crate::repositories::genotypes::store_genotypes(&pool, &[genotype]).await?;
        let genotype = stored.into_iter().next().unwrap();

        let sync = ctx.synchronization.clone();
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            started.notified().await;
            let mut tx = pool_clone.begin().await.unwrap();
            sync.raise(&mut tx, "test-drop").await.unwrap();
            tx.commit().await.unwrap();
        });

        let result = ctx
            .evaluation
            .evaluate_genotype(genotype, Uuid::nil(), "test", vec!["test-drop"], vec![])
            .await;
        assert!(result.is_ok());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_aborted_on_retry_semaphore(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (ctx, started) = setup_with_listeners_and_started(pool.clone()).await?;

        let request_id = create_request(&pool).await;
        let genotype = Genotype::new(
            "test",
            serde_json::json!([1, 2]),
            request_id,
            None,
            None,
            None,
        )?;

        let sync = ctx.synchronization.clone();
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            started.notified().await;
            let mut tx = pool_clone.begin().await.unwrap();
            sync.raise(&mut tx, "test-retry").await.unwrap();
            tx.commit().await.unwrap();
        });

        let result = ctx
            .evaluation
            .evaluate_genotype(genotype, Uuid::nil(), "test", vec![], vec!["test-retry"])
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::services::evaluation::Error::Aborted(_)
        ));

        Ok(())
    }
}
