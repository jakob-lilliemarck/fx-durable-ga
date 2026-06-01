use crate::{
    configuration::HostId,
    infrastructure::db,
    repositories::genotypes,
    services::synchronization,
};
use crate::services::evaluation::repositories::evaluations;
use chrono::{DateTime, Utc};
use evaluations::{Evaluation, SearchEvaluationsFilter};
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

    /// Returns the minimum fitness for a given optimization request.
    #[instrument(level = "debug", skip(self), fields(request_id = %request_id))]
    pub async fn get_min_fitness(&self, request_id: Uuid) -> Result<Option<f64>, super::Error> {
        let mut evals = self
            .evaluations_ro
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
    pub async fn get_max_fitness(&self, request_id: Uuid) -> Result<Option<f64>, super::Error> {
        let mut evals = self
            .evaluations_ro
            .search_evaluations(
                &SearchEvaluationsFilter::default()
                    .with_request_ids(vec![request_id])
                    .with_order_fitness_desc()
                    .with_limit(1),
            )
            .await?;
        Ok(evals.pop().map(|e| e.fitness()))
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
    #[instrument(level = "info", skip(self))]
    pub(crate) async fn evaluate_genotype(
        &self,
        genotype: Genotype,
        semaphores: Vec<&str>,
    ) -> Result<(), super::Error> {
        let started_at = Utc::now();

        // Wait for the first semaphore
        let semaphores = future::select_all(semaphores.iter().map(|s| {
            self.synchronization.wait_for(s, &started_at).boxed() as future::BoxFuture<_>
        }));

        let (fitness, completed_at) = tokio::select! {
            result = self.evaluate(&genotype.type_name, genotype.genome) => result?,
            (raised, _, _) = semaphores => return Ok(raised?)
        };

        let host_id = self.host_id.clone();
        db::begin(self.evaluations_wr.clone(), |tx| {
            Box::pin(async move {
                let mut evaluation = Evaluation::new(
                    genotype.id,
                    fitness,
                    Some(started_at),
                    Some(completed_at),
                    Some(host_id.value),
                );
                if let Some(request_id) = genotype.request_id {
                    evaluation = evaluation.with_request_id(request_id);
                }
                evaluations::WriteTx::new(tx)
                    .store_evaluations(&[evaluation.with_generated_at(genotype.generated_at)])
                    .await?;

                let mut publisher = fx_event_bus::Publisher::new_tx(tx);

                publisher
                    .publish(super::GenotypeEvaluatedEvent::new(
                        genotype.request_id,
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
