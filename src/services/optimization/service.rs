use super::Breeder;
use super::Error;
use super::optimizer::{self, OptimizerErased};
use crate::infrastructure::db;
use crate::repositories::genotypes;
use crate::repositories::genotypes::Genotype;
use crate::services::budgeting::{TransactionsFilter, TransactionsRead};
use crate::services::evaluation::SHUTDOWN_SEMAPHORE;
use crate::services::evaluation::repositories::evaluations;
use crate::services::evaluation::{GetEvaluationStatsFilter, SearchEvaluationsFilter};
use crate::services::optimization::jobs::{
    AddOptimizationBudgetMessage, ChargeOptimizationBudgetMessage, EvaluateGenotypeMessage,
};
use crate::services::optimization::repositories::requests;
use crate::services::optimization::repositories::requests::SearchRequestsFilter;
use crate::services::optimization::{FitnessGoal, Request, Selector};
use crate::services::{budgeting, locking};
use crate::services::{evaluation, synchronization};
use futures::lock::Mutex;
use fx_mq_jobs::Queries;
use std::collections::HashMap;
use std::ops::Neg;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

pub(super) const EVALUATION_REASON: &str = "optimization";
pub(super) const ACCOUNT_TYPE: &str = "OptimizationBudget";
pub(super) const REASON_OPTIMIZATION_CHARGED: &str = "OptimizationCharged";
pub(super) const REASON_OPTIMIZATION_CREATED: &str = "OptimizationCreated";

/// Manages optimization requests, breeding, and budget.
pub struct Service {
    pub(super) host_id: Uuid,
    pub(super) locking: Arc<locking::Service>,
    pub(super) requests_ro: requests::Read,
    pub(super) requests_wr: requests::Write,
    pub(super) genotypes_ro: genotypes::Read,
    pub(super) genotypes_wr: genotypes::Write,
    pub(super) evaluations_ro: evaluations::Read,
    pub(super) optimizers: Arc<Mutex<optimizer::OptimizerRegistry>>,
    pub(super) sync: Arc<synchronization::Service>,
    pub(super) transactions_ro: TransactionsRead,
    pub(super) budgeting: Arc<budgeting::Service>,
    pub(super) evaluation: Arc<evaluation::Service>,
    pub(super) mq: Arc<Queries>,
}

impl Service {
    /// Creates a new optimization service with the given dependencies.
    pub fn new(
        host_id: Uuid,
        requests_ro: requests::Read,
        requests_wr: requests::Write,
        genotypes_ro: genotypes::Read,
        genotypes_wr: genotypes::Write,
        evaluations_ro: evaluations::Read,
        optimizers: Arc<Mutex<optimizer::OptimizerRegistry>>,
        locking: Arc<locking::Service>,
        sync: Arc<synchronization::Service>,
        mq: Arc<Queries>,
        transactions_ro: TransactionsRead,
        budgeting: Arc<budgeting::Service>,
        evaluation: Arc<evaluation::Service>,
    ) -> Self {
        Self {
            host_id,
            requests_ro,
            requests_wr,
            genotypes_ro,
            genotypes_wr,
            evaluations_ro,
            optimizers,
            locking,
            sync,
            mq,
            transactions_ro,
            budgeting,
            evaluation,
        }
    }

    /// Creates a new optimization request with the given parameters.
    #[instrument(level = "debug", skip(self))]
    pub async fn request_new(
        &self,
        type_name: String,
        goal: FitnessGoal,
        schedule: crate::services::optimization::Schedule,
        selector: Selector,
    ) -> Result<Uuid, Error> {
        let has_registered_optimizer = {
            let lock = self.optimizers.lock().await;
            lock.has_registered(&type_name)
        };

        if !has_registered_optimizer {
            return Err(Error::UnknownTypeError { type_name });
        }

        let mq = self.mq.clone();
        let request = db::begin(self.requests_wr.clone(), |tx| {
            Box::pin(async move {
                let request = requests::WriteTx::new(tx)
                    .new_request(Request::new(&type_name, goal, selector, schedule.clone()))
                    .await?;

                let mut publisher = fx_mq_jobs::Publisher::new_tx(tx, &mq);

                // Dispatch a job to add evaluation budget. Once the a first transaction has
                // been made to the account, budgeting will emit an event TransactionCreated.
                // The optimization service listens for TransactionCreated events and dispatches
                // a job for maintain_population once a matching event is handled.
                publisher
                    .publish(&AddOptimizationBudgetMessage::new(
                        request.account_id,
                        schedule.max_evaluations as i64,
                        REASON_OPTIMIZATION_CREATED.to_string(),
                    ))
                    .await?;

                Ok(request)
            })
        })
        .await?;

        Ok(request.id)
    }

    /// Evaluate a genotype
    #[instrument(level = "info", skip(self))]
    pub(super) async fn evaluate_genotype(&self, genotype_id: Uuid) -> Result<(), Error> {
        tracing::info!("Evaluating genotype");

        let genotype = self.genotypes_ro.get_genotype(&genotype_id).await?;
        let request_id = genotype.request_id();
        self.evaluation
            .evaluate_genotype(
                genotype,
                request_id,
                EVALUATION_REASON,
                vec![&request_id.to_string()],
                vec![SHUTDOWN_SEMAPHORE],
            )
            .await?;

        Ok(())
    }

    /// Called after evaluation to figure out if the next generation should be bred.
    #[instrument(level = "info", skip(self))]
    pub(super) async fn should_breed_next_generation(
        &self,
        request_id: Uuid,
        fitness: f64,
    ) -> Result<(), Error> {
        let request = self.requests_ro.get_request(request_id).await?;

        // Raise and return if the goal has been reached
        if request.is_completed(fitness) {
            let sync = self.sync.clone();
            db::begin(self.requests_wr.clone(), |mut tx| {
                Box::pin(async move {
                    sync.raise(&mut tx, &request.id.to_string()).await?;
                    Ok(())
                })
            })
            .await?;
        }

        let genotype_pop = self.genotypes_ro.get_population(&request.id).await?;
        let eval_pop = self
            .evaluations_ro
            .get_evaluation_stats(
                &GetEvaluationStatsFilter::default().with_group_id(request.id),
                i64::MAX,
            )
            .await
            .unwrap_or(evaluations::EvaluationPopulation::empty());

        let live_genotypes = genotype_pop.total_genotypes() - eval_pop.evaluated_genotypes();

        if live_genotypes
            > (request.schedule.population_size - request.schedule.selection_interval) as i64
        {
            // Return early if we're not yet ready to breed.
            // Note that for "generational" schedules, this condition will only be falsy when
            // there are 0 live genotypes.
            return Ok(());
        }

        let balance = self.transactions_ro.balance(&request.account_id).await?;

        let is_exhausted = balance <= 0 && live_genotypes == 0;

        if is_exhausted {
            // Raise a semaphore if the optimization exhausted and evaluations complete.
            // Semaphores are idempotent, raising a semaphore multiple times has no
            // additional effect.
            //
            // Note that no way to explicitly mark an optimization as completed. This
            // allows for optimizations to be resumed.
            let sync = self.sync.clone();
            db::begin(self.requests_wr.clone(), |mut tx| {
                Box::pin(async move {
                    sync.raise(&mut tx, &request.id.to_string()).await?;
                    Ok(())
                })
            })
            .await?;

            return Ok(());
        }

        let filter = TransactionsFilter::default()
            .with_account_id(request.account_id)
            .with_reason(REASON_OPTIMIZATION_CHARGED);

        let expected_last_charge_id = self
            .transactions_ro
            .history(&filter, 1)
            .await?
            .into_iter()
            .next()
            .map(|t| t.id);

        let mq = self.mq.clone();
        db::begin(self.requests_wr.clone(), |tx| {
            Box::pin(async move {
                let mut publisher = fx_mq_jobs::Publisher::new_tx(tx, &mq);
                publisher
                    .publish(&ChargeOptimizationBudgetMessage::new(
                        request.account_id,
                        request.schedule.selection_interval as i64,
                        expected_last_charge_id,
                    ))
                    .await?;
                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    /// Attempts to charge the budget and breed the next generation.
    #[instrument(level = "debug", skip(self))]
    pub(super) async fn try_charge(
        &self,
        account_id: Uuid,
        amount: i64,
        expected_last_charge_id: Option<Uuid>,
    ) -> Result<(), Error> {
        // Advisory lock guarantees synchronous execution of this block
        self.locking
            .lock_while(&format!("try_charge_account:{}", account_id), async || {
                let filter = SearchRequestsFilter::default().with_account_id(account_id);

                let request = self
                    .requests_ro
                    .search_requests(&filter, 1)
                    .await?
                    .into_iter()
                    .next()
                    .ok_or(Error::NoRequestOfAccount(account_id))?;

                let filter = TransactionsFilter::default()
                    .with_account_id(request.account_id)
                    .with_reason(REASON_OPTIMIZATION_CHARGED);

                let actual_last_charge_id = self
                    .transactions_ro
                    .history(&filter, 1)
                    .await?
                    .into_iter()
                    .next()
                    .map(|t| t.id);

                // If the expected last_charge_id does not equal the found id, the charge is no longer valid.
                // We use IDs of charge operations only, not to invalidate charges if budget was added etc,
                // which should have no effect.
                //
                // This logic relies on a few assumptions:
                //  1. This method is the _only_ way to create transactions with reason "OptimizationCharged"
                //  2. This method triggers population maintenance _after_ the transaction has been written.
                //
                // While those assumptions are upheld, this check _should_ ensure that optimizations are charged
                // at most once.
                if expected_last_charge_id != actual_last_charge_id {
                    return Ok::<_, Error>(());
                };

                let balance = self.transactions_ro.balance(&request.account_id).await?;

                let num_offspring = balance.max(0).min(amount);

                if num_offspring == 0 {
                    return Ok(());
                }

                self.budgeting
                    .append(
                        request.account_id,
                        ACCOUNT_TYPE.to_string(),
                        num_offspring.neg(),
                        REASON_OPTIMIZATION_CHARGED.to_string(),
                    )
                    .await?;
                Ok(())
            })
            .await?
    }

    /// Seeds or breeds the next generation of genotypes.
    #[instrument(level = "info", skip(self))]
    pub(super) async fn breed_genotypes(&self, request_id: Uuid, count: i64) -> Result<(), Error> {
        if count <= 0 {
            tracing::error!(
                message = "Breed genotypes called with zero or negative count",
                count = count
            );
            return Err(Error::CouldNotBreed(count));
        }

        let request = self.requests_ro.get_request(request_id).await?;

        let population = self.genotypes_ro.get_population(&request.id).await?;

        if population.current_generation == 0 {
            return self.population_seed(request_id).await;
        }

        self.population_breed(&request, count as usize, population.current_generation + 1)
            .await
    }

    /// Seed a random population
    #[instrument(level = "debug", skip(self))]
    async fn population_seed(&self, request_id: Uuid) -> Result<(), Error> {
        let request = self.requests_ro.get_request(request_id).await?;

        let manager = self.get_optimization_manager(&request.type_name).await?;

        let population_size = request.schedule.population_size as usize;
        let mut genotypes = Vec::with_capacity(population_size);
        let mut jobs = Vec::with_capacity(population_size);
        {
            for _ in 0..population_size {
                let genome = manager.random().map_err(Error::Internal)?;

                let genotype = Genotype::new(
                    &request.type_name,
                    genome,
                    request.id,
                    Some(1), // First generation
                    None,    // No parent_a
                    None,    // No parent_b
                )?;

                jobs.push(EvaluateGenotypeMessage::new(genotype.id()));
                genotypes.push(genotype);
            }
        }

        let mq = self.mq.clone();
        db::begin(self.genotypes_wr.clone(), |tx| {
            Box::pin(async move {
                let inserted = genotypes::WriteTx::new(tx)
                    .store_genotypes(&genotypes)
                    .await?;

                if !inserted.is_empty() {
                    let mut publisher = fx_mq_jobs::Publisher::new_tx(tx, &mq);
                    publisher.publish_many(&jobs).await?;
                }

                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    #[instrument(level = "debug", skip(self, request))]
    async fn population_breed(
        &self,
        request: &Request,
        num_offspring: usize,
        next_generation_id: i32,
    ) -> Result<(), Error> {
        // Only breed if there is no current generation with this ID
        if self
            .genotypes_ro
            .check_if_generation_exists(&request.id, next_generation_id)
            .await?
        {
            tracing::warn!(
                "breed_genotypes was called for a generation that already exists. Generation ID: {}",
                next_generation_id
            );
            return Ok(());
        }

        // Get the GenotypeManager from the service
        let manager = self.get_optimization_manager(&request.type_name).await?;

        // Get candidate genotypes for parent selection
        let population_size = request.schedule.population_size() as i64;

        let (candidates, fitness_map) = if request.schedule.is_generational() {
            let genotypes = self
                .genotypes_ro
                .search_genotypes(
                    &genotypes::SearchGenotypesFilter::default()
                        .with_request_id(request.id)
                        .with_generation_id(next_generation_id - 1)
                        .with_order_random(),
                    population_size,
                )
                .await?;

            let ids: Vec<Uuid> = genotypes.iter().map(|g| g.id()).collect();
            let evals = self
                .evaluations_ro
                .search_evaluations(&SearchEvaluationsFilter::default().with_genotype_ids(ids))
                .await?;
            let fitness_map: HashMap<Uuid, f64> = evals
                .into_iter()
                .map(|e| (*e.genotype_id(), e.fitness()))
                .collect();

            (genotypes, fitness_map)
        } else {
            let evals = self
                .evaluations_ro
                .search_evaluations(
                    &SearchEvaluationsFilter::default()
                        .with_group_ids(vec![request.id])
                        .with_order_completed_at_desc()
                        .with_limit(population_size),
                )
                .await?;

            let ids: Vec<Uuid> = evals.iter().map(|e| *e.genotype_id()).collect();

            let genotypes = self
                .genotypes_ro
                .search_genotypes(
                    &genotypes::SearchGenotypesFilter::default().with_genotype_ids(ids.clone()),
                    population_size,
                )
                .await?;

            let fitness_map: HashMap<Uuid, f64> = evals
                .into_iter()
                .map(|e| (*e.genotype_id(), e.fitness()))
                .collect();

            (genotypes, fitness_map)
        };

        // Build candidates with fitness, error out if any is missing fitness
        let candidates_with_fitness: Vec<(Genotype, f64)> = candidates
            .into_iter()
            .map(|g| {
                let fitness = fitness_map.get(&g.id()).copied().ok_or(Error::NoFitness {
                    genotype_id: g.id(),
                })?;
                Ok((g, fitness))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Pass candidates with fitness to the selector to get pairs of selected parents
        let pairs = request.selector.select_parents(
            num_offspring,
            &candidates_with_fitness,
            &request.goal,
        )?;

        let genotypes =
            Breeder::breed_batch(request, manager.as_ref(), &pairs, next_generation_id)?;

        let jobs: Vec<EvaluateGenotypeMessage> = genotypes
            .iter()
            .map(|g| EvaluateGenotypeMessage::new(g.id()))
            .collect();

        let mq = self.mq.clone();
        db::begin(self.genotypes_wr.clone(), |tx| {
            Box::pin(async move {
                let mut wr_genotypes = genotypes::WriteTx::new(tx);

                wr_genotypes.store_genotypes(&genotypes).await?;

                let mut publisher = fx_mq_jobs::Publisher::new_tx(tx, &mq);
                publisher.publish_many(&jobs).await?;

                Ok(())
            })
        })
        .await?;

        Ok(())
    }

    /// Returns the genotype with the highest (or lowest) fitness for a request.
    #[instrument(level = "debug", skip(self))]
    pub async fn get_best_genotype(
        &self,
        request_id: Uuid,
    ) -> Result<Option<(Genotype, f64)>, Error> {
        let request = self.requests_ro.get_request(request_id).await?;

        let order_filter = match request.goal {
            FitnessGoal::Minimize { .. } => {
                SearchEvaluationsFilter::default().with_order_fitness_asc()
            }
            FitnessGoal::Maximize { .. } => {
                SearchEvaluationsFilter::default().with_order_fitness_desc()
            }
        };

        let mut best_evals = self
            .evaluations_ro
            .search_evaluations(&order_filter.with_group_ids(vec![request_id]).with_limit(1))
            .await?;

        match best_evals.pop() {
            Some(eval) => {
                let genotype = self.genotypes_ro.get_genotype(&eval.genotype_id()).await?;
                Ok(Some((genotype, eval.fitness())))
            }
            None => Ok(None),
        }
    }

    /// Searches genotypes for a request with optional filtering.
    /// Stops the optimization service.
    #[instrument(level = "debug", skip(self))]
    pub async fn stop(&self) -> Result<(), Error> {
        self.sync.stop().await?;
        Ok(())
    }

    async fn get_optimization_manager(
        &self,
        type_name: &str,
    ) -> anyhow::Result<Arc<dyn OptimizerErased>> {
        let lock = self.optimizers.lock().await;
        let manager = lock.get(type_name)?;
        Ok(manager.clone())
    }

    /// Returns the list of registered optimizer type names.
    pub async fn get_registered_type_names(&self) -> Vec<String> {
        let lock = self.optimizers.lock().await;
        lock.get_registered_type_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repositories::genotypes::TypeName;
    use crate::services::optimization as foreign_service;
    use crate::services::optimization::Schedule;
    use crate::test_tools::TestConfig;
    use serde::{Deserialize, Serialize};
    use sqlx::PgPool;
    use std::sync::Arc;

    const TEST_TYPE_NAME: &str = "test::optimization";

    #[derive(Serialize, Deserialize)]
    struct TestGenome;

    struct TestOptimizer;

    impl TypeName for TestOptimizer {
        fn type_name(&self) -> &str {
            TEST_TYPE_NAME
        }
    }

    impl foreign_service::Optimizer for TestOptimizer {
        type Type = TestGenome;

        fn random(&self) -> anyhow::Result<Self::Type> {
            Ok(TestGenome)
        }

        fn crossover(
            &self,
            _parent1: Self::Type,
            _parent2: Self::Type,
        ) -> anyhow::Result<Self::Type> {
            Ok(TestGenome)
        }

        fn mutate(&self, _instance: &mut Self::Type) -> anyhow::Result<()> {
            Ok(())
        }
    }

    async fn setup(pool: PgPool) -> anyhow::Result<Arc<Service>> {
        let mut c = TestConfig::new(pool)
            .with_optimizer(TestOptimizer)
            .build()
            .await?;
        let svc = c.get::<Arc<Service>>().await?;
        Ok(svc)
    }

    #[sqlx::test(migrations = false)]
    async fn get_registered_type_names_returns_registered_types(
        pool: PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = setup(pool.clone()).await?;
        let names = svc.get_registered_type_names().await;

        assert!(names.contains(&TEST_TYPE_NAME.to_string()));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn get_registered_type_names_returns_empty_when_no_optimizers(
        pool: PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let mut c = TestConfig::new(pool).build().await?;
        let svc = c.get::<Arc<Service>>().await?;
        let names = svc.get_registered_type_names().await;

        assert!(names.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn request_new_returns_error_for_unknown_type(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = setup(pool.clone()).await?;

        let result = svc
            .request_new(
                "unknown::type".to_string(),
                FitnessGoal::maximize(1.0)?,
                Schedule::generational(10, 2),
                Selector::tournament(3),
            )
            .await;

        assert!(matches!(result, Err(Error::UnknownTypeError { .. })));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn request_new_creates_request(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = setup(pool.clone()).await?;

        let request_id = svc
            .request_new(
                TEST_TYPE_NAME.to_string(),
                FitnessGoal::maximize(1.0)?,
                Schedule::generational(10, 2),
                Selector::tournament(3),
            )
            .await?;

        // Verify the request exists via the read repository
        let request = svc.requests_ro.get_request(request_id).await?;
        assert_eq!(request.type_name, TEST_TYPE_NAME);

        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test_tools {
    use super::*;
    use crate::repositories::genotypes::{Genotype, TypeName, store_genotypes};
    use crate::services::budgeting;
    use crate::services::evaluation::repositories::evaluations;
    use crate::services::evaluation::{Evaluator, Service as EvaluationService};
    use crate::services::optimization as foreign_service;
    use crate::services::optimization::Schedule;
    use crate::test_tools::TestConfig;
    use chrono::Utc;
    use futures::future::BoxFuture;
    use serde::{Deserialize, Serialize};
    use sqlx::PgPool;
    use std::sync::Arc;
    use uuid::Uuid;

    pub(crate) const TEST_TYPE_NAME: &str = "test::optimization";
    pub(crate) const POPULATION_SIZE: u32 = 5;
    pub(crate) const SELECTION_INTERVAL: u32 = 4;

    #[derive(Serialize, Deserialize)]
    pub(crate) struct TestGenome;

    pub(crate) struct TestOptimizer;

    impl TypeName for TestOptimizer {
        fn type_name(&self) -> &str {
            TEST_TYPE_NAME
        }
    }

    impl foreign_service::Optimizer for TestOptimizer {
        type Type = TestGenome;

        fn random(&self) -> anyhow::Result<Self::Type> {
            Ok(TestGenome)
        }

        fn crossover(
            &self,
            _parent1: Self::Type,
            _parent2: Self::Type,
        ) -> anyhow::Result<Self::Type> {
            Ok(TestGenome)
        }

        fn mutate(&self, _instance: &mut Self::Type) -> anyhow::Result<()> {
            Ok(())
        }
    }

    pub(crate) struct TestEvaluator;

    impl Evaluator for TestEvaluator {
        type Type = serde_json::Value;

        fn evaluate<'a>(
            &'a self,
            _instance: &'a Self::Type,
        ) -> BoxFuture<'a, Result<f64, Box<dyn std::error::Error + Send + Sync>>> {
            Box::pin(async move { Ok(0.5) })
        }
    }

    pub(crate) struct SeedData {
        pub(crate) svc: Arc<Service>,
        pub(crate) request_id: Uuid,
        pub(crate) account_id: Uuid,
    }

    pub(crate) async fn seed(pool: &PgPool) -> anyhow::Result<SeedData> {
        let svc = build_service(pool.clone()).await?;

        let max_evaluations = POPULATION_SIZE * 10;
        let schedule = Schedule::rolling(max_evaluations, POPULATION_SIZE, SELECTION_INTERVAL);
        let request_id = svc
            .request_new(
                TEST_TYPE_NAME.to_string(),
                FitnessGoal::maximize(1.0)?,
                schedule,
                Selector::tournament(3),
            )
            .await?;

        let request = svc.requests_ro.get_request(request_id).await?;
        let account_id = request.account_id;

        seed_budget(pool, account_id).await?;

        let genotypes: Vec<Genotype> = (0..POPULATION_SIZE as usize)
            .map(|i| {
                Genotype::new(
                    TEST_TYPE_NAME,
                    serde_json::json!({}),
                    request_id,
                    Some(1),
                    None,
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let stored = store_genotypes(pool, &genotypes).await?;

        let now = Utc::now();
        let evals: Vec<evaluations::Evaluation> = stored[..SELECTION_INTERVAL as usize]
            .iter()
            .map(|g| {
                evaluations::Evaluation::new(
                    g.id(),
                    request_id,
                    "optimization".to_string(),
                    0.5,
                    Some(now),
                    Some(now),
                    Some(Uuid::nil()),
                )
            })
            .collect();
        evaluations::queries::store_evaluations(pool, &evals).await?;

        Ok(SeedData {
            svc,
            request_id,
            account_id,
        })
    }

    async fn seed_budget(pool: &PgPool, account_id: Uuid) -> anyhow::Result<()> {
        let mut c = TestConfig::new(pool.clone())
            .with_optimizer(TestOptimizer)
            .build()
            .await?;
        let budgeting = c.get::<Arc<budgeting::Service>>().await?;
        budgeting
            .append(
                account_id,
                super::ACCOUNT_TYPE.to_string(),
                100,
                super::REASON_OPTIMIZATION_CREATED.to_string(),
            )
            .await?;
        Ok(())
    }

    pub(crate) async fn build_service(pool: PgPool) -> anyhow::Result<Arc<Service>> {
        let mut c = TestConfig::new(pool)
            .with_optimizer(TestOptimizer)
            .build()
            .await?;
        let evaluation = c.get::<Arc<EvaluationService>>().await?;
        evaluation.register(TEST_TYPE_NAME, TestEvaluator).await;
        let svc = c.get::<Arc<Service>>().await?;
        Ok(svc)
    }

    pub(crate) async fn seed_genotypes(
        pool: &PgPool,
        request_id: Uuid,
        count: usize,
    ) -> anyhow::Result<Vec<Genotype>> {
        let genotypes: Vec<Genotype> = (0..count)
            .map(|i| {
                Genotype::new(
                    TEST_TYPE_NAME,
                    serde_json::json!({}),
                    request_id,
                    Some(1),
                    None,
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let stored = store_genotypes(pool, &genotypes).await?;
        Ok(stored)
    }
}

#[cfg(test)]
mod tests_evaluate_genotype {
    use super::test_tools::*;
    use sqlx::PgPool;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn evaluate_genotype_errors_on_missing_genotype(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        seed(&pool).await?;

        let missing_id = Uuid::parse_str("00000000-0000-0000-0000-000000000099")?;
        let result = svc.evaluate_genotype(missing_id).await;

        assert!(result.is_err());
        Ok(())
    }
}

#[cfg(test)]
mod tests_should_breed_next_generation {
    use super::test_tools::*;
    use crate::services::optimization::jobs::ChargeOptimizationBudgetMessage;
    use fx_mq_building_blocks::testing_tools::TestQueries;
    use fx_mq_jobs::{FX_MQ_JOBS_SCHEMA_NAME, Message};
    use sqlx::PgPool;

    #[sqlx::test(migrations = false)]
    async fn it_skips_breeding_when_not_ready(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        // Add 1 extra genotype without evaluation so live > threshold
        seed_genotypes(&pool, data.request_id, 1).await?;

        data.svc
            .should_breed_next_generation(data.request_id, 0.1)
            .await?;

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let charge_jobs: Vec<_> = mq
            .get_all_messages(&mut tx)
            .await?
            .into_iter()
            .filter(|j| j.name == ChargeOptimizationBudgetMessage::NAME)
            .collect();
        tx.commit().await?;
        assert!(charge_jobs.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_dispatches_charge_job_when_ready(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        data.svc
            .should_breed_next_generation(data.request_id, 0.1)
            .await?;

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let charge_jobs: Vec<_> = mq
            .get_all_messages(&mut tx)
            .await?
            .into_iter()
            .filter(|j| j.name == ChargeOptimizationBudgetMessage::NAME)
            .collect();
        tx.commit().await?;
        assert_eq!(charge_jobs.len(), 1);

        Ok(())
    }
}

#[cfg(test)]
mod tests_try_charge {
    use super::test_tools::*;
    use super::*;
    use crate::services::budgeting::TransactionsFilter;
    use crate::services::optimization::Schedule;
    use crate::test_tools::TestConfig;
    use sqlx::PgPool;
    use std::sync::Arc;
    use uuid::Uuid;

    async fn count_charges(svc: &Service, account_id: Uuid) -> anyhow::Result<usize> {
        let filter = TransactionsFilter::default().with_account_id(account_id);
        let txs = svc.transactions_ro.history(&filter, i64::MAX).await?;
        Ok(txs
            .iter()
            .filter(|t| t.reason() == super::REASON_OPTIMIZATION_CHARGED)
            .count())
    }

    async fn seed_budget_amount(
        pool: &PgPool,
        account_id: Uuid,
        amount: i64,
    ) -> anyhow::Result<Arc<crate::services::budgeting::Service>> {
        let mut c = TestConfig::new(pool.clone())
            .with_optimizer(TestOptimizer)
            .build()
            .await?;
        let budgeting = c.get::<Arc<crate::services::budgeting::Service>>().await?;
        budgeting
            .append(
                account_id,
                super::ACCOUNT_TYPE.to_string(),
                amount,
                super::REASON_OPTIMIZATION_CREATED.to_string(),
            )
            .await?;
        Ok(budgeting)
    }

    #[sqlx::test(migrations = false)]
    async fn try_charge_partial_when_balance_lower_than_amount(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let schedule = Schedule::rolling(POPULATION_SIZE * 10, POPULATION_SIZE, SELECTION_INTERVAL);
        let request_id = svc
            .request_new(
                TEST_TYPE_NAME.to_string(),
                FitnessGoal::maximize(1.0)?,
                schedule,
                Selector::tournament(3),
            )
            .await?;
        let request = svc.requests_ro.get_request(request_id).await?;

        // Seed a budget of 2, lower than the charge amount (SELECTION_INTERVAL = 4)
        seed_budget_amount(&pool, request.account_id, 2).await?;

        svc.try_charge(request.account_id, SELECTION_INTERVAL as i64, None)
            .await?;

        let txs = svc
            .transactions_ro
            .history(
                &TransactionsFilter::default().with_account_id(request.account_id),
                i64::MAX,
            )
            .await?;
        let charges: Vec<_> = txs
            .iter()
            .filter(|t| t.reason() == super::REASON_OPTIMIZATION_CHARGED)
            .collect();
        assert_eq!(charges.len(), 1);
        assert_eq!(charges[0].amount(), -2);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn try_charge_skips_when_idempotent(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let data = seed(&pool).await?;

        // First call with expected=None — no prior charges, so actual=None → match → charges
        data.svc
            .try_charge(data.account_id, SELECTION_INTERVAL as i64, None)
            .await?;

        assert_eq!(count_charges(&data.svc, data.account_id).await?, 1);

        // Second call with expected=None again — simulates the same job being replayed.
        // Now actual=Some(first_charge_id) ≠ None → skip.
        data.svc
            .try_charge(data.account_id, SELECTION_INTERVAL as i64, None)
            .await?;

        let all_txs = data
            .svc
            .transactions_ro
            .history(
                &TransactionsFilter::default().with_account_id(data.account_id),
                i64::MAX,
            )
            .await?;
        let charge_txs: Vec<_> = all_txs
            .iter()
            .filter(|t| t.reason() == super::REASON_OPTIMIZATION_CHARGED)
            .collect();
        assert_eq!(charge_txs.len(), 1);
        assert_eq!(charge_txs[0].amount(), -(SELECTION_INTERVAL as i64));

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn try_charge_skips_when_insufficient_balance(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // No with_listeners() → AddOptimizationBudgetMessage dispatched by
        // request_new is never processed, so the account balance stays at 0.
        let svc = build_service(pool.clone()).await?;
        let max_evaluations = POPULATION_SIZE * 10;
        let schedule = Schedule::rolling(max_evaluations, POPULATION_SIZE, SELECTION_INTERVAL);
        let request_id = svc
            .request_new(
                TEST_TYPE_NAME.to_string(),
                FitnessGoal::maximize(1.0)?,
                schedule,
                Selector::tournament(3),
            )
            .await?;
        let request = svc.requests_ro.get_request(request_id).await?;

        svc.try_charge(request.account_id, SELECTION_INTERVAL as i64, None)
            .await?;

        let count = count_charges(&svc, request.account_id).await?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn try_charge_errors_for_unknown_account(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let unknown_id = Uuid::parse_str("00000000-0000-0000-0000-000000000099")?;

        let result = svc
            .try_charge(unknown_id, SELECTION_INTERVAL as i64, None)
            .await;

        assert!(matches!(result, Err(Error::NoRequestOfAccount(_))));

        Ok(())
    }
}

#[cfg(test)]
mod tests_breed_genotypes {
    use super::test_tools::*;
    use super::*;
    use crate::services::evaluation::repositories::evaluations;
    use crate::services::optimization::Schedule;
    use crate::services::optimization::jobs::EvaluateGenotypeMessage;
    use chrono::Utc;
    use fx_mq_building_blocks::testing_tools::TestQueries;
    use fx_mq_jobs::{FX_MQ_JOBS_SCHEMA_NAME, Message};
    use sqlx::PgPool;
    use uuid::Uuid;

    /// Tournament(2) needs ≥4 candidates. POPULATION_SIZE = 5, so all tests
    /// that breed from existing genotypes have enough evaluated parents.
    async fn make_request(svc: &Service) -> Result<Uuid, Error> {
        let schedule = Schedule::rolling(POPULATION_SIZE * 10, POPULATION_SIZE, SELECTION_INTERVAL);
        let goal = FitnessGoal::maximize(1.0).expect("1.0 is a valid threshold");
        svc.request_new(
            TEST_TYPE_NAME.to_string(),
            goal,
            schedule,
            Selector::tournament(2),
        )
        .await
    }

    async fn evaluate_all(svc: &Service, pool: &PgPool, request_id: Uuid) -> anyhow::Result<()> {
        let pop = svc.genotypes_ro.get_population(&request_id).await?;
        let genotypes = svc
            .genotypes_ro
            .search_genotypes(
                &crate::SearchGenotypesFilter::default()
                    .with_request_id(request_id)
                    .with_generation_id(pop.current_generation()),
                pop.total_genotypes(),
            )
            .await?;
        let now = Utc::now();
        let evals: Vec<evaluations::Evaluation> = genotypes
            .iter()
            .map(|g| {
                evaluations::Evaluation::new(
                    g.id(),
                    request_id,
                    "optimization".to_string(),
                    0.5,
                    Some(now),
                    Some(now),
                    Some(Uuid::nil()),
                )
            })
            .collect();
        evaluations::queries::store_evaluations(pool, &evals).await?;
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_seeds_population_when_generation_zero(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let request_id = make_request(&svc).await?;

        let pop_before = svc.genotypes_ro.get_population(&request_id).await?;
        assert_eq!(pop_before.current_generation(), 0);
        assert_eq!(pop_before.total_genotypes(), 0);

        svc.breed_genotypes(request_id, POPULATION_SIZE as i64)
            .await?;

        let pop_after = svc.genotypes_ro.get_population(&request_id).await?;
        assert_eq!(pop_after.total_genotypes(), POPULATION_SIZE as i64);
        assert_eq!(pop_after.current_generation(), 1);

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let jobs = mq.get_all_messages(&mut tx).await?;
        tx.commit().await?;
        let eval_jobs: Vec<_> = jobs
            .iter()
            .filter(|j| j.name == EvaluateGenotypeMessage::NAME)
            .collect();
        assert_eq!(eval_jobs.len(), POPULATION_SIZE as usize);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_breeds_next_generation(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let request_id = make_request(&svc).await?;

        // First breed: seed the initial population
        svc.breed_genotypes(request_id, POPULATION_SIZE as i64)
            .await?;

        // Evaluate all seeded genotypes so breed can select parents
        evaluate_all(&svc, &pool, request_id).await?;

        // Second breed: create next generation
        let count = 3;
        svc.breed_genotypes(request_id, count).await?;

        let pop = svc.genotypes_ro.get_population(&request_id).await?;
        assert_eq!(pop.total_genotypes(), POPULATION_SIZE as i64 + count);
        assert_eq!(pop.current_generation(), 2);

        let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
        let mut tx = pool.begin().await?;
        let jobs = mq.get_all_messages(&mut tx).await?;
        tx.commit().await?;
        let eval_jobs: Vec<_> = jobs
            .iter()
            .filter(|j| j.name == EvaluateGenotypeMessage::NAME)
            .collect();
        assert_eq!(eval_jobs.len(), (POPULATION_SIZE + count as u32) as usize);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_invalid_count(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let request_id = make_request(&svc).await?;

        let result = svc.breed_genotypes(request_id, 0).await;
        assert!(matches!(result, Err(Error::CouldNotBreed(_))));

        let result = svc.breed_genotypes(request_id, -1).await;
        assert!(matches!(result, Err(Error::CouldNotBreed(_))));

        Ok(())
    }
}

#[cfg(test)]
mod tests_get_best_genotype {
    use super::test_tools::*;
    use super::*;
    use crate::services::evaluation::repositories::evaluations;
    use crate::services::optimization::Schedule;
    use chrono::Utc;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed_data(pool: &PgPool, goal: FitnessGoal) -> anyhow::Result<(Arc<Service>, Uuid)> {
        let svc = build_service(pool.clone()).await?;
        let schedule = Schedule::rolling(100, 10, 2);
        let request_id = svc
            .request_new(
                TEST_TYPE_NAME.to_string(),
                goal,
                schedule,
                Selector::tournament(2),
            )
            .await?;

        let genotypes = super::test_tools::seed_genotypes(pool, request_id, 3).await?;
        let now = Utc::now();
        let fitnesses = [0.1, 0.5, 0.9];
        let evals: Vec<evaluations::Evaluation> = genotypes
            .iter()
            .zip(fitnesses.iter())
            .map(|(g, &f)| {
                evaluations::Evaluation::new(
                    g.id(),
                    request_id,
                    "optimization".to_string(),
                    f,
                    Some(now),
                    Some(now),
                    Some(Uuid::nil()),
                )
            })
            .collect();
        evaluations::queries::store_evaluations(pool, &evals).await?;

        Ok((svc, request_id))
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_best_for_maximize(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (svc, request_id) =
            seed_data(&pool, FitnessGoal::maximize(1.0).expect("valid threshold")).await?;

        let result = svc.get_best_genotype(request_id).await?;

        let (genotype, fitness) = result.expect("expected a best genotype");
        assert!((fitness - 0.9).abs() < 1e-12);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_best_for_minimize(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let (svc, request_id) =
            seed_data(&pool, FitnessGoal::minimize(1.0).expect("valid threshold")).await?;

        let result = svc.get_best_genotype(request_id).await?;

        let (genotype, fitness) = result.expect("expected a best genotype");
        assert!((fitness - 0.1).abs() < 1e-12);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_none_when_no_evals(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let schedule = Schedule::rolling(100, 10, 2);
        let goal = FitnessGoal::maximize(1.0).expect("valid threshold");
        let request_id = svc
            .request_new(
                TEST_TYPE_NAME.to_string(),
                goal,
                schedule,
                Selector::tournament(2),
            )
            .await?;

        let result = svc.get_best_genotype(request_id).await?;
        assert!(result.is_none());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_errors_on_unknown_request(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let missing_id = Uuid::parse_str("00000000-0000-0000-0000-000000000099")?;

        let result = svc.get_best_genotype(missing_id).await;
        assert!(result.is_err());

        Ok(())
    }
}

#[cfg(test)]
mod tests_stop {
    use super::test_tools::*;
    use super::*;
    use sqlx::PgPool;

    #[sqlx::test(migrations = false)]
    async fn it_returns_ok(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        let svc = build_service(pool.clone()).await?;
        let result = svc.stop().await;
        assert!(result.is_ok());

        Ok(())
    }
}
