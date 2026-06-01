use super::Breeder;
use super::Error;
use super::optimizer::{self, OptimizerErased};
use crate::infrastructure::db;
use crate::repositories::genotypes;
use crate::repositories::genotypes::Genotype;
use crate::services::evaluation::repositories::evaluations;
use crate::services::evaluation::{
    GetEvaluationStatsFilter, SearchEvaluationsFilter,
};
use crate::services::evaluation::SHUTDOWN_SEMAPHORE;
use crate::services::budgeting::{TransactionsFilter, TransactionsRead};
use crate::services::optimization::jobs::{
    AddOptimizationBudgetMessage, ChargeOptimizationBudgetMessage, EvaluateGenotypeMessage,
};
use crate::services::optimization::repositories::requests;
use crate::services::optimization::repositories::requests::SearchRequestsFilter;
use crate::services::optimization::{FitnessGoal, Request, Selector};
use crate::services::{budgeting, locking};
use crate::services::{evaluation, synchronization};
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::lock::Mutex;
use fx_mq_jobs::Queries;
use std::collections::HashMap;
use std::ops::Neg;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

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
        let type_hash = fnv1a_hash_str_32(&type_name) as i32;

        let has_registered_optimizer = {
            let lock = self.optimizers.lock().await;
            lock.has_registered(&type_name)
        };

        if !has_registered_optimizer {
            return Err(Error::UnknownTypeError {
                type_hash,
                type_name,
            });
        }

        let mq = self.mq.clone();
        let request = db::begin(self.requests_wr.clone(), |tx| {
            Box::pin(async move {
                let request = requests::WriteTx::new(tx)
                    .new_request(Request::new(
                        &type_name,
                        type_hash,
                        goal,
                        selector,
                        schedule.clone(),
                    ))
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
    pub(super) async fn evaluate_genotype(
        &self,
        request_id: Uuid,
        genotype_id: Uuid,
    ) -> Result<(), Error> {
        tracing::info!("Evaluating genotype");

        let genotype = self.genotypes_ro.get_genotype(&genotype_id).await?;

        self.evaluation
            .evaluate_genotype(genotype, vec![&request_id.to_string()], vec![SHUTDOWN_SEMAPHORE])
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
                &GetEvaluationStatsFilter::default().with_request_id(request.id),
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
                let genome = manager
                    .random()
                    .map_err(Error::Internal)?;

                let genotype = Genotype::new(
                    &request.type_name,
                    request.type_hash,
                    genome,
                    Some(request.id),
                    Some(1), // First generation
                    None,    // No parent_a
                    None,    // No parent_b
                )?;

                jobs.push(EvaluateGenotypeMessage::new(request.id, genotype.id()));
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
                .search_evaluations(
                    &SearchEvaluationsFilter::default().with_genotype_ids(ids),
                )
                .await?;
            let fitness_map: HashMap<Uuid, f64> =
                evals.into_iter().map(|e| (*e.genotype_id(), e.fitness())).collect();

            (genotypes, fitness_map)
        } else {
            let evals = self
                .evaluations_ro
                .search_evaluations(
                    &SearchEvaluationsFilter::default()
                        .with_request_ids(vec![request.id])
                        .with_order_completed_at_desc()
                        .with_limit(population_size),
                )
                .await?;

            let ids: Vec<Uuid> = evals.iter().map(|e| *e.genotype_id()).collect();

            let genotypes = self
                .genotypes_ro
                .search_genotypes(
                    &genotypes::SearchGenotypesFilter::default()
                        .with_genotype_ids(ids.clone()),
                    population_size,
                )
                .await?;

            let fitness_map: HashMap<Uuid, f64> =
                evals.into_iter().map(|e| (*e.genotype_id(), e.fitness())).collect();

            (genotypes, fitness_map)
        };

        // Build candidates with fitness, error out if any is missing fitness
        let candidates_with_fitness: Vec<(Genotype, f64)> = candidates
            .into_iter()
            .map(|g| {
                let fitness = fitness_map
                    .get(&g.id())
                    .copied()
                    .ok_or(Error::NoFitness { genotype_id: g.id() })?;
                Ok((g, fitness))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Pass candidates with fitness to the selector to get pairs of selected parents
        let pairs = request.selector.select_parents(
            num_offspring,
            &candidates_with_fitness,
            &request.goal,
        )?;

        let genotypes = Breeder::breed_batch(
            request,
            manager.as_ref(),
            &pairs,
            next_generation_id,
        )?;

        let jobs: Vec<EvaluateGenotypeMessage> = genotypes
            .iter()
            .map(|g| EvaluateGenotypeMessage::new(request.id, g.id()))
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
            FitnessGoal::Minimize { .. } => SearchEvaluationsFilter::default().with_order_fitness_asc(),
            FitnessGoal::Maximize { .. } => SearchEvaluationsFilter::default().with_order_fitness_desc(),
        };

        let mut best_evals = self
            .evaluations_ro
            .search_evaluations(
                &order_filter
                    .with_request_ids(vec![request_id])
                    .with_limit(1),
            )
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
