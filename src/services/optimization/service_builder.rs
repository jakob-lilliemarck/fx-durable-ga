use super::models::TypeErasedEvaluator;
use crate::{
    models::{Encodeable, Evaluator, Morphology},
    repositories::{genotypes, morphologies, requests},
    services::{lock, optimization::Service},
};
use std::collections::HashMap;
use tracing::instrument;

/// Builder for creating optimization services with registered type evaluators.
/// Handles morphology registration and evaluator type erasure.
pub struct ServiceBuilder {
    pub(super) locking: lock::Service,
    pub(super) requests: requests::Repository,
    pub(super) morphologies: morphologies::Repository,
    pub(super) genotypes: genotypes::Repository,
    pub(super) evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
    pub(super) max_deduplication_attempts: i32,
}

impl ServiceBuilder {
    /// Registers a problem type and its fitness evaluator with the optimization service.
    ///
    /// This is the core method for connecting your domain-specific problems to the genetic
    /// algorithm engine. It establishes the relationship between your data structures,
    /// their genetic encoding, and the fitness evaluation logic.
    ///
    /// # Core Concepts
    ///
    /// ## Type Registration Process
    /// 1. **Morphology Creation**: Automatically creates and stores the search space definition
    /// 2. **Type Erasure**: Converts your specific evaluator into a type-erased form for internal storage  
    /// 3. **Hash Mapping**: Associates your type's hash with its evaluator for efficient lookup
    /// 4. **Database Integration**: Ensures the type's morphology exists in persistent storage
    ///
    /// ## Required Trait Implementations
    /// - **`Encodeable`**: Defines how your type maps to/from genetic representation
    /// - **`Evaluator`**: Provides the fitness function that guides evolution
    ///
    /// # Parameters
    ///
    /// * `evaluator` - Your fitness evaluator implementing `Evaluator<T::Phenotype>`
    ///
    /// # Type Parameters
    ///
    /// * `T: Encodeable` - Your problem type that can be encoded as genes
    /// * `E: Evaluator<T::Phenotype>` - Your fitness evaluator for decoded phenotypes
    ///
    /// # Encodeable Implementation Requirements
    ///
    /// Your type must implement `Encodeable` with:
    /// - `NAME`: Unique string identifier for your type
    /// - `Phenotype`: The decoded form passed to evaluators
    /// - `morphology()`: Defines the search space bounds and structure
    /// - `encode()`: Converts instances to genetic representation
    /// - `decode()`: Converts genes back to phenotypes for evaluation
    ///
    /// # Evaluator Implementation Requirements  
    ///
    /// Your evaluator must implement `Evaluator<T::Phenotype>` with:
    /// - `fitness()`: Async function returning fitness scores (0.0-1.0 recommended)
    /// - Support for early termination via the `Terminated` trait
    /// - Error handling for failed evaluations
    ///
    /// # Common Registration Patterns
    ///
    /// ## Simple Numerical Optimization
    /// ```rust,no_run
    /// use fx_durable_ga::models::*;
    /// # use fx_durable_ga::services::optimization::ServiceBuilder;
    /// # use futures::future::BoxFuture;
    ///
    /// #[derive(Debug, Clone)]
    /// struct Parameters {
    ///     learning_rate: f64,
    ///     batch_size: i64,
    ///     layers: i64,
    /// }
    ///
    /// impl Encodeable for Parameters {
    ///     const NAME: &'static str = "Parameters";
    ///     type Phenotype = Parameters;
    ///
    ///     fn morphology() -> Vec<GeneBounds> {
    ///         vec![
    ///             GeneBounds::decimal(0.001, 1.0, 1000, 3).unwrap(),  // learning_rate
    ///             GeneBounds::integer(8, 512, 32).unwrap(),            // batch_size  
    ///             GeneBounds::integer(1, 10, 10).unwrap(),             // layers
    ///         ]
    ///     }
    ///
    ///     fn encode(&self) -> Vec<i64> {
    ///         let bounds = Self::morphology();
    ///         vec![
    ///             bounds[0].from_sample(self.learning_rate / 1.0),
    ///             bounds[1].from_sample((self.batch_size - 8) as f64 / (512 - 8) as f64),
    ///             bounds[2].from_sample((self.layers - 1) as f64 / 9.0),
    ///         ]
    ///     }
    ///
    ///     fn decode(genes: &[i64]) -> Self::Phenotype {
    ///         let bounds = Self::morphology();
    ///         Parameters {
    ///             learning_rate: bounds[0].to_f64(genes[0]),
    ///             batch_size: (bounds[1].to_f64(genes[1]) * (512 - 8) as f64 + 8.0) as i64,
    ///             layers: (bounds[2].to_f64(genes[2]) * 9.0 + 1.0) as i64,
    ///         }
    ///     }
    /// }
    ///
    /// struct ModelEvaluator;
    ///
    /// impl Evaluator<Parameters> for ModelEvaluator {
    ///     fn fitness<'a>(
    ///         &self,
    ///         params: Parameters,
    ///         _terminated: &'a Box<dyn Terminated>,
    ///     ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
    ///         Box::pin(async move {
    ///             // Train model with parameters and return accuracy
    ///             let accuracy = train_model(params.learning_rate, params.batch_size, params.layers).await?;
    ///             Ok(accuracy)
    ///         })
    ///     }
    /// }
    ///
    /// # async fn train_model(_lr: f64, _batch: i64, _layers: i64) -> Result<f64, anyhow::Error> { Ok(0.85) }
    /// # async fn example(builder: ServiceBuilder) -> Result<ServiceBuilder, Box<dyn std::error::Error>> {
    /// let builder = builder.register::<Parameters, _>(ModelEvaluator).await?;
    /// # Ok(builder)
    /// # }
    /// ```
    ///
    /// ## Geometric/Spatial Problems  
    /// ```rust,no_run
    /// use fx_durable_ga::models::*;
    /// # use fx_durable_ga::services::optimization::ServiceBuilder;
    /// # use futures::future::BoxFuture;
    ///
    /// #[derive(Debug, Clone)]
    /// struct Point3D {
    ///     x: f64,
    ///     y: f64, 
    ///     z: f64,
    /// }
    ///
    /// impl Encodeable for Point3D {
    ///     const NAME: &'static str = "Point3D";
    ///     type Phenotype = Point3D;
    ///
    ///     fn morphology() -> Vec<GeneBounds> {
    ///         vec![
    ///             GeneBounds::decimal(-10.0, 10.0, 2000, 3).unwrap(),  // x: -10.0 to 10.0
    ///             GeneBounds::decimal(-10.0, 10.0, 2000, 3).unwrap(),  // y: -10.0 to 10.0
    ///             GeneBounds::decimal(0.0, 20.0, 2000, 3).unwrap(),    // z: 0.0 to 20.0
    ///         ]
    ///     }
    ///
    ///     fn encode(&self) -> Vec<i64> {
    ///         let bounds = Self::morphology();
    ///         vec![
    ///             bounds[0].from_sample((self.x + 10.0) / 20.0),
    ///             bounds[1].from_sample((self.y + 10.0) / 20.0),
    ///             bounds[2].from_sample(self.z / 20.0),
    ///         ]
    ///     }
    ///
    ///     fn decode(genes: &[i64]) -> Self::Phenotype {
    ///         let bounds = Self::morphology();
    ///         Point3D {
    ///             x: bounds[0].to_f64(genes[0]) * 20.0 - 10.0,
    ///             y: bounds[1].to_f64(genes[1]) * 20.0 - 10.0,
    ///             z: bounds[2].to_f64(genes[2]) * 20.0,
    ///         }
    ///     }
    /// }
    ///
    /// struct DistanceEvaluator {
    ///     target: Point3D,
    /// }
    ///
    /// impl Evaluator<Point3D> for DistanceEvaluator {
    ///     fn fitness<'a>(
    ///         &self,
    ///         point: Point3D,
    ///         terminated: &'a Box<dyn Terminated>,
    ///     ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
    ///         let target = self.target.clone();
    ///         Box::pin(async move {
    ///             // Check for early termination
    ///             if terminated.is_terminated().await {
    ///                 return Err(anyhow::anyhow!("Evaluation terminated"));
    ///             }
    ///
    ///             let dx = point.x - target.x;
    ///             let dy = point.y - target.y;
    ///             let dz = point.z - target.z;
    ///             let distance = (dx*dx + dy*dy + dz*dz).sqrt();
    ///             
    ///             // Convert distance to fitness (closer = higher fitness)
    ///             let fitness = 1.0 / (1.0 + distance);
    ///             Ok(fitness)
    ///         })
    ///     }
    /// }
    ///
    /// # async fn example(builder: ServiceBuilder) -> Result<ServiceBuilder, Box<dyn std::error::Error>> {
    /// let evaluator = DistanceEvaluator {
    ///     target: Point3D { x: 5.0, y: -2.0, z: 10.0 }
    /// };
    /// let builder = builder.register::<Point3D, _>(evaluator).await?;
    /// # Ok(builder)
    /// # }
    /// ```
    ///
    /// ## Combinatorial/Discrete Problems
    /// ```rust,no_run
    /// use fx_durable_ga::models::*;
    /// # use fx_durable_ga::services::optimization::ServiceBuilder;
    /// # use futures::future::BoxFuture;
    ///
    /// #[derive(Debug, Clone)]
    /// struct Schedule {
    ///     assignments: Vec<i64>,  // job assignments to workers
    /// }
    ///
    /// impl Encodeable for Schedule {
    ///     const NAME: &'static str = "Schedule";
    ///     type Phenotype = Schedule;
    ///
    ///     fn morphology() -> Vec<GeneBounds> {
    ///         // 10 jobs, each can be assigned to 5 workers (0-4)
    ///         (0..10)
    ///             .map(|_| GeneBounds::integer(0, 4, 5).unwrap())
    ///             .collect()
    ///     }
    ///
    ///     fn encode(&self) -> Vec<i64> {
    ///         self.assignments.clone()
    ///     }
    ///
    ///     fn decode(genes: &[i64]) -> Self::Phenotype {
    ///         Schedule {
    ///             assignments: genes.to_vec()
    ///         }
    ///     }
    /// }
    ///
    /// struct ScheduleEvaluator {
    ///     job_costs: Vec<Vec<f64>>,  // job_costs[job][worker] = cost
    /// }
    ///
    /// impl Evaluator<Schedule> for ScheduleEvaluator {
    ///     fn fitness<'a>(
    ///         &self,
    ///         schedule: Schedule,
    ///         _terminated: &'a Box<dyn Terminated>,
    ///     ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
    ///         let costs = self.job_costs.clone();
    ///         Box::pin(async move {
    ///             let total_cost: f64 = schedule.assignments
    ///                 .iter()
    ///                 .enumerate()
    ///                 .map(|(job, &worker)| costs[job][worker as usize])
    ///                 .sum();
    ///             
    ///             // Lower cost = higher fitness
    ///             let max_possible_cost = 1000.0;
    ///             let fitness = (max_possible_cost - total_cost) / max_possible_cost;
    ///             Ok(fitness.max(0.0))
    ///         })
    ///     }
    /// }
    ///
    /// # async fn example(builder: ServiceBuilder) -> Result<ServiceBuilder, Box<dyn std::error::Error>> {
    /// let evaluator = ScheduleEvaluator {
    ///     job_costs: vec![vec![10.0, 20.0, 15.0, 25.0, 12.0]; 10]
    /// };
    /// let builder = builder.register::<Schedule, _>(evaluator).await?;
    /// # Ok(builder)
    /// # }
    /// ```
    ///
    /// # Best Practices
    ///
    /// ## Morphology Design
    /// - **Use appropriate bounds**: Match your problem's natural constraints
    /// - **Balance precision vs. search space**: More precision = larger search space
    /// - **Consider gene interactions**: Related parameters should be adjacent when possible
    /// - **Use meaningful step sizes**: Align with your problem's natural granularity
    ///
    /// ## Fitness Function Design
    /// - **Normalize to [0.0, 1.0] range**: Improves selection algorithm performance
    /// - **Handle edge cases gracefully**: Return 0.0 for invalid solutions
    /// - **Check termination regularly**: Use `terminated.is_terminated()` in long evaluations
    /// - **Avoid fitness cliffs**: Gradual fitness landscapes converge better than binary ones
    /// - **Return errors for true failures**: Let the system handle evaluation failures
    ///
    /// ## Performance Considerations
    /// - **Minimize evaluation cost**: Fitness evaluation is the bottleneck
    /// - **Use caching when appropriate**: Cache expensive computations
    /// - **Parallelize evaluations**: The system handles concurrent fitness evaluations
    /// - **Profile your evaluator**: Identify and optimize slow paths
    ///
    /// # Registration Order
    ///
    /// Multiple types can be registered with the same service:
    /// ```rust,no_run
    /// # use fx_durable_ga::services::optimization::ServiceBuilder;
    /// # struct TypeA; struct TypeB; struct EvaluatorA; struct EvaluatorB;
    /// # impl fx_durable_ga::models::Encodeable for TypeA { 
    /// #     const NAME: &'static str = "TypeA"; type Phenotype = TypeA;
    /// #     fn morphology() -> Vec<fx_durable_ga::models::GeneBounds> { vec![] }
    /// #     fn encode(&self) -> Vec<i64> { vec![] }
    /// #     fn decode(_: &[i64]) -> Self::Phenotype { TypeA }
    /// # }
    /// # impl fx_durable_ga::models::Encodeable for TypeB { 
    /// #     const NAME: &'static str = "TypeB"; type Phenotype = TypeB;
    /// #     fn morphology() -> Vec<fx_durable_ga::models::GeneBounds> { vec![] }
    /// #     fn encode(&self) -> Vec<i64> { vec![] }
    /// #     fn decode(_: &[i64]) -> Self::Phenotype { TypeB }
    /// # }
    /// # impl fx_durable_ga::models::Evaluator<TypeA> for EvaluatorA {
    /// #     fn fitness<'a>(&self, _: TypeA, _: &'a Box<dyn fx_durable_ga::models::Terminated>) 
    /// #         -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> 
    /// #         { Box::pin(async move { Ok(0.5) }) }
    /// # }
    /// # impl fx_durable_ga::models::Evaluator<TypeB> for EvaluatorB {
    /// #     fn fitness<'a>(&self, _: TypeB, _: &'a Box<dyn fx_durable_ga::models::Terminated>) 
    /// #         -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> 
    /// #         { Box::pin(async move { Ok(0.5) }) }
    /// # }
    /// # async fn example(builder: ServiceBuilder) -> Result<(), Box<dyn std::error::Error>> {
    /// let service = builder
    ///     .register::<TypeA, _>(EvaluatorA).await?
    ///     .register::<TypeB, _>(EvaluatorB).await?
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if registration succeeds, allowing method chaining.
    /// The morphology is created in the database if it doesn't exist.
    ///
    /// # Errors
    ///
    /// - Database connection or transaction failures
    /// - Morphology creation failures (invalid gene bounds)
    /// - Type hash collisions (extremely rare with different type names)
    ///
    /// # Example Usage
    ///
    /// ```rust,no_run
    /// use fx_durable_ga::{bootstrap::bootstrap, models::*};
    /// # use futures::future::BoxFuture;
    /// # use sqlx::PgPool;
    ///
    /// #[derive(Debug, Clone)]
    /// struct OptimizationTarget {
    ///     value: f64,
    /// }
    ///
    /// impl Encodeable for OptimizationTarget {
    ///     const NAME: &'static str = "OptimizationTarget";
    ///     type Phenotype = OptimizationTarget;
    ///
    ///     fn morphology() -> Vec<GeneBounds> {
    ///         vec![GeneBounds::decimal(-100.0, 100.0, 10000, 4).unwrap()]
    ///     }
    ///
    ///     fn encode(&self) -> Vec<i64> {
    ///         let bounds = Self::morphology();
    ///         vec![bounds[0].from_sample((self.value + 100.0) / 200.0)]
    ///     }
    ///
    ///     fn decode(genes: &[i64]) -> Self::Phenotype {
    ///         let bounds = Self::morphology();
    ///         OptimizationTarget {
    ///             value: bounds[0].to_f64(genes[0]) * 200.0 - 100.0
    ///         }
    ///     }
    /// }
    ///
    /// struct QuadraticEvaluator;
    ///
    /// impl Evaluator<OptimizationTarget> for QuadraticEvaluator {
    ///     fn fitness<'a>(
    ///         &self,
    ///         target: OptimizationTarget,
    ///         _terminated: &'a Box<dyn Terminated>,
    ///     ) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
    ///         Box::pin(async move {
    ///             // Maximize -(x-10)^2 + 100, so optimal x = 10
    ///             let fitness = -(target.value - 10.0).powi(2) + 100.0;
    ///             Ok(fitness / 100.0)  // Normalize to [0, 1]
    ///         })
    ///     }
    /// }
    ///
    /// # async fn example(pool: PgPool) -> Result<(), Box<dyn std::error::Error>> {
    /// let service = bootstrap(pool)
    ///     .await?
    ///     .register::<OptimizationTarget, _>(QuadraticEvaluator)
    ///     .await?
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(level = "debug", skip(self, evaluator), fields(type_name = T::NAME, type_hash = T::HASH))]
    pub async fn register<T, E>(mut self, evaluator: E) -> Result<Self, super::Error>
    where
        T: Encodeable + 'static,
        E: Evaluator<T::Phenotype> + Send + Sync + 'static,
    {
        // Insert the morphology of the type if it does not already exist in the database
        if let Err(morphologies::Error::NotFound) = self.morphologies.get_morphology(T::HASH).await
        {
            self.morphologies
                .new_morphology(Morphology::new(T::NAME, T::HASH, T::morphology()))
                .await?;
        }

        // Erase the type and store the evaluator
        let erased = super::models::ErasedEvaluator::new(evaluator, T::decode);
        self.evaluators.insert(T::HASH, Box::new(erased));

        Ok(self)
    }

    /// Sets the maximum number of deduplication attempts when breeding genotypes.
    pub fn with_max_deduplication_attempts(mut self, attempts: i32) -> Self {
        self.max_deduplication_attempts = attempts;
        self
    }

    /// Builds the optimization service with all registered evaluators.
    #[instrument(level = "debug", skip(self), fields(evaluators_count = self.evaluators.len()))]
    pub fn build(self) -> Service {
        Service {
            locking: self.locking,
            requests: self.requests,
            morphologies: self.morphologies,
            genotypes: self.genotypes,
            evaluators: self.evaluators,
            max_deduplication_attempts: self.max_deduplication_attempts,
        }
    }
}
