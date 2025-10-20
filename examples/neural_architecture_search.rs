//! # Neural Architecture Search Example
//!
//! This example demonstrates how to use the fx_durable_ga crate to optimize neural network
//! architectures for regression tasks. We search over architectural parameters like:
//! - Hidden layer size
//! - Number of hidden layers  
//! - Activation function type
//! - Whether to use bias
//! - Learning rate
//!
//! The fitness function trains each candidate architecture on the California Housing dataset
//! and returns the validation loss as fitness (lower loss = better fitness).

use anyhow::Result;
use burn::backend::{ndarray::NdArray, Autodiff};
use fx_durable_ga::{
    bootstrap::bootstrap,
    models::{
        Crossover, Distribution, Encodeable, Evaluator, FitnessGoal, GeneBounds, Mutagen,
        MutationRate, Schedule, Selector, Temperature, Terminated,
    },
    services::optimization,
};
use fx_mq_building_blocks::queries::Queries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use neural_architecture_search::{
    model::{ActivationFunction, RegressionModelConfig},
    training::train_silent,
};
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use std::{env, sync::Arc};
use uuid::Uuid;

type Backend = Autodiff<NdArray>;

/// Neural network architecture representation for GA optimization.
///
/// This struct defines the searchable parameters of our neural network:
/// - hidden_size: Size of hidden layers (32-256)
/// - num_hidden_layers: Number of hidden layers (1-3)
/// - activation_fn: Type of activation function (ReLU, GELU, Sigmoid)
/// - use_bias: Whether to use bias in layers (true/false)
/// - learning_rate: Training learning rate (1e-4 to 1e-2)
#[derive(Debug, Clone)]
struct NeuralArchitecture {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub activation_fn: ActivationFunction,
    pub use_bias: bool,
    pub learning_rate: f64,
}

impl Encodeable for NeuralArchitecture {
    const NAME: &'static str = "neural_architecture";

    type Phenotype = NeuralArchitecture;

    /// Defines the search space for each architectural parameter.
    ///
    /// Gene 0: hidden_size (32, 64, 128, 256) -> encoded as 0-3
    /// Gene 1: num_hidden_layers (1, 2, 3) -> encoded as 0-2  
    /// Gene 2: activation_fn (ReLU, GELU, Sigmoid) -> encoded as 0-2
    /// Gene 3: use_bias (false, true) -> encoded as 0-1
    /// Gene 4: learning_rate (1e-4, 1e-3, 1e-2) -> encoded as 0-2
    fn morphology() -> Vec<GeneBounds> {
        vec![
            GeneBounds::integer(0, 3, 4).unwrap(),   // hidden_size: 4 options (32, 64, 128, 256)
            GeneBounds::integer(0, 2, 3).unwrap(),   // num_hidden_layers: 3 options (1, 2, 3)
            GeneBounds::integer(0, 2, 3).unwrap(),   // activation_fn: 3 options (ReLU, GELU, Sigmoid)
            GeneBounds::integer(0, 1, 2).unwrap(),   // use_bias: 2 options (false, true)
            GeneBounds::integer(0, 2, 3).unwrap(),   // learning_rate: 3 options (1e-4, 1e-3, 1e-2)
        ]
    }

    /// Converts this architecture to genetic representation.
    fn encode(&self) -> Vec<i64> {
        let hidden_size_idx = match self.hidden_size {
            32 => 0,
            64 => 1, 
            128 => 2,
            256 => 3,
            _ => 1, // Default to 64
        };

        let layers_idx = (self.num_hidden_layers - 1).min(2) as i64;

        let activation_idx = match self.activation_fn {
            ActivationFunction::Relu => 0,
            ActivationFunction::Gelu => 1,
            ActivationFunction::Sigmoid => 2,
        };

        let bias_idx = if self.use_bias { 1 } else { 0 };

        let lr_idx = if self.learning_rate <= 1e-4 {
            0
        } else if self.learning_rate <= 1e-3 {
            1
        } else {
            2
        };

        vec![hidden_size_idx, layers_idx, activation_idx, bias_idx, lr_idx]
    }

    /// Converts genotype (integer genes) to phenotype (NeuralArchitecture).
    fn decode(genes: &[i64]) -> Self::Phenotype {
        let hidden_size = match genes[0] {
            0 => 32,
            1 => 64,
            2 => 128,
            3 => 256,
            _ => 64,
        };

        let num_hidden_layers = (genes[1] + 1).clamp(1, 3) as usize;

        let activation_fn = match genes[2] {
            0 => ActivationFunction::Relu,
            1 => ActivationFunction::Gelu,
            2 => ActivationFunction::Sigmoid,
            _ => ActivationFunction::Relu,
        };

        let use_bias = genes[3] == 1;

        let learning_rate = match genes[4] {
            0 => 1e-4,
            1 => 1e-3,
            2 => 1e-2,
            _ => 1e-3,
        };

        NeuralArchitecture {
            hidden_size,
            num_hidden_layers,
            activation_fn,
            use_bias,
            learning_rate,
        }
    }
}

/// Fitness evaluator that trains neural architectures and returns validation performance.
///
/// Lower validation loss = higher fitness (we convert loss to fitness by inverting it).
struct ArchitectureEvaluator;

impl Evaluator<NeuralArchitecture> for ArchitectureEvaluator {
    /// Trains the neural architecture and returns fitness based on validation loss.
    ///
    /// Higher fitness (closer to 1.0) means better performance (lower validation loss).
    fn fitness<'a>(
        &self,
        phenotype: NeuralArchitecture,
        terminated: &'a Box<dyn Terminated>,
    ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
            if terminated.is_terminated().await {
                return Ok(0.0); // Return poor fitness if terminated
            }

            // Convert to Burn model config
            let mut model_config = RegressionModelConfig::new();
            model_config.hidden_size = phenotype.hidden_size;
            model_config.num_hidden_layers = phenotype.num_hidden_layers;
            model_config.activation_fn = phenotype.activation_fn;
            model_config.use_bias = phenotype.use_bias;
            model_config.learning_rate = phenotype.learning_rate;

            // Create device
            let device: <Backend as burn::prelude::Backend>::Device = Default::default();

            // Train the architecture and get validation loss
            // Use fewer epochs for speed (3-5 epochs should be enough to see differences)
            let validation_loss = train_silent::<Backend>(model_config, device, 3);

            // Convert loss to fitness: lower loss = higher fitness
            // Use exponential decay to reward significantly better architectures
            let fitness = (-validation_loss as f64).exp().min(1.0).max(0.001);

            println!(
                "Architecture: hidden_size={}, layers={}, activation={:?}, bias={}, lr={:.0e} -> loss={:.4}, fitness={:.4}",
                phenotype.hidden_size,
                phenotype.num_hidden_layers, 
                phenotype.activation_fn,
                phenotype.use_bias,
                phenotype.learning_rate,
                validation_loss,
                fitness
            );

            Ok(fitness)
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::from_filename(".env.local").ok();

    // Initialize logging to see optimization progress
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🧠 Neural Architecture Search with Genetic Algorithms");
    println!("Search space: 4 × 3 × 3 × 2 × 3 = 216 possible architectures");
    println!("Dataset: California Housing (regression)");
    println!();

    // Database setup - genetic algorithms need persistent storage for populations
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(50)
        .connect(&database_url)
        .await?;

    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_building_blocks::migrator::run_migrations(&pool, fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME)
        .await?;

    // Bootstrap the optimization service and register our architecture problem type
    let service = Arc::new(
        bootstrap(pool.clone())
            .await?
            .register::<NeuralArchitecture, _>(ArchitectureEvaluator)
            .await?
            .build(),
    );

    // Setup event handling and spawn an event handling agent
    let mut registry = fx_event_bus::EventHandlerRegistry::new();
    optimization::register_event_handlers(
        Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME)),
        service.clone(),
        &mut registry,
    );
    let mut listener = fx_event_bus::Listener::new(pool.clone(), registry);
    let _events_handle = tokio::spawn(async move {
        listener.listen(None).await?;
        Ok::<(), sqlx::Error>(())
    });

    // Setup job handling and initiate workers
    let host_id = Uuid::parse_str("neural-arch-search-host-12345678").expect("valid uuid");
    let hold_for = Duration::from_secs(300);
    let mut jobs_listener = fx_mq_jobs::Listener::new(
        pool.clone(),
        optimization::register_job_handlers(&service, fx_mq_jobs::RegistryBuilder::new()),
        4, // 4 workers for parallel architecture evaluation
        host_id,
        hold_for,
    )
    .await?;
    let _jobs_handle = tokio::spawn(async move {
        jobs_listener.listen().await?;
        Ok::<(), anyhow::Error>(())
    });

    // Start neural architecture search optimization
    println!("🚀 Starting neural architecture search...");
    
    service
        .new_optimization_request(
            NeuralArchitecture::NAME,
            NeuralArchitecture::HASH,
            FitnessGoal::maximize(0.95)?, // Stop when fitness reaches 95% (very good architecture found)
            Schedule::generational(50, 5), // 50 generations, 5 parallel evaluations
            Selector::tournament(3, 20),   // Tournament selection (size 3, population 20)
            Mutagen::new(
                Temperature::exponential(0.9, 0.1, 0.9, 3)?, // Cooling schedule for mutations
                MutationRate::exponential(0.4, 0.1, 0.9, 3)?, // Adaptive mutation rate
            ),
            Crossover::uniform(0.6)?, // 60% uniform crossover rate
            Distribution::random(20), // Random initial population of 20 architectures
        )
        .await?;

    println!("✅ Neural architecture search started!");
    println!("⏱️  This will run for a maximum of 5 minutes to find optimal architectures...");

    // Run for a maximum of 5 minutes
    let timeout_duration = Duration::from_secs(300);
    let start_time = std::time::Instant::now();

    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;

        // Check if we've exceeded the timeout
        if start_time.elapsed() >= timeout_duration {
            println!(
                "⏰ Timeout reached after {} seconds. Neural architecture search completed.",
                timeout_duration.as_secs()
            );
            break;
        }
    }

    println!("🎯 Check the logs above to see which architectures performed best!");
    println!("💡 Lower validation loss with higher fitness indicates better architectures.");

    Ok(())
}