//! Neural Architecture Search with Genetic Algorithms
//!
//! Demonstrates using fx_durable_ga to optimize neural network hyperparameters:
//! hidden size, number of layers, activation function, bias usage, and learning rate.

use anyhow::Result;
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
use serde::Deserialize;
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use std::{env, sync::Arc};
use tracing::Level;
use uuid::Uuid;

const WORKERS: usize = 4;
const FITNESS_TARGET: f64 = 0.05;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum ActivationFunction {
    Relu,
    Gelu,
    Sigmoid,
}

impl std::fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Relu => write!(f, "relu"),
            Self::Gelu => write!(f, "gelu"),
            Self::Sigmoid => write!(f, "sigmoid"),
        }
    }
}

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

    fn morphology() -> Vec<GeneBounds> {
        vec![
            GeneBounds::integer(0, 3, 4).unwrap(), // hidden_size: [32, 64, 128, 256]
            GeneBounds::integer(0, 2, 3).unwrap(), // num_hidden_layers: [1, 2, 3]
            GeneBounds::integer(0, 2, 3).unwrap(), // activation_fn: [ReLU, GELU, Sigmoid]
            GeneBounds::integer(0, 1, 2).unwrap(), // use_bias: [false, true]
            GeneBounds::integer(0, 2, 3).unwrap(), // learning_rate: [1e-4, 1e-3, 1e-2]
        ]
    }

    fn encode(&self) -> Vec<i64> {
        let hidden_size_idx = match self.hidden_size {
            32 => 0,
            64 => 1,
            128 => 2,
            256 => 3,
            _ => 1,
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

        vec![
            hidden_size_idx,
            layers_idx,
            activation_idx,
            bias_idx,
            lr_idx,
        ]
    }

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

struct ArchitectureEvaluator;

#[derive(Deserialize)]
struct ResultOutput {
    validation_loss: f64,
}

impl Evaluator<NeuralArchitecture> for ArchitectureEvaluator {
    fn fitness<'a>(
        &self,
        phenotype: NeuralArchitecture,
        _: &'a Box<dyn Terminated>,
    ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
            // Spawn the binary
            let output = std::process::Command::new("fx-example-regression")
                .args([
                    "--hidden-size",
                    &phenotype.hidden_size.to_string(),
                    "--num-hidden-layers",
                    &phenotype.num_hidden_layers.to_string(),
                    "--activation-fn",
                    &phenotype.activation_fn.to_string(),
                    "--learning-rate",
                    "0.001",
                ])
                .output()
                .expect("Failed to run fx-example-regression");

            // Print stderr logs (your tracing output)
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));

            tracing::info!(message = "received JSON", output = ?output);
            // Parse stdout as JSON (the ResultOutput struct)
            let result: ResultOutput =
                serde_json::from_slice(&output.stdout).expect("Invalid JSON from training binary");

            Ok(result.validation_loss)
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::from_filename(".env.local").ok();
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_building_blocks::migrator::run_migrations(&pool, FX_MQ_JOBS_SCHEMA_NAME).await?;

    let service = Arc::new(
        bootstrap(pool.clone())
            .await?
            .register::<NeuralArchitecture, _>(ArchitectureEvaluator)
            .await?
            .build(),
    );

    let mut registry = fx_event_bus::EventHandlerRegistry::new();
    optimization::register_event_handlers(
        Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME)),
        service.clone(),
        &mut registry,
    );
    let mut listener = fx_event_bus::Listener::new(pool.clone(), registry);
    tokio::spawn(async move { listener.listen(None).await });

    let host_id = Uuid::parse_str("00000000-0000-0000-0000-123456789abc")?;
    let mut jobs_listener = fx_mq_jobs::Listener::new(
        pool.clone(),
        optimization::register_job_handlers(&service, fx_mq_jobs::RegistryBuilder::new()),
        WORKERS,
        host_id,
        Duration::from_secs(600),
    )
    .await?;
    tokio::spawn(async move { jobs_listener.listen().await });

    service
        .new_optimization_request(
            NeuralArchitecture::NAME,
            NeuralArchitecture::HASH,
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(10, 10),
            Selector::tournament(5, 15),
            Mutagen::new(Temperature::constant(0.8)?, MutationRate::constant(0.4)?),
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(15),
        )
        .await?;

    tokio::time::sleep(Duration::from_secs(3600)).await;
    Ok(())
}
