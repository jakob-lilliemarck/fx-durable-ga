//! Neural Architecture Search with Genetic Algorithms
//!
//! Demonstrates using fx_durable_ga to optimize neural network hyperparameters:
//! hidden size, number of layers, activation function, bias usage, and learning rate.
//!
//! **IMPORTANT!**
//! This library requires fx-durable-ga-example-simple-regression to be installed and available on the PATH. The crate is available here:
//! https://github.com/jakob-lilliemarck/fx-durable-ga-simple-regression
//!
//! This workaround is required as the Autodiff backend of the Burn ML framework currently does not free memory between training run.
//! As such, running multiple training runs will cause unbounded memory allocation.
//!
//! This example handles that by running each training run as a subprocess, in which case all memory allocations are freed after each run.

use anyhow::Result;
use const_fnv1a_hash::fnv1a_hash_str_32;
use fx_durable_ga::{
    bootstrap,
    models::{FitnessGoal, GenotypeManager, Schedule, Selector},
    register_event_handlers, register_job_handlers,
};
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use fx_mq_jobs::Queries;
use rand::{Rng, RngCore};
use serde::Deserialize;
use serde_json::Value;
use sqlx::postgres::PgPoolOptions;
use std::{env, sync::Arc};
use std::{str::FromStr, time::Duration};
use tracing::Level;
use uuid::Uuid;

const WORKERS: usize = 4;
const FITNESS_TARGET: f64 = 0.1;

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
struct ArchitectureManager;

#[derive(Deserialize)]
struct ResultOutput {
    validation_loss: f64,
}

const TYPE_NAME: &str = "neural_architecture";
const TYPE_HASH: i32 = fnv1a_hash_str_32(TYPE_NAME) as i32;

impl ArchitectureManager {
    fn random_arch(rng: &mut dyn RngCore) -> NeuralArchitecture {
        let hidden_choices = [32usize, 64, 128, 256];
        let layer_choices = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let activation_choices = [
            ActivationFunction::Relu,
            ActivationFunction::Gelu,
            ActivationFunction::Sigmoid,
        ];
        let lr_choices = [1e-4f64, 1e-3, 1e-2];

        NeuralArchitecture {
            hidden_size: hidden_choices[rng.random_range(0..hidden_choices.len())],
            num_hidden_layers: layer_choices[rng.random_range(0..layer_choices.len())],
            activation_fn: activation_choices[rng.random_range(0..activation_choices.len())],
            use_bias: rng.random_range(0..2) == 1,
            learning_rate: lr_choices[rng.random_range(0..lr_choices.len())],
        }
    }

    fn to_json(arch: &NeuralArchitecture) -> Value {
        serde_json::json!({
            "hidden_size": arch.hidden_size,
            "num_hidden_layers": arch.num_hidden_layers,
            "activation_fn": arch.activation_fn.to_string(),
            "use_bias": arch.use_bias,
            "learning_rate": arch.learning_rate,
        })
    }

    fn from_json(genome: &Value) -> NeuralArchitecture {
        let hidden_size = genome
            .get("hidden_size")
            .and_then(Value::as_u64)
            .unwrap_or(64) as usize;
        let num_hidden_layers = genome
            .get("num_hidden_layers")
            .and_then(Value::as_u64)
            .unwrap_or(2) as usize;
        let activation_fn = match genome
            .get("activation_fn")
            .and_then(Value::as_str)
            .unwrap_or("relu")
        {
            "gelu" => ActivationFunction::Gelu,
            "sigmoid" => ActivationFunction::Sigmoid,
            _ => ActivationFunction::Relu,
        };
        let use_bias = genome
            .get("use_bias")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let learning_rate = genome
            .get("learning_rate")
            .and_then(Value::as_f64)
            .unwrap_or(1e-3);

        NeuralArchitecture {
            hidden_size,
            num_hidden_layers: num_hidden_layers.clamp(1, 8),
            activation_fn,
            use_bias,
            learning_rate,
        }
    }
}

impl GenotypeManager for ArchitectureManager {
    fn name(&self) -> &'static str {
        TYPE_NAME
    }

    fn random(&self, rng: &mut dyn RngCore, _user_defined: &Value) -> anyhow::Result<Value> {
        Ok(Self::to_json(&Self::random_arch(rng)))
    }

    fn crossover(
        &self,
        parent1: &Value,
        parent2: &Value,
        rng: &mut dyn RngCore,
        _user_defined: &Value,
    ) -> anyhow::Result<Value> {
        let mut pick = |k: &str| {
            if rng.random_range(0..2) == 0 {
                parent1[k].clone()
            } else {
                parent2[k].clone()
            }
        };

        Ok(serde_json::json!({
            "hidden_size": pick("hidden_size"),
            "num_hidden_layers": pick("num_hidden_layers"),
            "activation_fn": pick("activation_fn"),
            "use_bias": pick("use_bias"),
            "learning_rate": pick("learning_rate"),
        }))
    }

    fn mutate(
        &self,
        genome: &mut Value,
        rng: &mut dyn RngCore,
        progress: f64,
        user_defined: &Value,
    ) -> anyhow::Result<()> {
        let mut arch = Self::from_json(genome);
        let (mutation_rate, _temperature) = user_defined
            .get("mutate")
            .and_then(Value::as_object)
            .map(|cfg| {
                let mr = cfg
                    .get("mutation_rate")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.4);
                let temp = cfg
                    .get("temperature")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.8);
                (mr * (1.0 - progress).max(0.0), temp)
            })
            .unwrap_or((0.4, 0.8));

        let maybe = |rate: f64, rng: &mut dyn RngCore| rng.random_range(0.0..1.0) < rate;

        let hidden_choices = [32usize, 64, 128, 256];
        let layer_choices = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let activation_choices = [
            ActivationFunction::Relu,
            ActivationFunction::Gelu,
            ActivationFunction::Sigmoid,
        ];
        let lr_choices = [1e-4f64, 1e-3, 1e-2];

        if maybe(mutation_rate, rng) {
            arch.hidden_size = hidden_choices[rng.random_range(0..hidden_choices.len())];
        }
        if maybe(mutation_rate, rng) {
            arch.num_hidden_layers = layer_choices[rng.random_range(0..layer_choices.len())];
        }
        if maybe(mutation_rate, rng) {
            arch.activation_fn = activation_choices[rng.random_range(0..activation_choices.len())];
        }
        if maybe(mutation_rate, rng) {
            arch.use_bias = !arch.use_bias;
        }
        if maybe(mutation_rate, rng) {
            arch.learning_rate = lr_choices[rng.random_range(0..lr_choices.len())];
        }

        *genome = Self::to_json(&arch);
        Ok(())
    }

    fn evaluate<'a>(
        &'a self,
        genome: &'a Value,
        _user_defined: &'a Value,
    ) -> futures::future::BoxFuture<'a, anyhow::Result<f64>> {
        Box::pin(async move {
            let arch = ArchitectureManager::from_json(genome);

            let output = tokio::process::Command::new("fx-example-regression")
                .args([
                    "--hidden-size",
                    &arch.hidden_size.to_string(),
                    "--num-hidden-layers",
                    &arch.num_hidden_layers.to_string(),
                    "--activation-fn",
                    &arch.activation_fn.to_string(),
                    "--learning-rate",
                    &arch.learning_rate.to_string(),
                ])
                .output()
                .await
                .expect("Failed to run fx-example-regression");

            if !output.status.success() {
                anyhow::bail!(
                    "fx-example-regression failed: status={} stderr={}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                );
            }

            eprintln!("{}", String::from_utf8_lossy(&output.stderr));

            let result: ResultOutput =
                serde_json::from_slice(&output.stdout).expect("Invalid JSON from training binary");

            Ok(result.validation_loss)
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    let host_id_str = env::var("HOST_ID").expect("HOST_ID must be set");
    let host_id = Uuid::from_str(&host_id_str).expect("HOST_ID could not be parsed to UUID");

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    // Run all default migrations
    fx_durable_ga::migrations::run_default_migrations(&pool).await?;

    let service = Arc::new(
        bootstrap(host_id, pool.clone())
            .await?
            .with_genotype_manager(ArchitectureManager)
            .build()
            .await?,
    );

    let mut registry = fx_event_bus::EventHandlerRegistry::new();
    register_event_handlers(
        Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME)),
        service.clone(),
        &mut registry,
    );
    let mut listener = fx_event_bus::Listener::new(pool.clone(), registry);
    tokio::spawn(async move { listener.listen(None).await });

    let host_id = Uuid::parse_str("00000000-0000-0000-0000-123456789abc")?;
    let mut jobs_listener = fx_mq_jobs::Listener::new(
        pool.clone(),
        register_job_handlers(&service, fx_mq_jobs::RegistryBuilder::new()),
        WORKERS,
        host_id,
        Duration::from_secs(600),
    )
    .await?;
    tokio::spawn(async move { jobs_listener.listen().await });

    service
        .new_optimization_request(
            TYPE_NAME,
            TYPE_HASH,
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(10, 10),
            Selector::tournament(5),
            serde_json::json!({
                "crossover": { "probability": 0.5 },
                "mutate": { "mutation_rate": 0.4, "temperature": 0.8 },
                "distribution": { "population_size": 15 }
            }),
            None::<()>,
        )
        .await?;

    tokio::time::sleep(Duration::from_secs(3600)).await;
    Ok(())
}
