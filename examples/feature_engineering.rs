//! Feature Engineering Optimization with Genetic Algorithms
//!
//! Demonstrates using fx_durable_ga to optimize feature selection and preprocessing
//! pipelines for time series forecasting.
//!
//! **IMPORTANT!**
//! This library requires fx-durable-ga-example-feature-engineering to be installed
//! and available on the PATH as `feng`. The crate is available at:
//! https://github.com/jakob-lilliemarck/fx-durable-ga-example-feature-engineering
//!
//! This example optimizes:
//! - Which 7 features to use from available columns (TEMP, PRES, DEWP, etc.)
//! - Preprocessing pipeline for each feature (max 2 transforms: ZSCORE, ROC, STD)
//! - Neural network hyperparameters (hidden size, learning rate, sequence length)
//!
//! Search space: ~2.8 × 10^26 possible configurations
//! (11^7 source columns × 3^7 pipeline lengths × 12^7 transform_1 × 12^7 transform_2 × 6 hidden sizes × 3 learning rates × 10 sequence lengths)

use anyhow::Result;
use const_fnv1a_hash::fnv1a_hash_str_32;
use fx_durable_ga::{
    bootstrap,
    models::{FitnessGoal, GenotypeManager, Schedule, Selector, Terminated},
    register_event_handlers, register_job_handlers,
};
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use fx_mq_jobs::Queries;
use rand::{Rng, RngCore};
use serde::Deserialize;
use serde_json::Value;
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use std::{env, sync::Arc};
use tracing::Level;
use uuid::Uuid;

const WORKERS: usize = 5;
const FITNESS_TARGET: f64 = 1.0;

/// Available source columns for features
const SOURCE_COLUMNS: &[&str] = &[
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
];

/// Transform types with their parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transform {
    ZScore10,
    ZScore24,
    ZScore48,
    ZScore96,
    Roc1,
    Roc4,
    Roc8,
    Roc12,
    Std10,
    Std24,
    Std48,
    Std96,
}

impl Transform {
    fn from_gene(gene: i64) -> Option<Self> {
        match gene {
            0 => Some(Self::ZScore10),
            1 => Some(Self::ZScore24),
            2 => Some(Self::ZScore48),
            3 => Some(Self::ZScore96),
            4 => Some(Self::Roc1),
            5 => Some(Self::Roc4),
            6 => Some(Self::Roc8),
            7 => Some(Self::Roc12),
            8 => Some(Self::Std10),
            9 => Some(Self::Std24),
            10 => Some(Self::Std48),
            11 => Some(Self::Std96),
            _ => None,
        }
    }

    fn to_gene(self) -> i64 {
        match self {
            Self::ZScore10 => 0,
            Self::ZScore24 => 1,
            Self::ZScore48 => 2,
            Self::ZScore96 => 3,
            Self::Roc1 => 4,
            Self::Roc4 => 5,
            Self::Roc8 => 6,
            Self::Roc12 => 7,
            Self::Std10 => 8,
            Self::Std24 => 9,
            Self::Std48 => 10,
            Self::Std96 => 11,
        }
    }

    fn to_string(self) -> String {
        match self {
            Self::ZScore10 => "ZSCORE(10)".to_string(),
            Self::ZScore24 => "ZSCORE(24)".to_string(),
            Self::ZScore48 => "ZSCORE(48)".to_string(),
            Self::ZScore96 => "ZSCORE(96)".to_string(),
            Self::Roc1 => "ROC(1)".to_string(),
            Self::Roc4 => "ROC(4)".to_string(),
            Self::Roc8 => "ROC(8)".to_string(),
            Self::Roc12 => "ROC(12)".to_string(),
            Self::Std10 => "STD(10)".to_string(),
            Self::Std24 => "STD(24)".to_string(),
            Self::Std48 => "STD(48)".to_string(),
            Self::Std96 => "STD(96)".to_string(),
        }
    }
}

/// A single feature with its source and preprocessing pipeline
#[derive(Debug, Clone)]
struct Feature {
    source: String,
    transforms: Vec<Transform>,
}

impl Feature {
    fn to_cli_arg(&self, name: &str) -> String {
        let pipeline = self
            .transforms
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        if pipeline.is_empty() {
            format!("{}={}", name, self.source)
        } else {
            format!("{}={}:{}", name, self.source, pipeline)
        }
    }
}

/// Configuration for feature engineering optimization
#[derive(Debug, Clone)]
struct FeatureConfig {
    features: Vec<Feature>,
    hidden_size: usize,
    learning_rate: f64,
    sequence_length: usize,
}

impl FeatureConfig {
    fn random(rng: &mut dyn RngCore) -> Self {
        let mut features = Vec::new();
        for _ in 0..7 {
            let source = SOURCE_COLUMNS[rng.random_range(0..SOURCE_COLUMNS.len())].to_string();
            let pipeline_length = rng.random_range(0..3); // 0,1,2
            let mut transforms = Vec::new();
            for _ in 0..pipeline_length {
                let gene = rng.random_range(0..12) as i64;
                if let Some(t) = Transform::from_gene(gene) {
                    transforms.push(t);
                }
            }
            features.push(Feature { source, transforms });
        }

        let hidden_choices = [4usize, 8, 16, 32, 64, 128];
        let lr_choices = [1e-4f64, 5e-4, 1e-3];
        let seq_choices = [10usize, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        FeatureConfig {
            features,
            hidden_size: hidden_choices[rng.random_range(0..hidden_choices.len())],
            learning_rate: lr_choices[rng.random_range(0..lr_choices.len())],
            sequence_length: seq_choices[rng.random_range(0..seq_choices.len())],
        }
    }

    fn to_json(&self) -> Value {
        serde_json::json!({
            "features": self.features.iter().map(|f| {
                serde_json::json!({
                    "source": f.source,
                    "transforms": f.transforms.iter().map(|t| t.to_gene()).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
        })
    }

    fn from_json(genome: &Value) -> Self {
        let features = genome
            .get("features")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|f| {
                let source = f
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or("TEMP")
                    .to_string();
                let transforms = f
                    .get("transforms")
                    .and_then(Value::as_array)
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_i64())
                    .filter_map(Transform::from_gene)
                    .take(2)
                    .collect::<Vec<_>>();
                Feature { source, transforms }
            })
            .collect::<Vec<_>>();

        FeatureConfig {
            features: if features.len() == 7 {
                features
            } else {
                FeatureConfig::random(&mut rand::rng()).features
            },
            hidden_size: genome
                .get("hidden_size")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(32),
            learning_rate: genome
                .get("learning_rate")
                .and_then(Value::as_f64)
                .unwrap_or(5e-4),
            sequence_length: genome
                .get("sequence_length")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(60),
        }
    }
}

struct FeatureManager;

#[derive(Deserialize)]
struct ResultOutput {
    validation_loss: f64,
}

impl GenotypeManager for FeatureManager {
    fn name(&self) -> &'static str {
        TYPE_NAME
    }
    fn random(&self, rng: &mut dyn RngCore, _user_defined: &Value) -> anyhow::Result<Value> {
        Ok(FeatureConfig::random(rng).to_json())
    }

    fn crossover(
        &self,
        parent1: &Value,
        parent2: &Value,
        rng: &mut dyn RngCore,
        _user_defined: &Value,
    ) -> anyhow::Result<Value> {
        let random_feature = |rng: &mut dyn RngCore| {
            let source = SOURCE_COLUMNS[rng.random_range(0..SOURCE_COLUMNS.len())].to_string();
            let len = rng.random_range(0..3);
            let mut transforms = Vec::new();
            for _ in 0..len {
                let gene = rng.random_range(0..12) as i64;
                if let Some(t) = Transform::from_gene(gene) {
                    transforms.push(t.to_gene());
                }
            }
            serde_json::json!({ "source": source, "transforms": transforms })
        };

        // Features: per index pick parent feature or random fallback
        let feats1 = parent1
            .get("features")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let feats2 = parent2
            .get("features")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut features = Vec::new();
        for i in 0..7 {
            let chosen = if rng.random_range(0..2) == 0 {
                feats1.get(i)
            } else {
                feats2.get(i)
            };
            features.push(chosen.cloned().unwrap_or_else(|| random_feature(rng)));
        }

        let mut pick_scalar = |k: &str| {
            if rng.random_range(0..2) == 0 {
                parent1[k].clone()
            } else {
                parent2[k].clone()
            }
        };

        Ok(serde_json::json!({
            "features": features,
            "hidden_size": pick_scalar("hidden_size"),
            "learning_rate": pick_scalar("learning_rate"),
            "sequence_length": pick_scalar("sequence_length"),
        }))
    }

    fn mutate(
        &self,
        genome: &mut Value,
        rng: &mut dyn RngCore,
        progress: f64,
        user_defined: &Value,
    ) -> anyhow::Result<()> {
        let mut cfg = FeatureConfig::from_json(genome);
        let maybe = |rng: &mut dyn RngCore, rate: f64| rng.random_range(0.0..1.0) < rate;
        let mutation_rate = user_defined
            .get("mutate")
            .and_then(Value::as_object)
            .and_then(|o| o.get("mutation_rate"))
            .and_then(Value::as_f64)
            .unwrap_or(0.35)
            * (1.0 - progress).max(0.0);

        if maybe(rng, mutation_rate) {
            cfg.hidden_size = [4usize, 8, 16, 32, 64, 128][rng.random_range(0..6)];
        }
        if maybe(rng, mutation_rate) {
            cfg.learning_rate = [1e-4f64, 5e-4, 1e-3][rng.random_range(0..3)];
        }
        if maybe(rng, mutation_rate) {
            cfg.sequence_length =
                [10usize, 20, 30, 40, 50, 60, 70, 80, 90, 100][rng.random_range(0..10)];
        }

        for feat in cfg.features.iter_mut() {
            if maybe(rng, mutation_rate) {
                feat.source = SOURCE_COLUMNS[rng.random_range(0..SOURCE_COLUMNS.len())].to_string();
            }
            if maybe(rng, mutation_rate) {
                let len = rng.random_range(0..3);
                feat.transforms.clear();
                for _ in 0..len {
                    let gene = rng.random_range(0..12) as i64;
                    if let Some(t) = Transform::from_gene(gene) {
                        feat.transforms.push(t);
                    }
                }
            }
        }

        *genome = cfg.to_json();
        Ok(())
    }

    fn evaluate<'a>(
        &'a self,
        genome: &'a Value,
        terminated: &'a dyn Terminated,
        _user_defined: &'a Value,
    ) -> futures::future::BoxFuture<'a, anyhow::Result<f64>> {
        Box::pin(async move {
            if terminated.is_terminated().await {
                return Ok(f64::MAX);
            }

            let phenotype = FeatureConfig::from_json(genome);
            let genotype_id = Uuid::now_v7();
            let model_save_path = format!("./model_storage/{}", genotype_id);
            let mut args = vec![
                "train".to_string(),
                "--hidden-size".to_string(),
                phenotype.hidden_size.to_string(),
                "--learning-rate".to_string(),
                phenotype.learning_rate.to_string(),
                "--sequence-length".to_string(),
                phenotype.sequence_length.to_string(),
                "--prediction-horizon".to_string(),
                "1".to_string(),
                "--batch-size".to_string(),
                "100".to_string(),
                "--epochs".to_string(),
                "25".to_string(),
                "--model-save-path".to_string(),
                model_save_path,
            ];

            // Add time features (always included)
            args.push("--feature".to_string());
            args.push("hour_sin=hour:SIN(24)".to_string());
            args.push("--feature".to_string());
            args.push("hour_cos=hour:COS(24)".to_string());
            args.push("--feature".to_string());
            args.push("month_sin=month:SIN(12)".to_string());
            args.push("--feature".to_string());
            args.push("month_cos=month:COS(12)".to_string());

            // Add optimized features
            for (i, feature) in phenotype.features.iter().enumerate() {
                let feature_name = format!("feat_{}", i);
                args.push("--feature".to_string());
                args.push(feature.to_cli_arg(&feature_name));
            }

            // Add target (always TEMP)
            args.push("--target".to_string());
            args.push("target_temp=TEMP".to_string());

            // Spawn the binary
            let output = tokio::process::Command::new("feng")
                .args(&args)
                .output()
                .await
                .expect("Failed to run feng");

            // Print stderr logs
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                eprintln!("feng stderr: {}", stderr);
            }

            // Check exit code
            if !output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return Err(anyhow::anyhow!(
                    "feng failed with exit code {:?}. stdout: {}, stderr: {}",
                    output.status.code(),
                    stdout,
                    stderr
                ));
            }

            // Parse stdout as JSON (only the last line contains the JSON result)
            let stdout = String::from_utf8_lossy(&output.stdout);
            let last_line = stdout.lines().last().unwrap_or("");
            let result: ResultOutput = serde_json::from_str(last_line).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to parse JSON from feng: {}. Last line was: {}",
                    e,
                    last_line
                )
            })?;

            Ok(result.validation_loss)
        })
    }
}

const TYPE_NAME: &str = "feature_engineering";
const TYPE_HASH: i32 = fnv1a_hash_str_32(TYPE_NAME) as i32;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    // Run all default migrations
    fx_durable_ga::migrations::run_default_migrations(&pool).await?;

    let service = Arc::new(
        bootstrap(pool.clone())
            .await?
            .with_genotype_manager(FeatureManager)
            .build(),
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

    let request_id = service
        .new_optimization_request(
            TYPE_NAME,
            TYPE_HASH,
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(40, 10),
            Selector::tournament(5, 45)?,
            serde_json::json!({
                "crossover": { "probability": 0.5 },
                "mutate": { "mutation_rate": 0.35, "temperature": 0.7 },
                "distribution": { "population_size": 40 }
            }),
            None::<()>,
        )
        .await?;

    // Poll for completion every 15 seconds (timeout after 1 hour)
    let timeout = Duration::from_secs(7200);
    let poll_interval = Duration::from_secs(15);
    let start = std::time::Instant::now();

    loop {
        if service.is_request_concluded(request_id).await? {
            println!("\nOptimization completed!");
            break;
        }

        if start.elapsed() > timeout {
            println!("\nOptimization timed out after 1 hour.");
            break;
        }

        tokio::time::sleep(poll_interval).await;
    }

    // Get and print the best configuration
    if let Some((genotype, fitness)) = service.get_best_genotype(request_id).await? {
        let config = FeatureConfig::from_json(&genotype.genome());
        println!("\n=== Best Configuration ===");
        println!("Fitness (MSE): {:.6}", fitness);
        println!("RMSE: {:.6}°C", fitness.sqrt());
        println!("\nHyperparameters:");
        println!("  Hidden Size: {}", config.hidden_size);
        println!("  Learning Rate: {}", config.learning_rate);
        println!("  Sequence Length: {}", config.sequence_length);
        println!("\nFeatures:");
        for (i, feature) in config.features.iter().enumerate() {
            let pipeline = feature
                .transforms
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" → ");
            if pipeline.is_empty() {
                println!("  feat_{}: {}", i, feature.source);
            } else {
                println!("  feat_{}: {} → {}", i, feature.source, pipeline);
            }
        }
    } else {
        println!("No genotypes were evaluated.");
    }

    Ok(())
}
