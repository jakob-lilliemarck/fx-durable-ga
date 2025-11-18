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
use fx_durable_ga::{
    bootstrap,
    models::{
        Crossover, Distribution, Encodeable, Evaluator, FitnessGoal, GeneBounds, Mutagen,
        MutationRate, NoContext, Schedule, Selector, Temperature, Terminated,
    },
    register_event_handlers, register_job_handlers,
};
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use fx_mq_jobs::Queries;
use serde::Deserialize;
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

impl Encodeable for FeatureConfig {
    const NAME: &'static str = "feature_engineering";

    type Phenotype = FeatureConfig;

    fn morphology() -> Vec<GeneBounds> {
        let mut bounds = Vec::new();

        // 7 features, each with:
        // - source column (0-10)
        // - pipeline_length (0-2)
        // - transform_1 (0-11)
        // - transform_2 (0-11)
        for _ in 0..7 {
            bounds.push(GeneBounds::integer(0, 10, 11).unwrap()); // source
            bounds.push(GeneBounds::integer(0, 2, 3).unwrap()); // pipeline_length (max 2)
            bounds.push(GeneBounds::integer(0, 11, 12).unwrap()); // transform_1
            bounds.push(GeneBounds::integer(0, 11, 12).unwrap()); // transform_2
        }

        // Hyperparameters
        bounds.push(GeneBounds::integer(0, 5, 6).unwrap()); // hidden_size: [4, 8, 16, 32, 64, 128]
        bounds.push(GeneBounds::integer(0, 2, 3).unwrap()); // learning_rate: [1e-4, 5e-4, 1e-3]
        bounds.push(GeneBounds::integer(0, 9, 10).unwrap()); // sequence_length: [10..100 step 10]

        bounds
    }

    fn encode(&self) -> Vec<i64> {
        let mut genes = Vec::new();

        // Encode 7 features
        for feature in &self.features {
            // Source column
            let source_idx = SOURCE_COLUMNS
                .iter()
                .position(|&col| col == feature.source)
                .unwrap_or(0) as i64;
            genes.push(source_idx);

            // Pipeline length (max 2)
            genes.push(feature.transforms.len().min(2) as i64);

            // Transforms (pad with 0 if fewer than 2)
            for i in 0..2 {
                if i < feature.transforms.len() {
                    genes.push(feature.transforms[i].to_gene());
                } else {
                    genes.push(0);
                }
            }
        }

        // Encode hyperparameters
        let hidden_size_idx = match self.hidden_size {
            4 => 0,
            8 => 1,
            16 => 2,
            32 => 3,
            64 => 4,
            128 => 5,
            _ => 2, // default to 16
        };
        genes.push(hidden_size_idx);

        let lr_idx = if self.learning_rate <= 1e-4 {
            0
        } else if self.learning_rate <= 5e-4 {
            1
        } else {
            2
        };
        genes.push(lr_idx);

        let seq_len_idx = ((self.sequence_length / 10).saturating_sub(1)).min(9) as i64;
        genes.push(seq_len_idx);

        genes
    }

    fn decode(genes: &[i64]) -> Self::Phenotype {
        let mut features = Vec::new();

        // Decode 7 features (4 genes per feature now: source, length, t1, t2)
        for i in 0..7 {
            let base_idx = i * 4;

            let source_idx = genes[base_idx].clamp(0, 10) as usize;
            let source = SOURCE_COLUMNS[source_idx].to_string();

            let pipeline_length = genes[base_idx + 1].clamp(0, 2) as usize;

            let mut transforms = Vec::new();
            for j in 0..pipeline_length {
                if let Some(transform) = Transform::from_gene(genes[base_idx + 2 + j]) {
                    transforms.push(transform);
                }
            }

            features.push(Feature { source, transforms });
        }

        // Decode hyperparameters (genes are now at index 28, 29, 30)
        let hidden_size = match genes[28] {
            0 => 4,
            1 => 8,
            2 => 16,
            3 => 32,
            4 => 64,
            5 => 128,
            _ => 32,
        };

        let learning_rate = match genes[29] {
            0 => 1e-4,
            1 => 5e-4,
            2 => 1e-3,
            _ => 5e-4,
        };

        let sequence_length = ((genes[30].clamp(0, 9) + 1) * 10) as usize;

        FeatureConfig {
            features,
            hidden_size,
            learning_rate,
            sequence_length,
        }
    }
}

struct FeatureEvaluator;

#[derive(Deserialize)]
struct ResultOutput {
    validation_loss: f64,
}

impl Evaluator<FeatureConfig> for FeatureEvaluator {
    fn fitness<'a>(
        &self,
        genotype_id: Uuid,
        phenotype: FeatureConfig,
        _: &NoContext,
        _: &'a Box<dyn Terminated>,
    ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
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
            .register::<FeatureConfig, _>(FeatureEvaluator)
            .await?
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
            FeatureConfig::NAME,
            FeatureConfig::HASH,
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(40, 10),
            Selector::tournament(5, 45)?,
            Mutagen::new(Temperature::constant(0.7)?, MutationRate::constant(0.35)?),
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(40),
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
        let config = FeatureConfig::decode(&genotype.genome());
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
