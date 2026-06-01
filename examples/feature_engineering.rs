use anyhow::Result;
use chrono::Utc;
use futures::lock::Mutex;
use fx_durable_ga::repositories::genotypes::TypeName;
use fx_durable_ga::services::optimization::{
    self as foreign_service, FitnessGoal, OptimizerRegistry, Schedule, Selector,
};
use fx_durable_ga::{configuration, infrastructure::di::Container, services::evaluation};
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

const FITNESS_TARGET: f64 = 1.0;
const TYPE_NAME: &str = "feature_engineering";

/// Available source columns for features
const SOURCE_COLUMNS: &[&str] = &[
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
];

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    dotenvy::from_filename(".env.local").ok();

    // Create a DI container
    let mut c = Container::new();

    // Invoke optimizer registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register("feature_engineering", FeatureManager);
            Ok(())
        })
    });

    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<evaluation::Service>>().await?;
            provided.register(TYPE_NAME, FeatureManager).await;
            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    //
    // NOTE!
    // Any provider overwrites must happen after registration!
    fx_durable_ga::register(&mut c);

    // Overwrite the job worker count configuration provider
    c.provide(|_| Box::pin(async { Ok(configuration::JobWorkerCount { value: 8 }) }));

    // Invoke all invokables
    c.invoke().await?;

    // Get an app instance from the container
    let app = c.get::<Arc<fx_durable_ga::bootstrap::App>>().await?;

    // Get a timestamp just before we start the optimization
    let started = Utc::now();

    // Create the optimization request
    let request_id = app
        .services()
        .optimization()
        .request_new(
            String::from("feature_engineering"),
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(40, 10),
            Selector::tournament(5),
        )
        .await?;

    println!("Optimization request submitted: {}", request_id);

    app.services()
        .synchronization()
        .wait_for(&request_id.to_string(), &started)
        .await?;

    let (genotype, fitness) = app
        .services()
        .optimization()
        .get_best_genotype(request_id)
        .await?
        .expect("evaluations should exist");

    if fitness >= FITNESS_TARGET {
        println!("Exhausted the optimization budget without reaching the optimization goal")
    }

    let config = FeatureConfig::from_value(&genotype.genome());

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

    Ok(())
}

/// Transform types with their parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn from_value(value: &Value) -> Self {
        serde_json::from_value(value.clone())
            .unwrap_or_else(|_| FeatureConfig::random(&mut rand::rng()))
    }
}

struct FeatureManager;

#[derive(Deserialize)]
struct ResultOutput {
    validation_loss: f64,
}

impl TypeName for FeatureManager {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }
}

impl foreign_service::Optimizer for FeatureManager {
    type Type = FeatureConfig;

    fn random(&self) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        Ok(FeatureConfig::random(&mut rng))
    }

    fn crossover(
        &self,
        parent1: Self::Type,
        parent2: Self::Type,
    ) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        let mut features = Vec::new();

        for i in 0..7 {
            let chosen = if rng.random_range(0..2) == 0 {
                parent1.features.get(i)
            } else {
                parent2.features.get(i)
            };
            let feature = chosen.cloned().unwrap_or_else(|| {
                let source = SOURCE_COLUMNS[rng.random_range(0..SOURCE_COLUMNS.len())].to_string();
                let len = rng.random_range(0..3);
                let mut transforms = Vec::new();
                for _ in 0..len {
                    let gene = rng.random_range(0..12) as i64;
                    if let Some(t) = Transform::from_gene(gene) {
                        transforms.push(t);
                    }
                }
                Feature { source, transforms }
            });
            features.push(feature);
        }

        let mut pick = || rng.random_range(0..2) == 0;

        Ok(FeatureConfig {
            features,
            hidden_size: if pick() {
                parent1.hidden_size
            } else {
                parent2.hidden_size
            },
            learning_rate: if pick() {
                parent1.learning_rate
            } else {
                parent2.learning_rate
            },
            sequence_length: if pick() {
                parent1.sequence_length
            } else {
                parent2.sequence_length
            },
        })
    }

    fn mutate(&self, genome: &mut Self::Type) -> anyhow::Result<()> {
        let mut rng = rand::rng();
        const MUTATION_RATE: f64 = 0.35;
        const TEMPERATURE: f64 = 0.7;
        let disruptive_rate = (MUTATION_RATE * TEMPERATURE.max(0.1)).min(1.0);
        // Temperature scales how disruptive mutations are.

        let maybe = |rng: &mut dyn RngCore, rate: f64| rng.random_range(0.0..1.0) < rate;

        if maybe(&mut rng, MUTATION_RATE) {
            genome.hidden_size = [4usize, 8, 16, 32, 64, 128][rng.random_range(0..6)];
        }
        if maybe(&mut rng, MUTATION_RATE) {
            genome.learning_rate = [1e-4f64, 5e-4, 1e-3][rng.random_range(0..3)];
        }
        if maybe(&mut rng, MUTATION_RATE) {
            genome.sequence_length =
                [10usize, 20, 30, 40, 50, 60, 70, 80, 90, 100][rng.random_range(0..10)];
        }

        for feat in genome.features.iter_mut() {
            if maybe(&mut rng, MUTATION_RATE) {
                feat.source = SOURCE_COLUMNS[rng.random_range(0..SOURCE_COLUMNS.len())].to_string();
            }
            if maybe(&mut rng, disruptive_rate) {
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

        Ok(())
    }
}

impl fx_durable_ga::services::evaluation::Evaluator for FeatureManager {
    type Type = FeatureConfig;

    fn evaluate<'a>(
        &'a self,
        genome: &'a Self::Type,
    ) -> futures::future::BoxFuture<
        'a,
        std::result::Result<f64, Box<dyn std::error::Error + Send + Sync>>,
    > {
        Box::pin(async move {
            let genotype_id = Uuid::now_v7();
            let model_save_path = format!("./model_storage/{}", genotype_id);
            let mut args = vec![
                "train".to_string(),
                "--hidden-size".to_string(),
                genome.hidden_size.to_string(),
                "--learning-rate".to_string(),
                genome.learning_rate.to_string(),
                "--sequence-length".to_string(),
                genome.sequence_length.to_string(),
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
            for (i, feature) in genome.features.iter().enumerate() {
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
                return Err(format!(
                    "feng failed with exit code {:?}. stdout: {stdout}, stderr: {stderr}",
                    output.status.code()
                )
                .into());
            }

            // Parse stdout as JSON (only the last line contains the JSON result)
            let stdout = String::from_utf8_lossy(&output.stdout);
            let last_line = stdout.lines().last().unwrap_or("");
            let result: ResultOutput = serde_json::from_str(last_line).map_err(|e| {
                format!("Failed to parse JSON from feng: {e}. Last line was: {last_line}")
            })?;

            Ok(result.validation_loss)
        })
    }
}
