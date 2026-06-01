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
use std::sync::Arc;

const FITNESS_TARGET: f64 = 0.1;
const TYPE_NAME: &str = "neural_architecture";

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
            let svc = foreign_service::OptimizationService::new(
                "neural_architecture",
                ArchitectureManager,
            );
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(svc.type_name, svc.optimizer);
            Ok(())
        })
    });

    // Invoke evaluator registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<evaluation::Service>>().await?;
            provided.register(TYPE_NAME, ArchitectureManager).await;
            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    //
    // NOTE!
    // Any provider overwrites must happen after registration!
    fx_durable_ga::register(&mut c);

    // Overwrite a configuration provider
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
            String::from(TYPE_NAME),
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(10, 10),
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

    println!(
        "Request completed. Best genotype: {} with fitness {:.6}",
        genotype.id(),
        fitness
    );

    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NeuralArchitecture {
    hidden_size: usize,
    num_hidden_layers: usize,
    activation_fn: ActivationFunction,
    use_bias: bool,
    learning_rate: f64,
}

#[derive(Clone, Copy)]
struct ArchitectureManager;

#[derive(Deserialize)]
struct ResultOutput {
    validation_loss: f64,
}

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
}

impl TypeName for ArchitectureManager {
    fn type_name(&self) -> &'static str {
        TYPE_NAME
    }
}

impl foreign_service::Optimizer for ArchitectureManager {
    type Type = NeuralArchitecture;

    fn random(&self) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        Ok(Self::random_arch(&mut rng))
    }

    fn crossover(
        &self,
        parent1: Self::Type,
        parent2: Self::Type,
    ) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        let mut pick = || rng.random_range(0..2) == 0;

        Ok(NeuralArchitecture {
            hidden_size: if pick() {
                parent1.hidden_size
            } else {
                parent2.hidden_size
            },
            num_hidden_layers: if pick() {
                parent1.num_hidden_layers
            } else {
                parent2.num_hidden_layers
            },
            activation_fn: if pick() {
                parent1.activation_fn
            } else {
                parent2.activation_fn
            },
            use_bias: if pick() {
                parent1.use_bias
            } else {
                parent2.use_bias
            },
            learning_rate: if pick() {
                parent1.learning_rate
            } else {
                parent2.learning_rate
            },
        })
    }

    fn mutate(&self, genome: &mut Self::Type) -> anyhow::Result<()> {
        let mut rng = rand::rng();
        const MUTATION_RATE: f64 = 0.4;
        const TEMPERATURE: f64 = 0.8;
        let disruptive_rate = (MUTATION_RATE * TEMPERATURE.max(0.1)).min(1.0);
        // Temperature scales how disruptive mutations are.

        let maybe = |rate: f64, rng: &mut dyn RngCore| rng.random_range(0.0..1.0) < rate;

        let hidden_choices = [32usize, 64, 128, 256];
        let layer_choices = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let activation_choices = [
            ActivationFunction::Relu,
            ActivationFunction::Gelu,
            ActivationFunction::Sigmoid,
        ];
        let lr_choices = [1e-4f64, 1e-3, 1e-2];

        if maybe(MUTATION_RATE, &mut rng) {
            genome.hidden_size = hidden_choices[rng.random_range(0..hidden_choices.len())];
        }
        if maybe(disruptive_rate, &mut rng) {
            genome.num_hidden_layers = layer_choices[rng.random_range(0..layer_choices.len())];
        }
        if maybe(MUTATION_RATE, &mut rng) {
            genome.activation_fn =
                activation_choices[rng.random_range(0..activation_choices.len())];
        }
        if maybe(MUTATION_RATE, &mut rng) {
            genome.use_bias = !genome.use_bias;
        }
        if maybe(disruptive_rate, &mut rng) {
            genome.learning_rate = lr_choices[rng.random_range(0..lr_choices.len())];
        }

        Ok(())
    }
}

impl evaluation::Evaluator for ArchitectureManager {
    type Type = NeuralArchitecture;

    fn evaluate<'a>(
        &'a self,
        genome: &'a Self::Type,
    ) -> futures::future::BoxFuture<
        'a,
        std::result::Result<f64, Box<dyn std::error::Error + Send + Sync>>,
    > {
        Box::pin(async move {
            let output = tokio::process::Command::new("fx-example-regression")
                .args([
                    "--hidden-size",
                    &genome.hidden_size.to_string(),
                    "--num-hidden-layers",
                    &genome.num_hidden_layers.to_string(),
                    "--activation-fn",
                    &genome.activation_fn.to_string(),
                    "--learning-rate",
                    &genome.learning_rate.to_string(),
                ])
                .output()
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

            if !output.status.success() {
                return Err(format!(
                    "fx-example-regression failed: status={} stderr={}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                )
                .into());
            }

            eprintln!("{}", String::from_utf8_lossy(&output.stderr));

            let result: ResultOutput = serde_json::from_slice(&output.stdout)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

            Ok(result.validation_loss)
        })
    }
}
