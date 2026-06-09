use futures::future::BoxFuture;
use futures::lock::Mutex;
use fx_durable_ga::{
    bootstrap::App,
    configuration::BindAddr,
    infrastructure::di::InvokeError,
    repositories::genotypes::TypeName,
    services::evaluation,
    services::indexing::encoder::dataset::{SequenceDataSource, SequenceDataset, SequenceSample},
    services::indexing::encoder::train::AutoencoderTrainConfig,
    services::indexing::{self as indexable, EncodeInput, Indexer, TrainModelConfig},
    services::optimization::{self as foreign_service, OptimizationService, OptimizerRegistry},
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{Level, info};

#[derive(Debug, thiserror::Error)]
#[error("No bind address configured")]
pub struct NoBindAddr;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    // Create a DI container
    let mut c = fx_durable_ga::infrastructure::di::Container::new();

    // Register the optimizer
    c.invokable(|c| {
        Box::pin(async move {
            let svc = OptimizationService::new("dev::point", SimpleManager);
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(svc.optimizer);
            Ok(())
        })
    });

    // Register the evaluator
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<evaluation::Service>>().await?;
            provided.register("dev::point", SimpleManager).await;
            Ok(())
        })
    });

    // Register the indexer
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<indexable::Registry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(Arc::new(SimpleIndexer::new()))
                .await
                .map_err(|err| InvokeError::new(err))?;
            Ok(())
        })
    });

    // Register fx-durable-ga with the container
    fx_durable_ga::register(&mut c);

    // Perform any desired overwrites

    // Get the app instance from the container
    let app = c.get::<Arc<App>>().await?;

    let Some(bind_addr) = c.get::<Option<BindAddr>>().await? else {
        return Err(NoBindAddr.into());
    };

    info!("http://{}/docs", bind_addr.value);
    info!("http://{}/lineage", bind_addr.value);
    info!("http://{}/optimizations", bind_addr.value);

    c.invoke().await?;

    App::serve_http(app, bind_addr.value).await?;

    Ok(())
}

// ── Domain types ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct SimplePoint {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone)]
struct SimpleManager;

impl TypeName for SimpleManager {
    fn type_name(&self) -> &'static str {
        "dev::point"
    }
}

impl foreign_service::Optimizer for SimpleManager {
    type Type = SimplePoint;

    fn random(&self) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        Ok(SimplePoint {
            x: rng.random_range(-5.0..5.0),
            y: rng.random_range(-5.0..5.0),
            z: rng.random_range(-5.0..5.0),
        })
    }

    fn crossover(&self, p1: Self::Type, p2: Self::Type) -> anyhow::Result<Self::Type> {
        Ok(SimplePoint {
            x: (p1.x + p2.x) / 2.0,
            y: (p1.y + p2.y) / 2.0,
            z: (p1.z + p2.z) / 2.0,
        })
    }

    fn mutate(&self, instance: &mut Self::Type) -> anyhow::Result<()> {
        let mut rng = rand::rng();
        for coord in [&mut instance.x, &mut instance.y, &mut instance.z] {
            *coord += rng.random_range(-0.5..0.5);
            *coord = coord.clamp(-5.0, 5.0);
        }
        Ok(())
    }
}

impl evaluation::Evaluator for SimpleManager {
    type Type = SimplePoint;

    fn evaluate<'a>(
        &'a self,
        instance: &'a Self::Type,
    ) -> BoxFuture<'a, std::result::Result<f64, Box<dyn std::error::Error + Send + Sync>>> {
        Box::pin(async move {
            Ok(
                (instance.x * instance.x + instance.y * instance.y + instance.z * instance.z)
                    .sqrt(),
            )
        })
    }
}

#[derive(Clone)]
struct SimpleIndexer {
    dataset: SequenceDataset,
    train_config: TrainModelConfig,
}

impl SimpleIndexer {
    fn new() -> Self {
        let input_size = 3;
        let samples: Vec<SequenceSample> = (0..5)
            .map(|s| {
                let steps: Vec<Vec<f32>> = (0..20)
                    .map(|t| {
                        let mut rng = rand::rng();
                        (0..input_size)
                            .map(|f| {
                                rng.random_range(-5.0..5.0)
                                    + (s as f32 * 0.1)
                                    + (t as f32 * 0.01)
                                    + (f as f32 * 0.001)
                            })
                            .collect()
                    })
                    .collect();
                SequenceSample { steps }
            })
            .collect();
        Self {
            dataset: SequenceDataset::new(samples, input_size),
            train_config: TrainModelConfig::Lstm(AutoencoderTrainConfig {
                input_size,
                hidden_size: 32,
                latent_size: 8,
                batch_size: 8,
                epochs: 10,
                learning_rate: 0.001,
            }),
        }
    }
}

impl TypeName for SimpleIndexer {
    fn type_name(&self) -> &'static str {
        "dev::point"
    }
}

impl Indexer for SimpleIndexer {
    type Type = SimplePoint;

    fn preprocess(&self, entity: &Self::Type) -> EncodeInput {
        EncodeInput {
            values: vec![entity.x as f32, entity.y as f32, entity.z as f32],
            dimensions: vec![1, 3],
        }
    }

    fn dataset(&self) -> BoxFuture<'_, Arc<dyn SequenceDataSource>> {
        let dataset: Arc<dyn SequenceDataSource> = Arc::new(self.dataset.clone());
        Box::pin(async { dataset })
    }

    fn training_config(&self) -> &TrainModelConfig {
        &self.train_config
    }
}
