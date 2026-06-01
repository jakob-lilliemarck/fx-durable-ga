use anyhow::Result;
use chrono::Utc;
use futures::lock::Mutex;
use fx_durable_ga::repositories::genotypes::TypeName;
use fx_durable_ga::services::optimization::{
    self as foreign_service, FitnessGoal, OptimizerRegistry, Schedule, Selector,
};
use fx_durable_ga::{infrastructure::di::Container, services::evaluation};
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const FITNESS_TARGET: f64 = 0.05;

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
                "point",
                PointManager {
                    target: Point {
                        x: 1.0,
                        y: 1.5,
                        z: 2.5,
                    },
                },
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
            provided
                .register(
                    "point",
                    PointManager {
                        target: Point {
                            x: 1.0,
                            y: 1.5,
                            z: 2.5,
                        },
                    },
                )
                .await;
            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    //
    // NOTE!
    // Any provider overwrites must happen after registration!
    fx_durable_ga::register(&mut c);

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
            String::from("point"),
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(200, 30),
            Selector::tournament(7),
            Some(serde_json::json!({
                "mutation_rate": 0.3,
                "temperature": 0.7,
            })),
            None::<()>,
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
        "Best genotype: {} with distance {:.6}",
        genotype.id(),
        fitness
    );

    Ok(())
}

#[derive(Clone)]
struct PointManager {
    target: Point,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl TypeName for PointManager {
    fn type_name(&self) -> &'static str {
        "point"
    }
}

impl foreign_service::Optimizer for PointManager {
    type Type = Point;

    fn random(&self, _user_defined: &serde_json::Value) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        let x = rng.random_range(0.5..1.75);
        let y = rng.random_range(0.75..2.0);
        let z = rng.random_range(2.0..3.25);
        Ok(Point { x, y, z })
    }

    fn crossover(
        &self,
        parent1: Self::Type,
        parent2: Self::Type,
        _user_defined: &serde_json::Value,
    ) -> anyhow::Result<Self::Type> {
        let x = (parent1.x + parent2.x) / 2.0;
        let y = (parent1.y + parent2.y) / 2.0;
        let z = (parent1.z + parent2.z) / 2.0;
        Ok(Point { x, y, z })
    }

    fn mutate(
        &self,
        genotype: &mut Self::Type,
        user_defined: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let mut rng = rand::rng();
        // Reads mutation parameters from flat user_defined payload.
        let mutation_rate = user_defined
            .get("mutation_rate")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.3);
        let temperature = user_defined
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.7);
        let maybe_mutate = |value: &mut f64, lo: f64, hi: f64, rng: &mut dyn RngCore| {
            if rng.random_range(0.0..1.0) < mutation_rate {
                let span = (hi - lo) * temperature;
                let delta = rng.random_range(-span..span);
                *value = (*value + delta).clamp(lo, hi);
            }
        };

        maybe_mutate(&mut genotype.x, 0.5, 1.75, &mut rng);
        maybe_mutate(&mut genotype.y, 0.75, 2.0, &mut rng);
        maybe_mutate(&mut genotype.z, 2.0, 3.25, &mut rng);
        Ok(())
    }
}

impl evaluation::Evaluator for PointManager {
    type Type = Point;

    fn evaluate<'a>(
        &'a self,
        genotype: &'a Self::Type,
    ) -> futures::future::BoxFuture<
        'a,
        std::result::Result<f64, Box<dyn std::error::Error + Send + Sync>>,
    > {
        let target = self.target;
        Box::pin(async move {
            let dx = genotype.x - target.x;
            let dy = genotype.y - target.y;
            let dz = genotype.z - target.z;
            Ok((dx * dx + dy * dy + dz * dz).sqrt())
        })
    }
}
