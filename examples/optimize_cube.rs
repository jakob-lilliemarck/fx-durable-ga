use anyhow::Result;
use fx_durable_ga::{
    bootstrap::bootstrap,
    models::{
        Crossover, Distribution, Encodeable, Evaluator, FitnessGoal, GeneBounds, Mutagen, Schedule,
        Selector, Terminated,
    },
    services::optimization,
};
use fx_mq_building_blocks::queries::Queries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use std::{env, sync::Arc};
use uuid::Uuid;

#[derive(Debug)]
struct Cube;

impl Encodeable for Cube {
    const NAME: &'static str = "cube";

    type Phenotype = (f64, f64, f64);

    fn morphology() -> Vec<GeneBounds> {
        vec![
            GeneBounds::decimal(0.5, 1.75, 1000, 3).unwrap(), // x: 0.500 to 1.750 (scaled: 500 to 1750)
            GeneBounds::decimal(0.75, 2.00, 1000, 3).unwrap(), // y: 0.750 to 2.000 (scaled: 750 to 2000)
            GeneBounds::decimal(2.00, 3.25, 1000, 3).unwrap(), // z: 2.000 to 3.250 (scaled: 2000 to 3250)
        ]
    }

    fn encode(&self) -> Vec<i64> {
        vec![0, 0, 0] // Initial point, not used since we generate random points
    }

    fn decode(genes: &[i64]) -> Self::Phenotype {
        let bounds = Self::morphology();
        (
            bounds[0].to_f64(genes[0]), // Convert to actual decimal coordinates
            bounds[1].to_f64(genes[1]), // e.g., gene 0 → 0.5, gene 999 → 1.75
            bounds[2].to_f64(genes[2]), // e.g., gene 0 → 2.0, gene 999 → 3.25
        )
    }
}

struct CubeQuadraticEvaluator;

impl Evaluator<(f64, f64, f64)> for CubeQuadraticEvaluator {
    fn fitness<'a>(
        &self,
        phenotype: (f64, f64, f64),
        terminated: &'a Box<dyn Terminated>,
    ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
            if terminated.is_terminated().await {
                // abort
            }

            let (x, y, z) = phenotype;

            // Calculate distance from origin
            let dist = (x * x + y * y + z * z).sqrt();

            // Target distance (distance from origin to optimal point (0.5, 0.75, 2.0))
            let target_dist = (0.5_f64.powi(2) + 0.75_f64.powi(2) + 2.0_f64.powi(2)).sqrt(); // ≈ 2.1943

            // Fitness 1.0 if distance matches target, decreases as we deviate
            let distance_diff = (dist - target_dist).abs();
            let fitness = 1.0 / (1.0 + distance_diff);

            Ok(fitness)
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::from_filename(".env.local").ok();

    // Initialize logging at DEBUG level
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Get database URL from environment
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(50)
        .connect(&database_url)
        .await?;

    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_building_blocks::migrator::run_migrations(&pool, fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME)
        .await?;

    // Create service instance
    let service = Arc::new(
        bootstrap(pool.clone())
            .await?
            .register::<Cube, _>(CubeQuadraticEvaluator)
            .await?
            .build(),
    );

    // setup event handling and spawn an event handling agent
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

    // setup job handling and initiate workers
    let host_id = Uuid::parse_str("ba3a4752-c4ce-4129-aa1d-55a0b2107e68").expect("valid uuid");
    let hold_for = Duration::from_secs(300);
    let mut jobs_listener = fx_mq_jobs::Listener::new(
        pool.clone(),
        optimization::register_job_handlers(&service, fx_mq_jobs::RegistryBuilder::new()),
        8,
        host_id,
        hold_for,
    )
    .await?;
    let _jobs_handle = tokio::spawn(async move {
        jobs_listener.listen().await?;
        Ok::<(), anyhow::Error>(())
    });

    // Create n optimization requests
    for _ in 0..20 {
        service
            .new_optimization_request(
                Cube::NAME,
                Cube::HASH,
                FitnessGoal::maximize(0.99)?,
                Schedule::generational(100, 10),
                Selector::tournament(3, 50),
                Mutagen::constant(0.5, 0.1)?,
                Crossover::uniform(0.5)?,
                Distribution::random(50),
            )
            .await?;
    }

    // Run for a maximum of 5 seconds
    let timeout_duration = Duration::from_secs(300);
    let start_time = std::time::Instant::now();

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Check if we've exceeded the timeout
        if start_time.elapsed() >= timeout_duration {
            println!(
                "Timeout reached after {} seconds. Stopping optimization.",
                timeout_duration.as_secs()
            );
            break;
        }
    }

    Ok(())
}
