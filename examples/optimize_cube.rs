use anyhow::Result;
use fx_durable_ga::{
    bootstrap::bootstrap,
    models::{Encodeable, Evaluator, FitnessGoal, GeneBounds, Schedule, Selector},
    service::{register_event_handlers, register_job_handlers},
};
use fx_mq_building_blocks::queries::Queries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use sqlx::PgPool;
use std::time::Duration;
use std::{env, sync::Arc};
use uuid::Uuid;

#[derive(Debug)]
struct Cube;

impl Encodeable for Cube {
    const NAME: &'static str = "cube";

    type Phenotype = (i64, i64, i64);

    fn morphology() -> Vec<GeneBounds> {
        vec![GeneBounds::new(0, 1000, 1000).unwrap(); 3]
    }

    fn encode(&self) -> Vec<i64> {
        vec![0, 0, 0] // Initial point, not used since we generate random points
    }

    fn decode(genes: &[i64]) -> Self::Phenotype {
        (genes[0], genes[1], genes[2])
    }
}

struct CubeQuadraticEvaluator;

impl Evaluator<(i64, i64, i64)> for CubeQuadraticEvaluator {
    fn fitness<'a>(
        &self,
        phenotype: (i64, i64, i64),
    ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
            let (x, y, z) = phenotype;
            // Scale from 0..1000 to 0..10 for distance calculation
            let x = x as f64 / 100.0;
            let y = y as f64 / 100.0;
            let z = z as f64 / 100.0;

            // Calculate distance from origin
            let dist = (x * x + y * y + z * z).sqrt();

            // Maximum possible distance in our scaled space (corner to origin)
            let max_distance = (10.0_f64.powi(2) * 3.0).sqrt(); // ~17.32

            // Quadratic fitness: 1 - (normalized_distance)^2
            // This gives more gradual fitness differences
            let normalized_dist = dist / max_distance;
            let fitness = 1.0 - normalized_dist.powi(2);

            // Ensure fitness is always positive
            Ok(fitness.max(0.0))
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::from_filename(".env.local").ok();

    // Initialize logging at DEBUG level
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // Get database URL from environment
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await?;

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
    register_event_handlers(
        Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME)),
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
        register_job_handlers(&service, fx_mq_jobs::RegistryBuilder::new()),
        8,
        host_id,
        hold_for,
    )
    .await?;
    let _jobs_handle = tokio::spawn(async move {
        jobs_listener.listen().await?;
        Ok::<(), anyhow::Error>(())
    });

    // Create optimization request
    service
        .new_optimization_request(
            Cube::NAME,
            Cube::HASH,
            FitnessGoal::maximize(0.99)?,
            Schedule::generational(50, 50),
            Selector::tournament(3, 50),
            0.3, // temperature
            0.1, // mutation_rate
        )
        .await?;

    // Run for a maximum of 5 seconds
    let timeout_duration = Duration::from_secs(30);
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
