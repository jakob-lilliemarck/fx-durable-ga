//! # Point Optimization Example
//!
//! This example demonstrates how to use the fx_durable_ga crate to solve an optimization problem
//! using genetic algorithms. We'll optimize 3D coordinates to find points that are at a specific
//! target distance from the origin.
//!
//! ## Key Concepts
//!
//! - **Genotype**: The encoded representation (genes as integers)
//! - **Phenotype**: The decoded representation (3D coordinates as floats)
//! - **Fitness**: How good a solution is (higher = better)
//! - **Population**: A collection of candidate solutions
//! - **Generation**: One iteration of the evolutionary process

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
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use std::{env, sync::Arc};
use uuid::Uuid;

/// A 3D point with x, y, z coordinates.
///
/// This struct defines how we encode/decode between genetic representation (integers)
/// and the actual problem space (floating-point coordinates).
#[derive(Debug)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Encodeable for Point {
    const NAME: &'static str = "point";

    type Phenotype = Point;

    /// Defines the search space for each dimension.
    ///
    /// Each GeneBounds specifies the range and precision for one coordinate:
    /// - x: 0.5 to 1.75 with 0.001 precision
    /// - y: 0.75 to 2.0 with 0.001 precision
    /// - z: 2.0 to 3.25 with 0.001 precision
    fn morphology() -> Vec<GeneBounds> {
        vec![
            GeneBounds::decimal(0.5, 1.75, 1000, 3).unwrap(), // x: 0.500 to 1.750 (scaled: 500 to 1750)
            GeneBounds::decimal(0.75, 2.00, 1000, 3).unwrap(), // y: 0.750 to 2.000 (scaled: 750 to 2000)
            GeneBounds::decimal(2.00, 3.25, 1000, 3).unwrap(), // z: 2.000 to 3.250 (scaled: 2000 to 3250)
        ]
    }

    /// Converts this point to genetic representation.
    ///
    /// Maps floating-point coordinates back to normalized [0,1] samples, then to gene indices.
    fn encode(&self) -> Vec<i64> {
        let bounds = Self::morphology();

        // Map coordinates to normalized [0,1] samples within each dimension's range
        let x_normalized = (self.x - 0.5) / (1.75 - 0.5); // x range: 0.5 to 1.75
        let y_normalized = (self.y - 0.75) / (2.00 - 0.75); // y range: 0.75 to 2.0
        let z_normalized = (self.z - 2.00) / (3.25 - 2.00); // z range: 2.0 to 3.25

        vec![
            bounds[0].from_sample(x_normalized.clamp(0.0, 1.0)),
            bounds[1].from_sample(y_normalized.clamp(0.0, 1.0)),
            bounds[2].from_sample(z_normalized.clamp(0.0, 1.0)),
        ]
    }

    /// Converts genotype (integer genes) to phenotype (Point).
    ///
    /// Maps integer gene values to floating-point coordinates within the defined bounds.
    fn decode(genes: &[i64]) -> Self::Phenotype {
        let bounds = Self::morphology();
        Point {
            x: bounds[0].to_f64(genes[0]), // Convert to actual decimal coordinates
            y: bounds[1].to_f64(genes[1]), // e.g., gene 0 → 0.5, gene 999 → 1.75
            z: bounds[2].to_f64(genes[2]), // e.g., gene 0 → 2.0, gene 999 → 3.25
        }
    }
}

/// Fitness evaluator that rewards points at a specific distance from the origin.
///
/// The target distance is ~2.1943 (distance from origin to point (0.5, 0.75, 2.0)).
/// Points closer to this target distance receive higher fitness scores.
struct PointDistanceEvaluator;

impl Evaluator<Point> for PointDistanceEvaluator {
    /// Calculates fitness based on how close a point's distance from origin matches the target.
    ///
    /// Higher fitness (closer to 1.0) means the point is closer to the target distance.
    fn fitness<'a>(
        &self,
        phenotype: Point,
        terminated: &'a Box<dyn Terminated>,
    ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
            if terminated.is_terminated().await {
                // abort
            }

            // Calculate distance from origin
            let dist =
                (phenotype.x * phenotype.x + phenotype.y * phenotype.y + phenotype.z * phenotype.z)
                    .sqrt();

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

    // Initialize logging to see optimization progress
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Database setup - genetic algorithms need persistent storage for populations
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(50)
        .connect(&database_url)
        .await?;

    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_building_blocks::migrator::run_migrations(&pool, fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME)
        .await?;

    // Bootstrap the optimization service and register our problem type
    let service = Arc::new(
        bootstrap(pool.clone())
            .await?
            .register::<Point, _>(PointDistanceEvaluator)
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

    // Create multiple optimization requests with different genetic algorithm parameters
    for _ in 0..20 {
        service
            .new_optimization_request(
                Point::NAME,
                Point::HASH,
                FitnessGoal::maximize(0.99)?, // Stop when fitness reaches 99%
                Schedule::generational(100, 10), // 100 generations, 10 parallel
                Selector::tournament(3, 50),  // Tournament selection (size 3, pop 50)
                Mutagen::new(
                    Temperature::exponential(0.8, 0.4, 0.8, 2)?, // Cooling schedule
                    MutationRate::exponential(0.5, 0.3, 0.8, 2)?, // Adaptive mutation rate
                ),
                Crossover::uniform(0.5)?, // 50% uniform crossover rate
                Distribution::random(50), // Random initial population of 50
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
