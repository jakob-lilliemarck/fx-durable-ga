//! # Point Optimization Example
//!
//! This example demonstrates how to use the fx_durable_ga crate to solve an optimization problem
//! using genetic algorithms. We'll optimize 3D coordinates to find points that are closest
//! to a target point using unbounded fitness values (raw distance).
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
    bootstrap,
    models::{
        Crossover, Distribution, Encodeable, Evaluator, FitnessGoal, GeneBounds, Mutagen,
        MutationRate, Schedule, Selector, Temperature, Terminated,
    },
    register_event_handlers, register_job_handlers,
};
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use fx_mq_jobs::Queries;
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use std::{env, sync::Arc};
use uuid::Uuid;

/// A 3D point with x, y, z coordinates.
///
/// This struct defines how we encode/decode between genetic representation (integers)
/// and the actual problem space (floating-point coordinates).
#[derive(Debug, Copy, Clone)]
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
    /// Uses the bounds' encode_f64 method to directly convert coordinates to gene indices.
    fn encode(&self) -> Vec<i64> {
        let bounds = Self::morphology();
        vec![
            bounds[0]
                .encode_f64(self.x)
                .expect("x coordinate within bounds"),
            bounds[1]
                .encode_f64(self.y)
                .expect("y coordinate within bounds"),
            bounds[2]
                .encode_f64(self.z)
                .expect("z coordinate within bounds"),
        ]
    }

    /// Converts genotype (integer genes) to phenotype (Point).
    ///
    /// Maps integer gene values to floating-point coordinates within the defined bounds.
    fn decode(genes: &[i64]) -> Self::Phenotype {
        let bounds = Self::morphology();
        Point {
            x: bounds[0].decode_f64(genes[0]), // Convert to actual decimal coordinates
            y: bounds[1].decode_f64(genes[1]), // e.g., gene 0 → 0.5, gene 999 → 1.75
            z: bounds[2].decode_f64(genes[2]), // e.g., gene 0 → 2.0, gene 999 → 3.25
        }
    }
}

/// Fitness evaluator that finds points closest to the target point.
///
/// Returns the raw distance to target - lower distances are better (minimization problem).
struct PointDistanceEvaluator {
    target_point: Point,
}

impl Evaluator<Point> for PointDistanceEvaluator {
    /// Returns the raw distance to the target point.
    ///
    /// Lower distances are better - this is a minimization problem.
    fn fitness<'a>(
        &self,
        _genotype_id: Uuid,
        phenotype: Point,
        terminated: &'a Box<dyn Terminated>,
    ) -> futures::future::BoxFuture<'a, Result<f64, anyhow::Error>> {
        let target = self.target_point;
        Box::pin(async move {
            if terminated.is_terminated().await {
                return Ok(f64::MAX); // Return worst possible fitness if terminated
            }

            // Calculate direct distance to target point
            let dx = phenotype.x - target.x;
            let dy = phenotype.y - target.y;
            let dz = phenotype.z - target.z;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            Ok(distance) // Raw distance - lower is better
        })
    }
}

const TIMEOUT_SECONDS: u64 = 900;

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
    fx_mq_jobs::run_migrations(&pool, fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME).await?;

    // Bootstrap the optimization service and register our problem type
    let target_point = Point {
        x: 1.0,
        y: 1.5,
        z: 2.5,
    }; // Target point to find
    let evaluator = PointDistanceEvaluator { target_point };
    let service = Arc::new(
        bootstrap(pool.clone())
            .await?
            .register::<Point, _>(evaluator)
            .await?
            .build(),
    );

    // setup event handling and spawn an event handling agent
    let mut registry = fx_event_bus::EventHandlerRegistry::new();
    register_event_handlers(
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
    let hold_for = Duration::from_secs(TIMEOUT_SECONDS);
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

    // Create multiple optimization requests with different genetic algorithm parameters
    for _ in 0..1 {
        service
            .new_optimization_request(
                Point::NAME,
                Point::HASH,
                FitnessGoal::minimize(0.01)?, // Stop when distance ≤ 0.01
                Schedule::generational(200, 30),
                Selector::tournament(7, 100)?,
                Mutagen::new(Temperature::constant(0.7)?, MutationRate::constant(0.3)?),
                Crossover::uniform(0.5)?,
                Distribution::latin_hypercube(1000),
            )
            .await?;
    }

    let timeout_duration = Duration::from_secs(TIMEOUT_SECONDS);
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
