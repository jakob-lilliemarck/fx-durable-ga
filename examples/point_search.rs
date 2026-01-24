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
        Crossover, Distribution, FitnessGoal, GenotypeManager, Mutagen, MutationRate, Schedule,
        Selector, Temperature, Terminated,
    },
    register_event_handlers, register_job_handlers,
};
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use fx_mq_jobs::Queries;
use rand::{Rng, RngCore};
use serde_json::Value;
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use std::{env, sync::Arc};
use uuid::Uuid;

/// JSON-genome manager for 3D point optimization.
/// Genome shape: {"x": f64, "y": f64, "z": f64}
struct PointManager {
    target: Point,
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl GenotypeManager for PointManager {
    fn name(&self) -> &'static str {
        "point"
    }

    fn random(&self, rng: &mut dyn RngCore) -> anyhow::Result<Value> {
        let x = rng.random_range(0.5..1.75);
        let y = rng.random_range(0.75..2.0);
        let z = rng.random_range(2.0..3.25);
        Ok(serde_json::json!({ "x": x, "y": y, "z": z }))
    }

    fn crossover(
        &self,
        parent1: &Value,
        parent2: &Value,
        _rng: &mut dyn RngCore,
    ) -> anyhow::Result<Value> {
        let x = (parent1["x"].as_f64().unwrap_or(0.0) + parent2["x"].as_f64().unwrap_or(0.0)) / 2.0;
        let y = (parent1["y"].as_f64().unwrap_or(0.0) + parent2["y"].as_f64().unwrap_or(0.0)) / 2.0;
        let z = (parent1["z"].as_f64().unwrap_or(0.0) + parent2["z"].as_f64().unwrap_or(0.0)) / 2.0;
        Ok(serde_json::json!({ "x": x, "y": y, "z": z }))
    }

    fn mutate(
        &self,
        genotype: &mut Value,
        rng: &mut dyn RngCore,
        mutation_rate: f64,
        temperature: f64,
    ) -> anyhow::Result<()> {
        let maybe_mutate =
            |v: &mut Value, lo: f64, hi: f64, rng: &mut dyn RngCore, rate: f64, temp: f64| {
                if rng.random_range(0.0..1.0) < rate {
                    let span = (hi - lo) * temp.max(0.05);
                    let delta = rng.random_range(-span..span);
                    if let Some(f) = v.as_f64() {
                        *v = Value::from((f + delta).clamp(lo, hi));
                    }
                }
            };

        if let Some(obj) = genotype.as_object_mut() {
            if let Some(x) = obj.get_mut("x") {
                maybe_mutate(x, 0.5, 1.75, rng, mutation_rate, temperature);
            }
            if let Some(y) = obj.get_mut("y") {
                maybe_mutate(y, 0.75, 2.0, rng, mutation_rate, temperature);
            }
            if let Some(z) = obj.get_mut("z") {
                maybe_mutate(z, 2.0, 3.25, rng, mutation_rate, temperature);
            }
        }
        Ok(())
    }

    fn evaluate<'a>(
        &'a self,
        genotype: &'a Value,
        terminated: &'a dyn Terminated,
    ) -> futures::future::BoxFuture<'a, anyhow::Result<f64>> {
        let target = self.target;
        let clone = genotype.clone();
        Box::pin(async move {
            if terminated.is_terminated().await {
                return Ok(f64::MAX);
            }
            let x = clone["x"].as_f64().unwrap_or(0.0);
            let y = clone["y"].as_f64().unwrap_or(0.0);
            let z = clone["z"].as_f64().unwrap_or(0.0);
            let dx = x - target.x;
            let dy = y - target.y;
            let dz = z - target.z;
            Ok((dx * dx + dy * dy + dz * dz).sqrt())
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
    };
    let manager = PointManager {
        target: target_point,
    };
    let service = Arc::new(
        bootstrap(pool.clone())
            .await?
            .with_genotype_manager(manager)
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
    let request_id = service
        .new_optimization_request(
            "point",
            const_fnv1a_hash::fnv1a_hash_str_32("point") as i32,
            FitnessGoal::minimize(0.1)?, // Stop when distance ≤ to this value
            Schedule::generational(200, 30),
            Selector::tournament(7, 100)?,
            Mutagen::new(Temperature::constant(0.7)?, MutationRate::constant(0.3)?),
            Crossover::uniform(0.5)?,
            Distribution::latin_hypercube(1000),
            None::<()>,
        )
        .await?;

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

        // Exit when the request is concluded
        if service.is_request_concluded(request_id).await? {
            if let Some((best, fitness)) = service.get_best_genotype(request_id).await? {
                println!(
                    "Request completed. Best genotype: {} with fitness {:.6}",
                    best.id(),
                    fitness
                );
            } else {
                println!("Request completed but no genotype results were recorded.");
            }
            break;
        }
    }

    Ok(())
}
