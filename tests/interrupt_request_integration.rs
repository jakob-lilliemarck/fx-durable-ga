use fx_durable_ga::{
    bootstrap, migrations,
    models::{
        Crossover, Distribution, FitnessGoal, Mutagen, MutationRate, Schedule, Selector,
        Temperature,
    },
    register_event_handlers, register_job_handlers,
};
use fx_mq_jobs::Queries;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

const FX_MQ_JOBS_SCHEMA_NAME: &str = "fx_mq_jobs";

#[sqlx::test(migrations = false)]
async fn test_interrupt_request_end_to_end(pool: PgPool) -> anyhow::Result<()> {
    // Run migrations
    migrations::run_default_migrations(&pool).await?;

    // Bootstrap service
    let service = Arc::new(bootstrap(pool.clone()).await?.build());

    // Register event handlers
    let mut registry = fx_event_bus::EventHandlerRegistry::new();
    register_event_handlers(
        Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME)),
        service.clone(),
        &mut registry,
    );

    // Spawn event listener
    let pool_clone = pool.clone();
    tokio::spawn(async move {
        let mut listener = fx_event_bus::Listener::new(pool_clone, registry);
        let _ = listener.listen(None).await;
    });

    // Spawn jobs listener
    let pool_clone = pool.clone();
    let service_clone = service.clone();
    tokio::spawn(async move {
        let host_id = Uuid::nil();
        let mut jobs_listener = fx_mq_jobs::Listener::new(
            pool_clone,
            register_job_handlers(&service_clone, fx_mq_jobs::RegistryBuilder::new()),
            1,
            host_id,
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        let _ = jobs_listener.listen().await;
    });

    // Create a test request
    let request_id = service
        .new_optimization_request(
            "TestType",
            123,
            FitnessGoal::maximize(0.95)?,
            Schedule::generational(100, 10),
            Selector::tournament(3, 20)?,
            Mutagen::new(Temperature::constant(0.5)?, MutationRate::constant(0.3)?),
            Crossover::uniform(0.5)?,
            Distribution::random(5),
            None::<()>,
        )
        .await?;

    // Interrupt the request
    service.interrupt_request(request_id).await?;

    // Wait for event to be processed (with timeout)
    let mut attempts = 0;
    let max_attempts = 3; // 3ms total with 1ms sleep
    loop {
        if service.is_request_concluded(request_id).await? {
            break;
        }
        if attempts >= max_attempts {
            panic!("Timeout waiting for request to be concluded");
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        attempts += 1;
    }

    // Verify the request was concluded
    assert!(
        service.is_request_concluded(request_id).await?,
        "Request should be concluded after interruption"
    );

    Ok(())
}
