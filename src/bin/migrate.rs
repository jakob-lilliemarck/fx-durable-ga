use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use sqlx::postgres::PgPoolOptions;
use tracing::Level;

// Use this binary to run all required migrations locally.
// Failing to run migrations will cause compilation issues as sqlx uses the schema to assert typing in queries
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_jobs::run_migrations(&pool, FX_MQ_JOBS_SCHEMA_NAME).await?;
    fx_durable_ga::migrations::run_migrations(&pool).await?;

    Ok(())
}
