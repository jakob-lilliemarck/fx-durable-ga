use sqlx::PgPool;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    tracing::info!("Connecting to database...");
    let pool = PgPool::connect(&database_url).await?;

    tracing::info!("Running default migrations...");
    fx_durable_ga::migrations::run_default_migrations(&pool).await?;

    tracing::info!("Migrations completed successfully!");

    Ok(())
}
