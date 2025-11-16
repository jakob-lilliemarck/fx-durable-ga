use sqlx::{Acquire, PgPool, Postgres};
use tracing::instrument;

// Embed the migrations directory at compile time
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!();

/// Runs database migrations for the fx_durable_ga schema.
///
/// Creates the 'fx_durable_ga' schema if it doesn't exist and runs all
/// embedded migrations within that schema.
///
/// # Arguments
///
/// * `conn` - Database connection or connection pool
///
/// # Errors
///
/// Returns `sqlx::Error` if schema creation or migration execution fails.
#[instrument(level = "debug", skip(conn))]
pub async fn run_migrations<'a, A>(conn: A) -> Result<(), sqlx::Error>
where
    A: Acquire<'a, Database = Postgres>,
{
    let mut tx = conn.begin().await?;

    // Ensure the 'fx_durable_ga' schema exists
    sqlx::query("CREATE SCHEMA IF NOT EXISTS fx_durable_ga;")
        .execute(&mut *tx)
        .await?;

    // Temporarily set search_path for this transaction
    sqlx::query("SET LOCAL search_path TO fx_durable_ga;")
        .execute(&mut *tx)
        .await?;

    // Run migrations within the 'fx_durable_ga' schema
    MIGRATOR.run(&mut *tx).await?;

    tx.commit().await?;

    Ok(())
}

/// Runs migrations for fx-event-bus, fx-mq-jobs and fx-durbale-ga in a single transaction
/// It's biased about the schema used by fx-mq-jobs. If you want to use another schema for fx-mq-jobs,
/// then compose the separate migrations from each crate instead.
#[instrument(level = "debug", skip(pool))]
pub async fn run_default_migrations(pool: &PgPool) -> anyhow::Result<()> {
    let mut tx = pool.begin().await?;

    fx_event_bus::run_migrations(&mut tx).await?;
    fx_mq_jobs::run_migrations(&mut tx, fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME).await?;
    run_migrations(&mut tx).await?;

    tx.commit().await?;

    Ok(())
}
