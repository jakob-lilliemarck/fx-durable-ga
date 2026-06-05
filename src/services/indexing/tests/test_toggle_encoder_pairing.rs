use crate::migrations;
use crate::services::indexing::tests::test_tools::{NoOpOptimizer, TestContext, TestIndexer};
use crate::services::indexing::{self as indexing, Registry, repositories::encoders};
use crate::test_tools::TestConfig;
use fx_event_bus::test_tools::get_unacknowledged_events;
use fx_mq_building_blocks::testing_tools::TestQueries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use sqlx::PgPool;
use std::sync::Arc;

async fn setup(pool: &PgPool) -> anyhow::Result<TestContext> {
    let indexer = TestIndexer::default();
    let indexer_id = Registry::get_indexer_id(&indexer).await?;
    let mut c = TestConfig::new(pool.clone())
        .with_optimizer(NoOpOptimizer)
        .with_indexer(indexer)
        .build()
        .await?;
    let app = c.get::<Arc<crate::bootstrap::App>>().await?;
    let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
    Ok(TestContext {
        app,
        indexer_id,
        mq,
        container: c,
    })
}

#[sqlx::test(migrations = false)]
async fn it_enabled_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let encoders = ctx.container.get::<encoders::Read>().await?;

    let encoder_digest = super::test_tools::seed_encoder(&mut ctx).await?;

    let events_before = get_unacknowledged_events(&pool).await?;

    let enabled = ctx
        .app
        .services()
        .indexing()
        .enable_encoder(encoder_digest)
        .await?;

    assert!(enabled);

    let encoder_pairings = encoders
        .get_enabled_encoder_digests(&[encoder_digest])
        .await?;

    let found = encoder_pairings
        .iter()
        .find(|digest| **digest == encoder_digest);

    assert!(found.is_some());

    let events_after = get_unacknowledged_events(&pool).await?;

    assert_eq!(events_after, events_before + 1);

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_disabled_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;
    let encoders = ctx.container.get::<encoders::Read>().await?;

    let encoder_digest = super::test_tools::seed_encoder(&mut ctx).await?;

    indexing.enable_encoder(encoder_digest).await?;

    let before_disable_events = get_unacknowledged_events(&pool).await?;

    let disabled = indexing.disable_encoder(encoder_digest).await?;
    assert!(!disabled);

    let encoder_pairings = encoders
        .get_enabled_encoder_digests(&[encoder_digest])
        .await?;
    assert!(encoder_pairings.is_empty());

    let after_disable_events = get_unacknowledged_events(&pool).await?;
    assert_eq!(after_disable_events, before_disable_events);

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_publishes_a_single_event_when_enabled_multiple_times(
    pool: sqlx::PgPool,
) -> anyhow::Result<()> {
    crate::test_tools::init_test_tracing();

    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;

    let encoder_digest = super::test_tools::seed_encoder(&mut ctx).await?;

    let before = get_unacknowledged_events(&pool).await?;

    let first = indexing.enable_encoder(encoder_digest).await?;
    assert!(first);

    let after_first = get_unacknowledged_events(&pool).await?;
    assert_eq!(after_first, before + 1);

    let second = indexing.enable_encoder(encoder_digest).await?;
    assert!(second);

    let after_second = get_unacknowledged_events(&pool).await?;
    assert_eq!(after_second, after_first);

    Ok(())
}
