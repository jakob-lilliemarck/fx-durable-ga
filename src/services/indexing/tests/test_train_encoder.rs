use crate::migrations;
use crate::services::indexing::tests::test_tools::{NoOpOptimizer, TestContext, TestIndexer};
use crate::services::indexing::{Registry, repositories::encoders};
use crate::test_tools::TestConfig;
use fx_mq_building_blocks::testing_tools::TestQueries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use sqlx::PgPool;
use std::sync::Arc;

async fn setup(pool: &PgPool) -> anyhow::Result<TestContext> {
    let indexer = TestIndexer::default();
    let indexer_id = Registry::get_indexer_id(&indexer)?;
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
async fn it_trains_an_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let encoders = ctx.container.get::<encoders::Read>().await?;

    let indexer = TestIndexer::default();

    let indexer_id = Registry::get_indexer_id(&indexer)?;

    let encoder_digest = ctx
        .app
        .services()
        .indexing()
        .train_encoder(&indexer_id)
        .await?;

    let fetched = encoders.get_encoder(&encoder_digest).await?;

    assert_eq!(fetched.digest, encoder_digest);
    assert_eq!(fetched.shape_in, vec![2]);
    assert_eq!(fetched.shape_out, 2);

    Ok(())
}
