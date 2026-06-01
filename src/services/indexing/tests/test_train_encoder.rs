use super::*;
use crate::migrations;
use crate::services::indexing::Registry;
use crate::services::indexing::repositories::encoders;

#[sqlx::test(migrations = false)]
async fn it_trains_an_encoder(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = test_tools::build_context_di(&pool).await?;
    let encoders = ctx.container.get::<encoders::Read>().await?;

    let indexer = test_tools::TestIndexer::default();

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
