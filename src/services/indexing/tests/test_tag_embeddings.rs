use super::super::{Digest, Registry};
use crate::infrastructure::db;
use crate::migrations;
use crate::services::indexing::repositories::embeddings::{
    self, EmbeddingNew, SearchEmbeddingsFilter,
};
use crate::services::indexing::tests::test_tools::{NoOpOptimizer, TestContext, TestIndexer};
use crate::test_tools::TestConfig;
use chrono::Utc;
use fx_mq_building_blocks::testing_tools::TestQueries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

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
async fn it_tags_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;

    let embedding_id = seed(&mut ctx).await?;

    let tag = "favorite";

    ctx.app
        .services()
        .indexing()
        .tag_embeddings(&[embedding_id], tag)
        .await?;

    let found = ctx
        .app
        .repositories()
        .embeddings()
        .search_embeddings(&SearchEmbeddingsFilter::default().with_tag(tag), 10)
        .await?;

    assert_eq!(found.len(), 1);

    let expectations = [(found[0].embedding_id, [tag])];

    for (i, (embedding_id, tags)) in expectations.iter().enumerate() {
        assert_eq!(&found[i].embedding_id, embedding_id);
        for tag in tags {
            assert!(found[i].tags.contains(&tag.to_string()));
        }
    }

    Ok(())
}

async fn seed(ctx: &mut TestContext) -> anyhow::Result<Uuid> {
    let encoder_digest =
        Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
            .unwrap();

    let embedding = EmbeddingNew::new(encoder_digest, Utc::now(), [0.0; 256]);

    let embedding_id = *embedding.id();

    let embeddings_wr = ctx.container.get::<embeddings::Write>().await?;

    db::begin(embeddings_wr.clone(), |tx| {
        Box::pin(async move {
            let mut wr = embeddings::WriteTx::new(tx);
            wr.store_embeddings(&[embedding]).await?;
            Ok(())
        })
    })
    .await?;

    Ok(embedding_id)
}
