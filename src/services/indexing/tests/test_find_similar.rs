use super::super::{Digest, Registry};
use crate::infrastructure::db;
use crate::migrations;
use crate::services::indexing::repositories::embeddings::{
    self, EmbeddingNew, SearchSimilarEmbeddingsFilter, TagNew,
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
async fn it_finds_similar(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;

    let (encoder_digest, _anchor, close, far) = seed(&mut ctx).await?;

    let similar = ctx
        .app
        .repositories()
        .embeddings()
        .find_similar(
            &encoder_digest,
            &SearchSimilarEmbeddingsFilter::default()
                .with_reference_tag("anchor")
                .with_search_tag("cluster"),
            2,
        )
        .await?;

    assert_eq!(similar.len(), 2);
    assert_eq!(&similar[0].0.embedding_id, &close);
    assert_eq!(&similar[1].0.embedding_id, &far);

    Ok(())
}

async fn seed(ctx: &mut TestContext) -> anyhow::Result<(Digest, Uuid, Uuid, Uuid)> {
    let encoder_digest =
        Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000000")
            .unwrap();

    let now = Utc::now();

    let tag = "cluster";
    let anchor_tag = "anchor";

    let build_embedding = |value: f32| {
        let mut vec = [0.0f32; 256];
        vec[0] = value;
        EmbeddingNew::new(encoder_digest, now, vec)
    };

    let anchor = build_embedding(0.5);
    let close = build_embedding(0.55);
    let far = build_embedding(0.9);

    let values = vec![anchor.clone(), close.clone(), far.clone()];

    let tags = vec![
        TagNew::new(anchor_tag, *anchor.id(), now),
        TagNew::new(tag, *close.id(), now),
        TagNew::new(tag, *far.id(), now),
    ];

    let embeddings_wr = ctx.container.get::<embeddings::Write>().await?;

    db::begin(embeddings_wr.clone(), |tx| {
        Box::pin(async move {
            let mut wr = embeddings::WriteTx::new(tx);
            wr.store_embeddings(&values).await?;
            wr.store_tags(&tags).await?;
            Ok(())
        })
    })
    .await?;

    Ok((encoder_digest, *anchor.id(), *close.id(), *far.id()))
}
