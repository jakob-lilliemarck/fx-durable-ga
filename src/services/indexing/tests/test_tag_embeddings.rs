use super::super::Digest;
use super::*;
use crate::infrastructure::db;
use crate::migrations;
use crate::services::indexing::repositories::embeddings::{
    self, EmbeddingNew, SearchEmbeddingsFilter,
};
use crate::services::indexing::tests::test_tools::TestContext;
use chrono::Utc;
use uuid::Uuid;

#[sqlx::test(migrations = false)]
async fn it_tags_embeddings(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = test_tools::build_context_di(&pool).await?;

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
