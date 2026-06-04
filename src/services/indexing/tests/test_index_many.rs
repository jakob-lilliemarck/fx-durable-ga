use super::super::{Digest, Registry};
use crate::infrastructure::db;
use crate::services::indexing::repositories::embeddings::{
    self, RequestedEmbedding, SearchEmbeddingsFilter, SearchRequestedEmbeddingsFilter,
};
use crate::services::indexing::tests::test_tools::{
    NoOpOptimizer, TestContext, TestIndexable, TestIndexer,
};
use crate::test_tools::TestConfig;
use crate::{
    migrations,
    services::indexing::{self, jobs::TrainEncoderMessage, repositories::encoders},
};
use chrono::Utc;
use fx_event_bus::test_tools::get_unacknowledged_events;
use fx_mq_building_blocks::testing_tools::TestQueries;
use fx_mq_jobs::{FX_MQ_JOBS_SCHEMA_NAME, Message};
use sqlx::PgPool;
use std::{collections::HashSet, sync::Arc};
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
async fn it_indexes_many(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;

    let encoder_id = super::test_tools::seed_encoder(&mut ctx).await?;

    let encodable = vec![
        (
            TestIndexable::new((1.0, 2.0)),
            vec!["type:Genotype".to_string(), "context:test_1".to_string()],
        ),
        (
            TestIndexable::new((3.0, 4.0)),
            vec![
                "type:Genotype".to_string(),
                "context:test_2".to_string(),
                "favorite".to_string(),
            ],
        ),
    ];

    let events_before = get_unacknowledged_events(&pool).await?;

    let embedding_ids = ctx
        .app
        .services()
        .indexing()
        .index_many(&encoder_id, &encodable, None)
        .await?
        .expect("expected some embedding ids");

    assert_eq!(
        embedding_ids.len(),
        encodable.len(),
        "Expected one embedding id per encodable item; expected: {:?}, got {:?}",
        encodable.len(),
        embedding_ids.len()
    );

    let events_after = get_unacknowledged_events(&pool).await?;

    assert_eq!(
        events_after - events_before,
        embedding_ids.len() as i64,
        "Expected one event per encodable item; expected: {:?}, got {:?}",
        embedding_ids.len() as i64,
        events_after - events_before,
    );

    let found = ctx
        .app
        .repositories()
        .embeddings()
        .search_embeddings(
            &SearchEmbeddingsFilter::default().with_encoder_id(&encoder_id),
            10,
        )
        .await?;

    let unique_embedding_ids: HashSet<Uuid> = found
        .iter()
        .map(|embedding| embedding.embedding_id)
        .collect();

    assert_eq!(
        unique_embedding_ids.len(),
        encodable.len(),
        "Expected 2 unique embedding IDs; expected: {:?}, got {:?}",
        encodable.len(),
        unique_embedding_ids
    );

    let mut expected_tag_sets: Vec<Vec<String>> = encodable
        .iter()
        .map(|(_, tags)| {
            let mut sorted_tags = tags.clone();
            sorted_tags.sort();
            sorted_tags
        })
        .collect();
    expected_tag_sets.sort();

    let mut actual_tag_sets: Vec<Vec<String>> = found
        .iter()
        .map(|embedding| {
            let mut sorted_tags = embedding.tags.clone();
            sorted_tags.sort();
            sorted_tags
        })
        .collect();
    actual_tag_sets.sort();

    assert_eq!(
        actual_tag_sets, expected_tag_sets,
        "Tag sets don't match; expected: {:?}, got: {:?}",
        expected_tag_sets, actual_tag_sets
    );

    let mut tx = pool.begin().await?;
    let jobs = ctx.mq.get_all_messages(&mut tx).await?;

    assert_eq!(
        jobs.len(),
        0,
        "Expected no jobs to have been dispatched; got: {:?}",
        jobs.len()
    );

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_returns_none_if_no_indexer_is_registered(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let ctx = setup(&pool).await?;

    let missing_encoder_digest =
        Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000001")
            .unwrap();

    assert!(
        ctx.indexer_id != missing_encoder_digest,
        "Expected the missing encoder digest to be different from the seeded indexer digest"
    );

    let encodable = vec![(TestIndexable::new((5.0, 6.0)), Vec::new())];

    let result = ctx
        .app
        .services()
        .indexing()
        .index_many(&missing_encoder_digest, &encodable, None)
        .await?;

    assert!(result.is_none(), "Expected None; got: {:?}", result);

    let deferred = ctx
        .app
        .repositories()
        .embeddings()
        .search_requested_embeddings(
            &SearchRequestedEmbeddingsFilter::default().with_indexer_id(&missing_encoder_digest),
            10,
        )
        .await?;

    assert_eq!(
        deferred.len(),
        0,
        "Expected deferred item count to be 0; got: {:?}",
        deferred.len()
    );

    let mut tx = pool.begin().await?;
    let jobs = ctx.mq.get_all_messages(&mut tx).await?;

    assert_eq!(
        jobs.len(),
        0,
        "Expected no jobs to have been dispatched; got: {:?}",
        jobs.len()
    );

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_defers_work_on_unavailable_encoder_and_dispatches_a_training_job(
    pool: sqlx::PgPool,
) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;
    let embeddings = ctx.container.get::<embeddings::Read>().await?;

    let encodable = vec![(TestIndexable::new((5.0, 6.0)), Vec::new())];

    let result = indexing
        .index_many(&ctx.indexer_id, &encodable, None)
        .await?;

    assert!(result.is_none(), "Expected None; got: {:?}", result);

    let deferred = embeddings
        .search_requested_embeddings(
            &SearchRequestedEmbeddingsFilter::default().with_indexer_id(&ctx.indexer_id),
            10,
        )
        .await?;

    assert_eq!(
        deferred.len(),
        1,
        "Expected deferred item count to be 1; got: {:?}",
        deferred.len()
    );

    let mut tx = pool.begin().await?;
    let jobs = ctx.mq.get_all_messages(&mut tx).await?;

    assert_eq!(
        jobs.len(),
        1,
        "Expected exactly 1 job to have been dispatched; got: {:?}",
        jobs.len()
    );

    assert_eq!(
        jobs[0].name,
        TrainEncoderMessage::NAME,
        "Expected job name did not match; expected {:?}, got{:?}",
        TrainEncoderMessage::NAME,
        jobs[0]
    );

    let expected_payload = serde_json::to_value(TrainEncoderMessage::new(ctx.indexer_id))
        .expect("Expected payload to serialize to json");

    assert_eq!(
        jobs[0].payload, expected_payload,
        "Expected job payload did not match; expected {:?}, got{:?}",
        expected_payload, jobs[0].payload
    );
    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_does_not_dispatch_training_jobs_if_there_is_an_availability_record(
    pool: sqlx::PgPool,
) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;
    let encoders_wr = ctx.container.get::<encoders::Write>().await?;
    let embeddings = ctx.container.get::<embeddings::Read>().await?;

    // Flag the encoder as "NotAvailable", indicating that it is known to the system but that it can not currently be used.
    db::begin(encoders_wr.clone(), |tx| {
        Box::pin(async move {
            let mut wr = encoders::WriteTx::new(tx);
            wr.store_encoder_availability(&ctx.indexer_id, false)
                .await?;
            Ok(())
        })
    })
    .await?;

    let encodable = vec![(TestIndexable::new((5.0, 6.0)), Vec::new())];

    let result = indexing
        .index_many(&ctx.indexer_id, &encodable, None)
        .await?;

    assert!(result.is_none(), "Expected None; got: {:?}", result);

    let deferred = embeddings
        .search_requested_embeddings(
            &SearchRequestedEmbeddingsFilter::default().with_indexer_id(&ctx.indexer_id),
            10,
        )
        .await?;

    assert_eq!(
        deferred.len(),
        1,
        "Expected deferred item count to be 1; got: {:?}",
        deferred.len()
    );

    let mut tx = pool.begin().await?;
    let jobs = ctx.mq.get_all_messages(&mut tx).await?;

    assert_eq!(
        jobs.len(),
        0,
        "Expected no jobs to have been dispatched; got: {:?}",
        jobs.len()
    );

    Ok(())
}

#[ignore]
#[sqlx::test(migrations = false)]
async fn it_does_not_race_while_dispatching_training_jobs(
    _pool: sqlx::PgPool,
) -> anyhow::Result<()> {
    // FIXME!
    // Assert that the training job dispatch can **never** race!
    unimplemented!()
}

#[sqlx::test(migrations = false)]
async fn get_indexer_ids_of_type_returns_registered_indexers(
    pool: sqlx::PgPool,
) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let expected = Registry::get_indexer_id(&TestIndexer::default())?;
    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;

    let ids = indexing.get_indexer_ids_of_type("test::indexable").await;

    assert_eq!(ids, vec![expected]);

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn get_indexer_ids_of_type_returns_empty_for_unknown_type(
    pool: sqlx::PgPool,
) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;

    let ids = indexing.get_indexer_ids_of_type("unknown::type").await;

    assert!(ids.is_empty());

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn get_deferred_items_returns_pending_items(pool: sqlx::PgPool) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;
    let embeddings_wr = ctx.container.get::<embeddings::Write>().await?;

    let expected_entity_id = Uuid::parse_str("00000000-0000-0000-0000-00000000a101")?;
    let requests = vec![RequestedEmbedding::new(
        expected_entity_id,
        "genotype".to_string(),
        Digest::from_hex("0000000000000000000000000000000000000000000000000000000000000010")?,
        serde_json::Value::Object(Default::default()),
        Utc::now(),
    )];

    db::begin(embeddings_wr.clone(), |tx| {
        Box::pin(async move {
            let mut wr = embeddings::WriteTx::new(tx);
            wr.store_requested_embeddings(&requests).await?;
            Ok(())
        })
    })
    .await?;

    let filter = SearchRequestedEmbeddingsFilter::default();
    let deferred = indexing.get_deferred_items(&filter, 10).await?;

    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred[0].entity_id, expected_entity_id);

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn get_deferred_items_returns_empty_when_none_pending(
    pool: sqlx::PgPool,
) -> anyhow::Result<()> {
    migrations::run_default_migrations(&pool).await?;

    let mut ctx = setup(&pool).await?;
    let indexing = ctx.container.get::<Arc<indexing::Service>>().await?;

    let filter = SearchRequestedEmbeddingsFilter::default();
    let deferred = indexing.get_deferred_items(&filter, 10).await?;

    assert!(deferred.is_empty());

    Ok(())
}
