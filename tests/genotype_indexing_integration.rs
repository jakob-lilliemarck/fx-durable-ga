use anyhow::Context;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_ndarray::NdArray;
use fx_durable_ga::services::indexing::EmbeddingCreatedEvent;
use fx_durable_ga::services::indexing::encoder::lstm::{
    AutoencoderConfig, AutoencoderModel, LstmAutoencoder,
};
use fx_durable_ga::{
    bootstrap::ApplicationBuilder,
    migrations,
    models::{EncodeInput, GenotypeIndexer, TypeName},
    services::{genotype_indexing, indexing::ModelConfig, optimization::GenotypeEvaluatedEvent},
};
use fx_mq_jobs::Queries;
use serde_json::{Map, Value, json};
use sqlx::{
    PgPool, PgTransaction,
    postgres::{PgConnectOptions, PgPoolOptions},
    Row,
};
use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
    sync::Arc,
    time::Duration,
};
use tokio::sync::{Mutex, oneshot};
use tracing::Level;
use uuid::Uuid;

const FX_MQ_JOBS_SCHEMA_NAME: &str = "fx_mq_jobs";
const GENOTYPE_ID: &str = "00000000-0000-0000-0000-000000000001";
const GENERATION_ID: i32 = 1;
const TIMEOUT_MS: u64 = 5_000;

#[sqlx::test(migrations = false)]
async fn genotype_indexing_end_to_end(
    pool_opts: PgPoolOptions,
    connect_opts: PgConnectOptions,
) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    let pool = pool_opts
        .clone()
        .max_connections(12)
        .connect_with(connect_opts.clone())
        .await
        .context("connect postgres")?;

    migrations::run_default_migrations(&pool)
        .await
        .context("apply migrations")?;

    let app_builder = ApplicationBuilder::default().with_pool(pool.clone());

    let indexing_service = app_builder.indexing_service().build();

    let encoder_id = seed_encoder(&pool).await?;
    indexing_service
        .enable_encoder_pairing(&encoder_id, TestIndexer::TYPE_HASH)
        .await?;

    let indexing_service = Arc::new(indexing_service);

    let genotype_indexing_svc = Arc::new(
        app_builder
            .genotype_indexing_service(indexing_service.clone())
            .with_indexable(TestIndexer)
            .build(),
    );

    let genotype_id = Uuid::parse_str(GENOTYPE_ID)?;

    let request_id = seed_request_data(&pool, &genotype_id, GENERATION_ID).await?;

    let queries = Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME));

    let mut event_handlers = fx_event_bus::EventHandlerRegistry::new();
    genotype_indexing::register_event_handlers(queries.clone(), &mut event_handlers);

    // register a test-local handler that listens to GenotypeIndexedEvent
    // and sets a semaphore when the expected genotype has been indexed
    let (tx, rx) = oneshot::channel::<()>();
    let done = Arc::new(Mutex::new(Some(tx)));
    event_handlers.with_handler(GenotypeIndexedHandler { done: done.clone() });

    let event_pool = pool.clone();
    let event_handle = tokio::spawn(async move {
        let mut listener = fx_event_bus::Listener::new(event_pool, event_handlers);
        let _ = listener.listen(None).await;
    });

    // Jobs
    let jobs_pool = pool.clone();
    let jobs_service = genotype_indexing_svc.clone();
    let job_registry = genotype_indexing::register_job_handlers(
        &jobs_service,
        fx_mq_jobs::RegistryBuilder::new(),
        queries.clone(),
    );
    let jobs_handle = tokio::spawn(async move {
        let mut listener = fx_mq_jobs::Listener::new(
            jobs_pool,
            job_registry,
            1,
            Uuid::nil(),
            Duration::from_millis(100),
        )
        .await
        .expect("start jobs listener");
        let _ = listener.listen().await;
    });

    publish_evaluated_event(&pool, request_id, genotype_id).await?;

    tokio::time::timeout(Duration::from_millis(TIMEOUT_MS), rx)
        .await
        .context("timed out waiting for genotype indexing")??;

    event_handle.abort();
    let _ = event_handle.await;
    jobs_handle.abort();
    let _ = jobs_handle.await;

    let record = sqlx::query(
        r#"
        SELECT id, encoded_with
        FROM fx_durable_ga.embeddings
        WHERE encoded_with = $1
        ORDER BY encoded_at DESC
        LIMIT 1
        "#,
    )
    .bind(encoder_id)
    .fetch_one(&pool)
    .await
    .context("fetch indexed embedding")?;

    let embedding_id: Uuid = record.try_get("id")?;
    let encoded_with: Uuid = record.try_get("encoded_with")?;
    assert_eq!(encoded_with, encoder_id, "embedding encoded with unexpected encoder");

    let rows = sqlx::query(
        r#"
        SELECT tag_name
        FROM fx_durable_ga.embedding_tags
        WHERE embedding_id = $1
        ORDER BY tag_name
        "#,
    )
    .bind(embedding_id)
    .fetch_all(&pool)
    .await
    .context("fetch embedding tags")?;

    let mut actual_tags: Vec<String> = rows
        .into_iter()
        .map(|row| row.try_get("tag_name"))
        .collect::<Result<_, _>>()?;
    actual_tags.sort();

    let mut expected_tags = vec![
        "type:Genotype".to_string(),
        format!("encoder_id:{}", encoder_id),
        format!("type_name:{}", TestIndexer::TYPE_NAME),
        format!("type_hash:{}", TestIndexer::TYPE_HASH),
        format!("context_hash:{}", TestIndexer::CONTEXT_HASH),
    ];
    expected_tags.sort();

    assert_eq!(actual_tags, expected_tags, "embedding tags mismatch");

    Ok(())
}

struct GenotypeIndexedHandler {
    done: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl fx_event_bus::Handler<EmbeddingCreatedEvent> for GenotypeIndexedHandler {
    type Error = fx_mq_jobs::PublishError;

    fn handle<'a>(
        &'a self,
        _: Arc<EmbeddingCreatedEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        let done = self.done.clone();

        Box::pin(async move {
            if let Some(sender) = done.lock().await.take() {
                let _ = sender.send(());
            }
            (tx, Ok(()))
        })
    }
}

async fn seed_request_data(
    pool: &PgPool,
    genotype_id: &Uuid,
    generation_id: i32,
) -> anyhow::Result<Uuid> {
    let request_id = Uuid::now_v7();
    let requested_at = chrono::Utc::now();
    let goal = json!({ "Maximize": { "threshold": 0.9 } });
    let schedule = json!({
        "max_evaluations": 10,
        "population_size": 2,
        "selection_interval": 1
    });
    let selector = json!({ "method": { "Tournament": { "size": 2 } } });
    let user_defined = json!({ "Uniform": { "probability": 1.0 } });

    sqlx::query(
        r#"
        INSERT INTO fx_durable_ga.requests (
            id,
            requested_at,
            type_name,
            type_hash,
            goal,
            schedule,
            selector,
            user_defined,
            data
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NULL)
        "#,
    )
    .bind(request_id)
    .bind(requested_at)
    .bind(TestIndexer::TYPE_NAME)
    .bind(TestIndexer::TYPE_HASH)
    .bind(goal)
    .bind(schedule)
    .bind(selector)
    .bind(user_defined)
    .execute(pool)
    .await
    .context("insert request")?;

    let genome = json!([1.0, 2.0]);
    let genome_hash = compute_genome_hash(&genome);
    let generated_at = chrono::Utc::now();

    sqlx::query(
        r#"
        INSERT INTO fx_durable_ga.genotypes (
            id,
            generated_at,
            type_name,
            type_hash,
            genome,
            genome_hash,
            request_id,
            generation_id,
            parent_a,
            parent_b
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NULL,NULL)
        "#,
    )
    .bind(genotype_id)
    .bind(generated_at)
    .bind(TestIndexer::TYPE_NAME)
    .bind(TestIndexer::TYPE_HASH)
    .bind(genome)
    .bind(genome_hash)
    .bind(request_id)
    .bind(generation_id)
    .execute(pool)
    .await
    .context("insert genotype")?;

    sqlx::query(
        r#"
        INSERT INTO fx_durable_ga.evaluations (
            genotype_id,
            fitness,
            started_at,
            completed_at,
            evaluated_by,
            copied_from
        ) VALUES ($1,$2,$3,$4,$5,$6)
        "#,
    )
    .bind(genotype_id)
    .bind(0.5_f64)
    .bind(Some(chrono::Utc::now()))
    .bind(Some(chrono::Utc::now()))
    .bind(Some(Uuid::now_v7()))
    .bind(Option::<Uuid>::None)
    .execute(pool)
    .await
    .context("insert evaluation")?;

    Ok(request_id)
}

async fn seed_encoder(pool: &PgPool) -> anyhow::Result<Uuid> {
    type TestBackend = NdArray<f32>;
    let id = Uuid::now_v7();
    let config = AutoencoderConfig {
        input_size: 2,
        hidden_size: 4,
        latent_size: 2,
    };
    let model_config = ModelConfig::Lstm(config);

    let device = <TestBackend as Backend>::Device::default();
    let model = LstmAutoencoder::<TestBackend>::new(&device, config);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let model_bytes = Recorder::<TestBackend>::record(&recorder, model.into_record(), ())?;
    let checksum = vec![1_u8, 2, 3, 4];
    let trained_at = chrono::Utc::now();
    let shape_in = vec![2];
    let shape_out = 2;

    sqlx::query(
        r#"
        INSERT INTO fx_durable_ga.encoders (
            id,
            model_type,
            model_config,
            model_weights,
            model_format,
            shape_in,
            shape_out,
            trained_at,
            trained_on_checksum
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        "#,
    )
    .bind(id)
    .bind("lstm")
    .bind(serde_json::to_value(&model_config).expect("serialize config"))
    .bind(model_bytes)
    .bind("burn-bin-f32")
    .bind(&shape_in)
    .bind(shape_out)
    .bind(trained_at)
    .bind(checksum)
    .execute(pool)
    .await
    .context("insert encoder")?;

    Ok(id)
}

async fn publish_evaluated_event(
    pool: &PgPool,
    request_id: Uuid,
    genotype_id: Uuid,
) -> anyhow::Result<()> {
    let tx = pool.begin().await?;
    let mut publisher = fx_event_bus::Publisher::new(tx);
    publisher
        .publish(GenotypeEvaluatedEvent::new(request_id, genotype_id))
        .await?;
    let tx: PgTransaction<'_> = publisher.into();
    tx.commit().await?;
    Ok(())
}

struct TestIndexer;

impl TestIndexer {
    const TYPE_NAME: &'static str = "TestIndexer";
    const TYPE_HASH: i32 = 9_001;
    const CONTEXT_HASH: &'static str = "test-context";
}

impl TypeName for TestIndexer {
    fn type_name(&self) -> &'static str {
        Self::TYPE_NAME
    }

    fn type_hash(&self) -> i32 {
        Self::TYPE_HASH
    }
}

impl GenotypeIndexer for TestIndexer {
    fn input(&self, genotype: &fx_durable_ga::models::Genotype) -> EncodeInput {
        let values: Vec<f32> = match genotype.genome() {
            Value::Array(items) => items
                .iter()
                .filter_map(|v| v.as_f64())
                .map(|v| v as f32)
                .collect(),
            _ => vec![0.0, 0.0],
        };
        let dimensions = vec![1, values.len().max(1)];
        EncodeInput { values, dimensions }
    }

    fn context_hash(&self) -> String {
        Self::CONTEXT_HASH.to_string()
    }
}

fn compute_genome_hash(genome: &Value) -> i64 {
    fn canonicalize(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut ordered = BTreeMap::new();
                for (k, v) in map {
                    ordered.insert(k.clone(), canonicalize(v));
                }
                let mut new_map = Map::new();
                for (k, v) in ordered {
                    new_map.insert(k, v);
                }
                Value::Object(new_map)
            }
            Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
            _ => value.clone(),
        }
    }

    let canonical = canonicalize(genome);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.to_string().hash(&mut hasher);
    hasher.finish() as i64
}
