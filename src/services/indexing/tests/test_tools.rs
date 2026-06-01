use super::super::Digest;
use super::super::repositories::encoders::Encoder;
use crate::infrastructure::db;
use crate::infrastructure::di::{Container, InvokeError};
use crate::repositories::genotypes::{Identifiable, TypeName};
use crate::services::indexing::EncodeInput;
use crate::services::indexing::repositories::encoders;
use crate::services::indexing::{self as indexable, Registry};
use crate::services::indexing::{
    encoder::lstm::{self, AutoencoderConfig, AutoencoderModel},
    service::MODEL_FORMAT,
};
use crate::services::optimization::{self as foreign_service, OptimizerRegistry};
use crate::{
    bootstrap::{self, App},
    configuration,
    services::indexing::{
        TrainModelConfig,
        encoder::{
            dataset::{SequenceDataSource, SequenceDataset, SequenceSample},
            train::AutoencoderTrainConfig,
        },
    },
};
use burn::prelude::Backend;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_ndarray::NdArray;
use chrono::Utc;
use futures::lock::Mutex;
use fx_mq_building_blocks::testing_tools::TestQueries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

pub(crate) struct TestContext {
    pub(crate) app: Arc<bootstrap::App>,
    pub(crate) indexer_id: Digest,
    pub(crate) mq: TestQueries,
    pub(crate) container: Container,
}

pub(crate) async fn build_context_di(pool: &PgPool) -> anyhow::Result<TestContext> {
    let pool = pool.clone();
    let indexer = TestIndexer::default();
    let indexer_id = Registry::get_indexer_id(&indexer)?;

    // Create a DI container
    let mut c = Container::new();

    // Invoke optimizer registration
    c.invokable(move |c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(TestIndexable::TYPE_NAME, NoOpOptimizer);
            Ok(())
        })
    });

    // Invoke indexer registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<indexable::Registry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(Arc::new(indexer))
                .map_err(|err| InvokeError::new(err))?;
            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    crate::register(&mut c);

    c.provide(|_| Box::pin(async { Ok(configuration::EnableListening { value: false }) }));

    // Overwrite with the provided pool
    let wr = pool.clone();
    c.provide(|_| Box::pin(async { Ok(db::WritePool { pool: wr }) }));
    let ro = pool.clone();
    c.provide(|_| Box::pin(async { Ok(db::ReadPool { pool: ro }) }));

    // Invoke all
    c.invoke().await?;

    let app = c.get::<Arc<App>>().await?;

    let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);

    Ok(TestContext {
        app,
        mq,
        indexer_id,
        container: c,
    })
}

// The indexable type
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TestIndexable {
    id: Uuid,
    values: Vec<f32>,
    dimensions: Vec<usize>,
}

struct NoOpOptimizer;

impl foreign_service::Optimizer for NoOpOptimizer {
    type Type = TestIndexable;

    fn random(&self, _user_defined: &serde_json::Value) -> anyhow::Result<Self::Type> {
        Ok(TestIndexable::new((0.0, 0.0)))
    }

    fn crossover(
        &self,
        _parent1: Self::Type,
        _parent2: Self::Type,
        _user_defined: &serde_json::Value,
    ) -> anyhow::Result<Self::Type> {
        Ok(TestIndexable::new((0.0, 0.0)))
    }

    fn mutate(
        &self,
        _instance: &mut Self::Type,
        _user_defined: &serde_json::Value,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl Identifiable for TestIndexable {
    fn id(&self) -> Uuid {
        self.id
    }
}

impl TypeName for TestIndexable {
    fn type_name(&self) -> &'static str {
        Self::TYPE_NAME
    }
}

impl TestIndexable {
    const TYPE_NAME: &'static str = "test::indexable";

    pub(crate) fn new(values: (f32, f32)) -> Self {
        Self {
            id: Uuid::now_v7(),
            values: vec![values.0, values.1],
            dimensions: vec![1, 2],
        }
    }

    fn to_encode_input(&self) -> EncodeInput {
        EncodeInput {
            values: self.values.clone(),
            dimensions: self.dimensions.clone(),
        }
    }
}

// The indexer that knows how to index the indexable type
#[derive(Clone)]
pub(crate) struct TestIndexer {
    train_config: TrainModelConfig,
}

impl TypeName for TestIndexer {
    fn type_name(&self) -> &'static str {
        TestIndexable::TYPE_NAME
    }
}

impl indexable::Indexer for TestIndexer {
    type Type = TestIndexable;

    fn preprocess(&self, entity: &Self::Type) -> EncodeInput {
        entity.to_encode_input()
    }

    fn dataset(&self) -> Arc<dyn SequenceDataSource> {
        Arc::new(SequenceDataset::new(
            vec![
                SequenceSample {
                    steps: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
                },
                SequenceSample {
                    steps: vec![vec![0.5, 0.6]],
                },
            ],
            2,
        ))
    }

    fn training_config(&self) -> &TrainModelConfig {
        &self.train_config
    }
}

impl Default for TestIndexer {
    fn default() -> Self {
        Self {
            train_config: TrainModelConfig::Lstm(AutoencoderTrainConfig {
                input_size: 2,
                hidden_size: 4,
                latent_size: 2,
                batch_size: 2,
                epochs: 1,
                learning_rate: 1e-3,
            }),
        }
    }
}

pub async fn seed_encoder(ctx: &mut TestContext) -> anyhow::Result<Digest> {
    type TestBackend = NdArray<f32>;

    let config = AutoencoderConfig {
        input_size: 2,
        hidden_size: 4,
        latent_size: 2,
    };

    let device = <TestBackend as Backend>::Device::default();
    let model = lstm::LstmAutoencoder::<TestBackend>::new(&device, config);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let model_bytes = Recorder::<TestBackend>::record(&recorder, model.into_record(), ())?;
    let trained_at = Utc::now();

    let encoder = Encoder {
        digest: ctx.indexer_id,
        encodable_type_name: "encodable_type_name".to_string(),
        model_config: serde_json::to_value(&config)?,
        model_weights: model_bytes,
        model_format: MODEL_FORMAT.to_string(),
        shape_in: vec![2],
        shape_out: 2,
    };

    let encoders_wr = ctx.container.get::<encoders::Write>().await?;

    let indexer_id = ctx.indexer_id;

    let encoder = db::begin(encoders_wr.clone(), |tx| {
        Box::pin(async move {
            let mut wr = encoders::WriteTx::new(tx);

            let encoder = wr.store_encoder(&encoder, &trained_at).await?;

            wr.store_encoder_availability(&indexer_id, false).await?;

            Ok(encoder)
        })
    })
    .await?;

    Ok(encoder.digest)
}
