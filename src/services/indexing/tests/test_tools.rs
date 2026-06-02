use super::super::Digest;
use super::super::repositories::encoders::Encoder;
use crate::bootstrap;
use crate::infrastructure::db;
use crate::infrastructure::di::Container;
use crate::repositories::genotypes::{Identifiable, TypeName};
use crate::services::indexing::EncodeInput;
use crate::services::indexing::repositories::encoders;
use crate::services::indexing::{self as indexable};
use crate::services::indexing::{
    TrainModelConfig,
    encoder::{
        dataset::{SequenceDataSource, SequenceDataset, SequenceSample},
        train::AutoencoderTrainConfig,
    },
};
use crate::services::indexing::{
    encoder::lstm::{self, AutoencoderConfig, AutoencoderModel},
    service::MODEL_FORMAT,
};
use crate::services::optimization as foreign_service;
use burn::prelude::Backend;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_ndarray::NdArray;
use chrono::Utc;
use fx_mq_building_blocks::testing_tools::TestQueries;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

pub(crate) struct TestContext {
    pub(crate) app: Arc<bootstrap::App>,
    pub(crate) indexer_id: Digest,
    pub(crate) mq: TestQueries,
    pub(crate) container: Container,
}

// The indexable type
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TestIndexable {
    id: Uuid,
    values: Vec<f32>,
    dimensions: Vec<usize>,
}

pub(crate) struct NoOpOptimizer;

impl TypeName for NoOpOptimizer {
    fn type_name(&self) -> &str {
        TestIndexable::TYPE_NAME
    }
}

impl foreign_service::Optimizer for NoOpOptimizer {
    type Type = TestIndexable;

    fn random(&self) -> anyhow::Result<Self::Type> {
        Ok(TestIndexable::new((0.0, 0.0)))
    }

    fn crossover(&self, _parent1: Self::Type, _parent2: Self::Type) -> anyhow::Result<Self::Type> {
        Ok(TestIndexable::new((0.0, 0.0)))
    }

    fn mutate(&self, _instance: &mut Self::Type) -> anyhow::Result<()> {
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
