use anyhow::Result;
use futures::future::BoxFuture;
use futures::lock::Mutex;
use fx_durable_ga::configuration;
use fx_durable_ga::infrastructure::di::{Container, InvokeError};
use fx_durable_ga::infrastructure::registrations::ProvidedEventHandlerRegistry;
use fx_durable_ga::repositories::genotypes::{Identifiable, TypeName};
use fx_durable_ga::services::events::EncoderAvailableEvent;
use fx_durable_ga::services::indexing::EncodeInput;
use fx_durable_ga::services::indexing::SearchSimilarEmbeddingsFilter;
use fx_durable_ga::services::indexing::TrainModelConfig;
use fx_durable_ga::services::indexing::encoder::dataset::{
    SequenceDataSource, SequenceDataset, SequenceSample,
};
use fx_durable_ga::services::indexing::encoder::train::AutoencoderTrainConfig;
use fx_durable_ga::services::indexing::{self as indexable, Registry};
use fx_durable_ga::services::optimization::{self as foreign_service, OptimizerRegistry};
use fx_durable_ga::services::{evaluation, indexing};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::Level;
use uuid::Uuid;

const LATENT_SIZE: usize = 32;
const HIDDEN_SIZE: usize = 32; //  HIDDEN_SIZE <= LATENT_SIZE
const BATCH_SIZE: usize = 2;
const EPOCHS: usize = 10;
const LEARNING_RATE: f64 = 1e-3;
const INPUT_SEQUENCES: usize = 1;
const TIME_STEPS: usize = 100;
const INPUT_FEATURES: usize = 5;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(Level::INFO)
        .init();

    let programs = build_programs();
    let function_inputs = ProgramInputsDataset::new(INPUT_SEQUENCES, TIME_STEPS, INPUT_FEATURES);
    let indexer_dataset = ProgramOutputsDataset::new(&programs, &function_inputs);
    let indexer = ProgramIndexer::new(indexer_dataset.clone());
    let indexer_id = Registry::get_indexer_id(&indexer).await?;

    // Create a DI container
    let mut c = Container::new();

    // Invoke optimizer registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(IndexingOnlyOptimizer);
            Ok(())
        })
    });

    // Invoke evaluator registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<evaluation::Service>>().await?;
            provided
                .register("examples::gp_function_indexing", IndexingOnlyOptimizer)
                .await;
            Ok(())
        })
    });

    // Invoke indexer registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<indexable::Registry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(Arc::new(indexer))
                .await
                .map_err(|err| InvokeError::new(err))?;
            Ok(())
        })
    });

    // Invoke event registration
    let (tx_encoder, mut rx_encoder) = mpsc::channel(1);
    c.invokable(move |c| {
        Box::pin(async move {
            let provided = c.get::<ProvidedEventHandlerRegistry>().await?;
            let mut lock = provided.lock().await;
            let Some(mut event_handler_registry) = lock.take() else {
                panic!("Could not take event handler registry")
            };

            event_handler_registry
                .with_handler(EncoderAvailableWatcher::new(indexer_id, tx_encoder));

            *lock = Some(event_handler_registry);

            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    //
    // NOTE!
    // Any provider overwrites must happen after registration!
    fx_durable_ga::register(&mut c);

    // Overwrite a configuration provider
    c.provide(|_| Box::pin(async { Ok(configuration::JobWorkerCount { value: 8 }) }));

    // Invoke all invokables
    c.invoke().await?;

    // Get an app instance from the container
    let app = c.get::<Arc<fx_durable_ga::bootstrap::App>>().await?;

    let tag = format!("indexing_example-{}", indexer_id);
    let encode_inputs = build_encode_inputs(&programs, &function_inputs, vec![tag.clone()]);

    // Indexing uses lazy encoder training. The first call may return None and
    // enqueue a TrainEncoder job. We wait for EncoderAvailableEvent, then index.
    let _embedding_ids = match app
        .services()
        .indexing()
        .index_many(&indexer_id, &encode_inputs, None)
        .await?
    {
        Some(ids) => ids,
        None => {
            tokio::time::timeout(Duration::from_secs(30), rx_encoder.recv())
                .await
                .map_err(|_| anyhow::anyhow!("timed out waiting for encoder training"))?
                .ok_or_else(|| anyhow::anyhow!("encoder availability channel closed"))?;

            app.services()
                .indexing()
                .index_many(&indexer_id, &encode_inputs, None)
                .await?
                .expect("expected embeddings after encoder training")
        }
    };

    let filter = SearchSimilarEmbeddingsFilter::default()
        .with_reference_tag(&tag)
        .with_search_tag(&tag);

    let similar = app
        .repositories()
        .embeddings()
        .find_similar(&indexer_id, &filter, 10)
        .await?;

    for (tagged_embedding, distance) in similar {
        println!(
            "embedding: {:?} centroid distance: {:.4}",
            tagged_embedding, distance
        );
    }

    Ok(())
}

// Indexing-only example: use a stub optimizer so we can register the indexer
// via `with_service`, which is the required app registration entrypoint.
#[derive(Clone, Copy)]
struct IndexingOnlyOptimizer;

impl TypeName for IndexingOnlyOptimizer {
    fn type_name(&self) -> &str {
        "examples::gp_function_indexing"
    }
}

impl IndexingOnlyOptimizer {
    fn stub_input() -> ProgramEncodeInput {
        ProgramEncodeInput::new(vec![0.0], 1, 1)
    }
}

impl foreign_service::Optimizer for IndexingOnlyOptimizer {
    type Type = ProgramEncodeInput;

    fn random(&self) -> anyhow::Result<Self::Type> {
        Ok(Self::stub_input())
    }

    fn crossover(&self, _parent1: Self::Type, _parent2: Self::Type) -> anyhow::Result<Self::Type> {
        Ok(Self::stub_input())
    }

    fn mutate(&self, _instance: &mut Self::Type) -> anyhow::Result<()> {
        Ok(())
    }
}

impl evaluation::Evaluator for IndexingOnlyOptimizer {
    type Type = ProgramEncodeInput;

    fn evaluate<'a>(
        &'a self,
        _instance: &'a Self::Type,
    ) -> futures::future::BoxFuture<
        'a,
        std::result::Result<f64, Box<dyn std::error::Error + Send + Sync>>,
    > {
        Box::pin(async { Ok(0.0) })
    }
}

struct EncoderAvailableWatcher {
    indexer_id: indexing::Digest,
    tx: mpsc::Sender<indexing::Digest>,
}

impl EncoderAvailableWatcher {
    fn new(indexer_id: indexing::Digest, tx: mpsc::Sender<indexing::Digest>) -> Self {
        Self { indexer_id, tx }
    }
}

impl fx_event_bus::Handler<EncoderAvailableEvent> for EncoderAvailableWatcher {
    type Error = std::convert::Infallible;

    fn handle<'a>(
        &'a self,
        event: Arc<EncoderAvailableEvent>,
        _: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        let sender = self.tx.clone();
        let indexer_id = self.indexer_id;

        Box::pin(async move {
            if event.indexer_id == indexer_id {
                let _ = sender.send(indexer_id).await;
            }
            (tx, Ok(()))
        })
    }
}

#[derive(Clone)]
struct ProgramInputsDataset {
    sequences: Vec<Vec<Vec<f32>>>,
    time_steps: usize,
}

impl ProgramInputsDataset {
    fn new(num_sequences: usize, time_steps: usize, feature_count: usize) -> Self {
        let mut sequences = Vec::with_capacity(num_sequences);

        for sample_idx in 0..num_sequences {
            let mut sequence = Vec::with_capacity(time_steps);
            for t in 0..time_steps {
                let mut row = Vec::with_capacity(feature_count);
                for feature in 0..feature_count {
                    row.push(sample_idx as f32 + t as f32 + feature as f32);
                }
                sequence.push(row);
            }
            sequences.push(sequence);
        }

        Self {
            sequences,
            time_steps,
        }
    }

    fn time_steps(&self) -> usize {
        self.time_steps
    }

    fn raw_sequences(&self) -> impl Iterator<Item = &[Vec<f32>]> {
        self.sequences.iter().map(|seq| seq.as_slice())
    }
}

#[derive(Clone)]
struct ProgramOutputsDataset {
    dataset: SequenceDataset,
}

impl ProgramOutputsDataset {
    fn new(programs: &[Program], inputs: &ProgramInputsDataset) -> Self {
        let mut samples = Vec::new();

        for program in programs {
            for sequence in inputs.raw_sequences() {
                let outputs: Vec<Vec<f32>> = sequence.iter().map(|row| program.eval(row)).collect();
                if outputs.is_empty() {
                    continue;
                }
                let transposed = transpose(&outputs);
                samples.push(SequenceSample { steps: transposed });
            }
        }

        let dataset = SequenceDataset::new(samples, inputs.time_steps());

        Self { dataset }
    }

    fn input_width(&self) -> usize {
        self.dataset.input_size()
    }
}

impl SequenceDataSource for ProgramOutputsDataset {
    fn checksum(&self) -> Vec<u8> {
        self.dataset.checksum()
    }

    fn sample(&self, index: usize) -> Option<SequenceSample> {
        self.dataset.sample(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct ProgramEncodeInput {
    id: Uuid,
    values: Vec<f32>,
    dimensions: Vec<usize>,
}

impl ProgramEncodeInput {
    fn new(values: Vec<f32>, seq_len: usize, input_size: usize) -> Self {
        Self {
            id: Uuid::now_v7(),
            values,
            dimensions: vec![seq_len, input_size],
        }
    }

    fn to_encode_input(&self) -> EncodeInput {
        EncodeInput {
            values: self.values.clone(),
            dimensions: self.dimensions.clone(),
        }
    }
}

impl Identifiable for ProgramEncodeInput {
    fn id(&self) -> uuid::Uuid {
        self.id
    }
}

impl TypeName for ProgramEncodeInput {
    fn type_name(&self) -> &'static str {
        "examples::gp_function_indexing"
    }
}

#[derive(Clone)]
struct ProgramIndexer {
    dataset: ProgramOutputsDataset,
    train_config: TrainModelConfig,
}

impl ProgramIndexer {
    fn new(dataset: ProgramOutputsDataset) -> Self {
        let input_size = dataset.input_width();
        Self {
            dataset,
            train_config: TrainModelConfig::Lstm(AutoencoderTrainConfig {
                input_size,
                hidden_size: HIDDEN_SIZE,
                latent_size: LATENT_SIZE,
                batch_size: BATCH_SIZE,
                epochs: EPOCHS,
                learning_rate: LEARNING_RATE,
            }),
        }
    }
}

impl TypeName for ProgramIndexer {
    fn type_name(&self) -> &'static str {
        "examples::gp_function_indexing"
    }
}

impl indexable::Indexer for ProgramIndexer {
    type Type = ProgramEncodeInput;

    fn preprocess(&self, entity: &Self::Type) -> EncodeInput {
        entity.to_encode_input()
    }

    fn dataset(&self) -> BoxFuture<'_, Arc<dyn SequenceDataSource>> {
        let dataset: Arc<dyn SequenceDataSource> = Arc::new(self.dataset.clone());
        Box::pin(async { dataset })
    }

    fn training_config(&self) -> &TrainModelConfig {
        &self.train_config
    }
}

fn build_encode_inputs(
    programs: &[Program],
    inputs: &ProgramInputsDataset,
    tags: Vec<String>,
) -> Vec<(ProgramEncodeInput, Vec<String>)> {
    let mut encode_inputs = Vec::new();

    for program in programs {
        let mut stacked_outputs: Vec<Vec<f32>> = Vec::new();

        for sequence in inputs.raw_sequences() {
            let outputs: Vec<Vec<f32>> = sequence.iter().map(|row| program.eval(row)).collect();
            if outputs.is_empty() {
                continue;
            }
            let transposed = transpose(&outputs);
            stacked_outputs.extend(transposed);
        }

        if stacked_outputs.is_empty() {
            continue;
        }

        let flattened = flatten(&stacked_outputs);
        let seq_len = stacked_outputs.len();
        let input_size = stacked_outputs.first().map(|r| r.len()).unwrap_or(0);
        encode_inputs.push((
            ProgramEncodeInput::new(flattened, seq_len, input_size),
            tags.clone(),
        ));
    }

    encode_inputs
}

fn flatten(matrix: &[Vec<f32>]) -> Vec<f32> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0f32; rows]; cols];

    for (r, row) in matrix.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            transposed[c][r] = *value;
        }
    }

    transposed
}

fn build_programs() -> Vec<Program> {
    vec![
        Program {
            trees: vec![Expr::Sub(
                Box::new(Expr::Select(2)),
                Box::new(Expr::Select(4)),
            )],
        },
        // Identical to first for input like [N, N+1, N+2, N+3, N+4]
        Program {
            trees: vec![Expr::Sub(
                Box::new(Expr::Add(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Select(2)),
                        Box::new(Expr::Select(4)),
                    )),
                    Box::new(Expr::Sub(
                        Box::new(Expr::Select(1)),
                        Box::new(Expr::Select(3)),
                    )),
                )),
                Box::new(Expr::Sub(
                    Box::new(Expr::Select(1)),
                    Box::new(Expr::Select(3)),
                )),
            )],
        },
        // Slightly different constant: (t+3) - (t+4) = -1
        Program {
            trees: vec![Expr::Sub(
                Box::new(Expr::Select(3)),
                Box::new(Expr::Select(4)),
            )],
        },
        // Very dissimilar for input like [N, N+1, N+2, N+3, N+4]
        Program {
            trees: vec![Expr::Add(
                Box::new(Expr::Pow(Box::new(Expr::Select(3)))),
                Box::new(Expr::Pow(Box::new(Expr::Select(2)))),
            )],
        },
    ]
}

pub enum Expr {
    Select(usize),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>),
}

impl Expr {
    pub fn eval(&self, input_data: &[f32]) -> f32 {
        match self {
            Expr::Select(idx) => input_data[*idx % input_data.len()],
            Expr::Add(l, r) => l.eval(input_data) + r.eval(input_data),
            Expr::Sub(l, r) => l.eval(input_data) - r.eval(input_data),
            Expr::Pow(child) => {
                let val = child.eval(input_data);
                val * val
            }
        }
    }
}

pub struct Program {
    pub trees: Vec<Expr>,
}

impl Program {
    pub fn eval(&self, input_data: &[f32]) -> Vec<f32> {
        self.trees
            .iter()
            .map(|node| node.eval(input_data))
            .collect()
    }
}
