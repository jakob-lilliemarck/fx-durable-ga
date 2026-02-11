//! # GP Function Indexing
//!
//! This example trains an LSTM autoencoder on the *outputs* of several GP
//! programs. Every program runs against the same known input sequences so we
//! can compare their behaviours in a shared latent space.
//!
//! Different programs may emit different numbers of values per timestep. To
//! keep the autoencoder's input width constant, we transpose each
//! `[time_steps][output_dim]` matrix into `[output_dim][time_steps]` before
//! training or encoding. The transposed data lets the trainer treat
//! `time_steps` (which is fixed) as the per-timestep width, while the varying
//! dimension becomes the sequence length and is handled via the existing mask.
//! Anyone reading this example only needs to remember: always transpose before
//! flattening so the encoder sees consistent widths.

use anyhow::Result;
use fx_durable_ga::bootstrap;
use fx_durable_ga::models::EncodeInput;
use fx_durable_ga::services::indexing::TrainModelConfig;
use fx_durable_ga::services::indexing::encoder::dataset::{
    SequenceDataSource, SequenceDataset, SequenceSample,
};
use fx_durable_ga::services::indexing::encoder::train::AutoencoderTrainConfig;
use sqlx::postgres::PgPoolOptions;
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

    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for the GP indexing example");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    let service = bootstrap::ApplicationBuilder::default()
        .with_pool(pool)
        .indexing_service()
        .build();

    let programs = build_programs();
    let function_inputs = ProgramInputsDataset::new(INPUT_SEQUENCES, TIME_STEPS, INPUT_FEATURES);
    let outputs_dataset = ProgramOutputsDataset::new(&programs, &function_inputs);

    let train_config = AutoencoderTrainConfig {
        input_size: outputs_dataset.input_width(),
        hidden_size: HIDDEN_SIZE,
        latent_size: LATENT_SIZE,
        batch_size: BATCH_SIZE,
        epochs: EPOCHS,
        learning_rate: LEARNING_RATE,
    };

    let encoder = service
        .train_encoder(
            Uuid::now_v7(),
            TrainModelConfig::Lstm(train_config.clone()),
            outputs_dataset.clone(),
        )
        .await?;

    let encode_inputs = build_encode_inputs(&programs, &function_inputs);

    let tag_name = &[format!("indexing_example-{}", encoder.id().to_string())];

    let embedding_ids = service
        .index_many(&encoder.id(), &encode_inputs, tag_name)
        .await?;

    let similar = service
        .find_similar(&embedding_ids[0], &tag_name[0], 10)
        .await?;

    for s in similar {
        println!(
            "Program\t{}\tProgam {} cosine similarity: {:.4}",
            embedding_ids[0],
            s.embedding_id(),
            s.distance()
        );
    }

    Ok(())
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

fn build_encode_inputs(programs: &[Program], inputs: &ProgramInputsDataset) -> Vec<EncodeInput> {
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
        encode_inputs.push(EncodeInput {
            values: flattened,
            dimensions: vec![
                stacked_outputs.len(),
                stacked_outputs.first().map(|r| r.len()).unwrap_or(0),
            ],
        });
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
