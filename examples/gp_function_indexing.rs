use anyhow::Result;
use fx_durable_ga::bootstrap;
use fx_durable_ga::services::indexing::encoder::dataset::{SequenceDataSource, SequenceSample};
use fx_durable_ga::services::indexing::encoder::train::AutoencoderTrainConfig;
use fx_durable_ga::services::indexing::{EncodeInput, TrainModelConfig};
use sqlx::postgres::PgPoolOptions;

const LATENT_SIZE: usize = 32;
const HIDDEN_SIZE: usize = 32; //  HIDDEN_SIZE <= LATENT_SIZE
const BATCH_SIZE: usize = 2;
const EPOCHS: usize = 5;
const LEARNING_RATE: f64 = 1e-3;
const SAMPLES: usize = 8;
const INPUT_SIZE: usize = 1_00;
const FEATURES: usize = 5;

#[tokio::main]
async fn main() -> Result<()> {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for the GP indexing example");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    let service = bootstrap::ServiceBuilder::default()
        .with_pool(pool)
        .build_indexing_svc();

    let dataset = ExampleDataset::new(SAMPLES, INPUT_SIZE, FEATURES);
    let train_config = AutoencoderTrainConfig {
        input_size: INPUT_SIZE,
        hidden_size: HIDDEN_SIZE,
        latent_size: LATENT_SIZE,
        batch_size: BATCH_SIZE,
        epochs: EPOCHS,
        learning_rate: LEARNING_RATE,
    };

    let encoder = service
        .train_encoder(
            TrainModelConfig::Lstm(train_config.clone()),
            dataset.clone(),
        )
        .await?;

    let sample = dataset
        .raw_sample(0)
        .expect("dataset to contain at least one sample");

    let programs = build_programs();
    let mut embeddings = Vec::with_capacity(programs.len());

    for program in programs.iter() {
        let outputs: Vec<Vec<f32>> = sample.iter().map(|row| program.eval(row)).collect();
        let transposed = transpose(&outputs);
        let flattened = flatten(&transposed);
        let encode_input = EncodeInput {
            values: flattened,
            dimensions: vec![
                transposed.len(),
                transposed.first().map(|r| r.len()).unwrap_or(0),
            ],
        };

        let embedding = service.encode(&encoder.id(), &encode_input).await?;
        embeddings.push(embedding.into_iter().collect::<Vec<f32>>());
    }

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let cosine = cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("Program {} vs {} cosine: {:.4}", i, j, cosine);
        }
    }

    Ok(())
}

#[derive(Clone)]
struct ExampleDataset {
    samples: Vec<SequenceSample>,
    raw_samples: Vec<Vec<Vec<f32>>>,
}

impl ExampleDataset {
    fn new(num_samples: usize, time_steps: usize, feature_count: usize) -> Self {
        let mut samples = Vec::with_capacity(num_samples);
        let mut raw_samples = Vec::with_capacity(num_samples);

        for sample_idx in 0..num_samples {
            let mut raw = Vec::with_capacity(time_steps);
            for t in 0..time_steps {
                let mut row = Vec::with_capacity(feature_count);
                for feature in 0..feature_count {
                    row.push(sample_idx as f32 + t as f32 + feature as f32);
                }
                raw.push(row);
            }

            let transposed = transpose(&raw);
            raw_samples.push(raw);
            samples.push(SequenceSample { steps: transposed });
        }

        Self {
            samples,
            raw_samples,
        }
    }

    fn raw_sample(&self, index: usize) -> Option<&[Vec<f32>]> {
        self.raw_samples.get(index).map(|sample| sample.as_slice())
    }
}

impl SequenceDataSource for ExampleDataset {
    fn sample(&self, index: usize) -> Option<SequenceSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

fn build_programs() -> Vec<Program> {
    vec![
        Program {
            trees: vec![Expr::Sub(
                Box::new(Expr::Select(2)),
                Box::new(Expr::Select(4)),
            )],
        },
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
