use crate::dataset::{HousingBatcher, HousingDataset};
use crate::model::RegressionModelConfig;
use burn::optim::AdamConfig;
use burn::train::LearningStrategy;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::AutodiffModule,
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

#[derive(Config, Debug)]
pub struct ExpConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    pub optimizer: AdamConfig,

    #[config(default = 256)]
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config
    let optimizer = AdamConfig::new();
    let config = ExpConfig::new(optimizer);
    let model = RegressionModelConfig::new().init(&device);
    B::seed(&device, config.seed);

    // Create dataset and split it
    let full_dataset = HousingDataset::new();
    let train_data = full_dataset.train();
    let valid_data = full_dataset.validation();

    println!("Train Dataset Size: {}", train_data.len());
    println!("Valid Dataset Size: {}", valid_data.len());

    let batcher_train = HousingBatcher::<B>::new(device.clone());

    let batcher_test = HousingBatcher::<B::InnerBackend>::new(device.clone());

    let train_dataset = InMemDataset::new(train_data);
    let valid_dataset = InMemDataset::new(valid_data);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .build(model, config.optimizer.init(), 1e-3);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .unwrap();

    model_trained
        .model
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}

/// Silent training function for GA optimization - returns validation loss
/// Uses custom training loop to avoid Learner's logging
#[tracing::instrument(level = "info", skip(model_config, device))]
pub fn train_silent<B: AutodiffBackend>(
    model_config: RegressionModelConfig,
    device: B::Device,
    epochs: usize,
) -> f32 {
    tracing::info!("Starting training");
    let batch_size = 128;
    let learning_rate = 1e-3;
    let seed = 1337u64;

    use burn::optim::{GradientsParams, Optimizer};

    let mut model = model_config.init(&device);
    let mut optim = AdamConfig::new().init();
    B::seed(&device, seed);

    // Create dataset
    let full_dataset = HousingDataset::new();
    let train_data = full_dataset.train();
    let valid_data = full_dataset.validation();

    let batcher_train = HousingBatcher::<B>::new(device.clone());
    let batcher_valid = HousingBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(1)
        .build(InMemDataset::new(train_data));

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(1)
        .build(InMemDataset::new(valid_data));

    // Custom training loop - no logging
    for _epoch in 0..epochs {
        // Training phase
        for batch in dataloader_train.iter() {
            let output = model.forward(batch.inputs);
            let targets = batch.targets.unsqueeze_dim(1);
            let loss = (output - targets).powf_scalar(2.0).mean();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads);
        }
    }

    // Validation phase - calculate final loss on inner backend
    let model = model.valid();
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    for batch in dataloader_valid.iter() {
        let output = model.forward(batch.inputs);
        let targets = batch.targets.unsqueeze_dim(1);
        let loss = (output - targets).powf_scalar(2.0).mean();
        let loss_value: f32 = loss.into_scalar().elem();
        total_loss += loss_value;
        num_batches += 1;
    }

    let result = total_loss / num_batches as f32;
    tracing::info!(validation_loss = result, "Training completed");
    result
}
