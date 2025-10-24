use crate::dataset::{HousingBatch, HousingBatcher, HousingDataset};
use crate::model::RegressionModelConfig;
use burn::optim::AdamConfig;
use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    module::AutodiffModule,
    prelude::ElementConversion,
    tensor::backend::AutodiffBackend,
};
use std::sync::Arc;

/// Type alias for training DataLoader to simplify signatures.
pub type TrainDataLoader<B> = Arc<dyn DataLoader<B, HousingBatch<B>>>;

/// Type alias for validation DataLoader to simplify signatures.
pub type ValidDataLoader<B> = Arc<dyn DataLoader<B, HousingBatch<B>>>;

/// Creates reusable DataLoaders for training and validation.
/// Should be called once at initialization to avoid memory leaks.
#[tracing::instrument(level = "debug", skip(device))]
pub fn create_dataloaders<B: AutodiffBackend>(
    device: B::Device,
    batch_size: usize,
    seed: u64,
) -> (TrainDataLoader<B>, ValidDataLoader<B::InnerBackend>) {
    let full_dataset = HousingDataset::new();
    let train_data = full_dataset.train();
    let valid_data = full_dataset.validation();

    let batcher_train = HousingBatcher::<B>::new(device.clone());
    let batcher_valid = HousingBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(0) // No background workers to avoid thread/memory leaks
        .build(InMemDataset::new(train_data));

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(0) // No background workers to avoid thread/memory leaks
        .build(InMemDataset::new(valid_data));

    (dataloader_train, dataloader_valid)
}

/// Silent training function for GA optimization - returns validation loss.
/// Uses pre-built DataLoaders passed as parameters to avoid memory leaks.
#[tracing::instrument(
    level = "info",
    skip(model_config, device, dataloader_train, dataloader_valid)
)]
pub fn train_silent<B: AutodiffBackend>(
    model_config: RegressionModelConfig,
    device: B::Device,
    epochs: usize,
    dataloader_train: &TrainDataLoader<B>,
    dataloader_valid: &ValidDataLoader<B::InnerBackend>,
) -> f32 {
    tracing::info!("Starting training");

    use burn::optim::{GradientsParams, Optimizer};

    let mut model = model_config.init(&device);
    let mut optim = AdamConfig::new().init();

    // Custom training loop - explicitly scope iterators to ensure cleanup
    for _epoch in 0..epochs {
        {
            let train_iter = dataloader_train.iter();
            for batch in train_iter {
                let output = model.forward(batch.inputs);
                let targets = batch.targets.unsqueeze_dim(1);
                let loss = (output - targets).powf_scalar(2.0).mean();

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(model_config.learning_rate, model, grads);
            }
        }
    }

    // Validation phase - calculate final loss on inner backend
    let model = model.valid();
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    {
        let valid_iter = dataloader_valid.iter();
        for batch in valid_iter {
            let output = model.forward(batch.inputs);
            let targets = batch.targets.unsqueeze_dim(1);
            let loss = (output - targets).powf_scalar(2.0).mean();
            let loss_value: f32 = loss.into_scalar().elem();
            total_loss += loss_value;
            num_batches += 1;
        }
    }

    let result = total_loss / num_batches as f32;

    // Explicitly drop model and optimizer to ensure cleanup
    drop(model);
    drop(optim);

    tracing::info!(validation_loss = result, "Training completed");
    result
}
