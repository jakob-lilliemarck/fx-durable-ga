use super::batcher::{AutoencoderBatch, AutoencoderBatcher};
use super::dataset::SequenceDataSource;
use super::lstm::{AutoencoderConfig, AutoencoderModel, LstmAutoencoder};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Minimal training configuration for the LSTM autoencoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoencoderTrainConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub latent_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
}

impl AutoencoderTrainConfig {
    #[instrument(level = "debug", skip(self))]
    pub fn autoencoder_config(&self) -> AutoencoderConfig {
        AutoencoderConfig {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            latent_size: self.latent_size,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingReport {
    pub epoch_losses: Vec<f32>,
}

pub fn train_autoencoder<B>(
    device: &B::Device,
    dataset: &impl SequenceDataSource,
    config: AutoencoderTrainConfig,
) -> (
    <LstmAutoencoder<B> as AutodiffModule<B>>::InnerModule,
    TrainingReport,
)
where
    B: AutodiffBackend,
    LstmAutoencoder<B>: AutodiffModule<B>,
{
    let mut model = LstmAutoencoder::<B>::new(device, config.autoencoder_config());
    let mut optimizer = AdamConfig::new().init();
    let batcher = AutoencoderBatcher::new(config.input_size);

    let mut epoch_losses = Vec::with_capacity(config.epochs);

    for _ in 0..config.epochs {
        let mut total_loss = 0.0f32;
        let mut batches = 0usize;

        for start in (0..dataset.len()).step_by(config.batch_size) {
            let end = (start + config.batch_size).min(dataset.len());
            let items: Vec<_> = (start..end).filter_map(|idx| dataset.sample(idx)).collect();
            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items, device);
            let (updated_model, loss) =
                train_batch(model, &mut optimizer, batch, config.learning_rate);
            model = updated_model;
            total_loss += loss;
            batches += 1;
        }

        let avg_loss = if batches > 0 {
            total_loss / batches as f32
        } else {
            0.0
        };
        epoch_losses.push(avg_loss);
    }

    let report = TrainingReport { epoch_losses };
    (model.valid(), report)
}

#[instrument(level = "debug", skip_all)]
fn train_batch<B, O>(
    model: LstmAutoencoder<B>,
    optimizer: &mut O,
    batch: AutoencoderBatch<B>,
    learning_rate: f64,
) -> (LstmAutoencoder<B>, f32)
where
    B: AutodiffBackend,
    LstmAutoencoder<B>: AutodiffModule<B>,
    O: Optimizer<LstmAutoencoder<B>, B>,
{
    let outputs = model.forward(batch.inputs.clone());
    let reconstruction_loss = masked_mse(outputs.reconstruction, batch.targets, batch.mask);
    let loss_value = reconstruction_loss.clone().into_scalar().elem::<f32>();
    let grads = reconstruction_loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let updated_model = optimizer.step(learning_rate, model, grads);
    (updated_model, loss_value)
}

#[instrument(level = "debug", skip_all)]
fn masked_mse<B: AutodiffBackend>(
    prediction: Tensor<B, 3>,
    target: Tensor<B, 3>,
    mask: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let squared_error = (prediction - target).powf_scalar(2.0) * mask.clone();
    let total_error = squared_error.sum();
    let valid_elements = mask.sum().into_scalar().elem::<f32>().max(1e-6);
    total_error / valid_elements
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::indexing::encoder::dataset::{SequenceDataset, SequenceSample};
    use burn::backend::Autodiff;
    use burn_ndarray::NdArray;

    type ADBackend = Autodiff<NdArray>;

    fn sample_dataset() -> SequenceDataset {
        SequenceDataset::new(
            vec![
                SequenceSample {
                    steps: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
                },
                SequenceSample {
                    steps: vec![vec![0.5, 0.6]],
                },
                SequenceSample {
                    steps: vec![vec![0.7, 0.8], vec![0.9, 1.0], vec![1.1, 1.2]],
                },
            ],
            2,
        )
    }

    #[test]
    fn training_produces_finite_losses() {
        let device = <ADBackend as Backend>::Device::default();
        let dataset = sample_dataset();
        let config = AutoencoderTrainConfig {
            input_size: 2,
            hidden_size: 4,
            latent_size: 2,
            batch_size: 2,
            epochs: 2,
            learning_rate: 1e-3,
        };

        let (_model, report) = train_autoencoder::<ADBackend>(&device, &dataset, config);
        assert_eq!(report.epoch_losses.len(), 2);
        assert!(report.epoch_losses.iter().all(|loss| loss.is_finite()));
    }
}
