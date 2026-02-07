use super::dataset::SequenceSample;
use burn::prelude::*;

/// Batch returned by the autoencoder batcher.
#[derive(Clone, Debug)]
pub struct AutoencoderBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,
    pub targets: Tensor<B, 3>,
    pub mask: Tensor<B, 3>,
    pub lengths: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct AutoencoderBatcher {
    input_size: usize,
}

impl AutoencoderBatcher {
    pub fn new(input_size: usize) -> Self {
        Self { input_size }
    }

    pub fn batch<B: Backend>(
        &self,
        items: Vec<SequenceSample>,
        device: &B::Device,
    ) -> AutoencoderBatch<B> {
        assert!(!items.is_empty(), "Cannot create a batch from zero samples");

        let batch_size = items.len();
        let max_seq_len = items.iter().map(|s| s.seq_len()).max().unwrap_or(0);

        let total_elements = batch_size * max_seq_len * self.input_size;
        let mut input_data = vec![0.0f32; total_elements];
        let mut mask_data = vec![0.0f32; total_elements];
        let mut lengths = Vec::with_capacity(batch_size);

        for (batch_idx, sample) in items.iter().enumerate() {
            lengths.push(sample.seq_len());
            for (t, timestep) in sample.steps.iter().enumerate() {
                assert_eq!(
                    timestep.len(),
                    self.input_size,
                    "Timestep width must match input_size"
                );
                let offset = (batch_idx * max_seq_len + t) * self.input_size;
                input_data[offset..offset + self.input_size].copy_from_slice(timestep);
                mask_data[offset..offset + self.input_size].fill(1.0);
            }
        }

        let shape = [batch_size, max_seq_len, self.input_size];

        let inputs = Tensor::<B, 3>::from_data(TensorData::new(input_data.clone(), shape), device);
        let targets = Tensor::<B, 3>::from_data(TensorData::new(input_data, shape), device);
        let mask = Tensor::<B, 3>::from_data(TensorData::new(mask_data, shape), device);

        AutoencoderBatch {
            inputs,
            targets,
            mask,
            lengths,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn batcher_pads_sequences_and_generates_mask() {
        type B = NdArray;
        let batcher = AutoencoderBatcher::new(2);
        let device = <B as Backend>::Device::default();

        let samples = vec![
            SequenceSample {
                steps: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            },
            SequenceSample {
                steps: vec![vec![5.0, 6.0]],
            },
        ];

        let batch: AutoencoderBatch<B> = batcher.batch(samples, &device);

        assert_eq!(batch.inputs.dims(), [2, 2, 2]);
        assert_eq!(batch.targets.dims(), [2, 2, 2]);
        assert_eq!(batch.mask.dims(), [2, 2, 2]);
        assert_eq!(batch.lengths, vec![2, 1]);

        let mask_data = batch
            .mask
            .into_data()
            .into_vec::<f32>()
            .expect("mask to vec");
        assert_eq!(mask_data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }
}
