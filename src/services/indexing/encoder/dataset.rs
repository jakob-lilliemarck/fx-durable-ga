use sha2::{Digest, Sha256};

/// A single variable-length sequence prepared for the autoencoder.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSample {
    /// Timesteps laid out as `[seq_len, input_size]` after any preprocessing.
    pub steps: Vec<Vec<f32>>,
}

impl SequenceSample {
    pub fn seq_len(&self) -> usize {
        self.steps.len()
    }
}

/// In-memory dataset storing all samples alongside the expected input width.
#[derive(Clone, Debug)]
pub struct SequenceDataset {
    samples: Vec<SequenceSample>,
    input_size: usize,
}

pub trait SequenceDataSource {
    fn checksum(&self) -> Vec<u8>;
    fn sample(&self, index: usize) -> Option<SequenceSample>;
    fn len(&self) -> usize;
}

impl SequenceDataset {
    pub fn new(samples: Vec<SequenceSample>, input_size: usize) -> Self {
        Self::validate(&samples, input_size);
        Self {
            samples,
            input_size,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    fn validate(samples: &[SequenceSample], input_size: usize) {
        for sample in samples {
            for step in &sample.steps {
                assert_eq!(
                    step.len(),
                    input_size,
                    "Every timestep must have exactly {} features",
                    input_size
                );
            }
        }
    }
}

impl SequenceDataSource for SequenceDataset {
    fn checksum(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update((self.input_size as u64).to_le_bytes());
        hasher.update((self.samples.len() as u64).to_le_bytes());

        for sample in &self.samples {
            hash_sample(&mut hasher, sample);
        }

        hasher.finalize().to_vec()
    }

    fn sample(&self, index: usize) -> Option<SequenceSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

fn hash_sample(hasher: &mut Sha256, sample: &SequenceSample) {
    hasher.update((sample.steps.len() as u64).to_le_bytes());

    for step in &sample.steps {
        hasher.update((step.len() as u64).to_le_bytes());
        for value in step {
            hasher.update(value.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_len_matches_steps() {
        let sample = SequenceSample {
            steps: vec![vec![0.0, 1.0], vec![2.0, 3.0]],
        };
        assert_eq!(sample.seq_len(), 2);
    }

    #[test]
    fn dataset_get_clones_samples() {
        let samples = vec![SequenceSample {
            steps: vec![vec![0.0, 1.0], vec![2.0, 3.0]],
        }];
        let dataset = SequenceDataset::new(samples.clone(), 2);

        assert_eq!(SequenceDataSource::len(&dataset), 1);
        let fetched = SequenceDataSource::sample(&dataset, 0).unwrap();
        assert_eq!(fetched, samples[0]);
    }

    #[test]
    fn checksum_is_deterministic() {
        let samples = vec![SequenceSample {
            steps: vec![vec![0.0, 1.0], vec![2.0, 3.0]],
        }];

        let dataset_a = SequenceDataset::new(samples.clone(), 2);
        let dataset_b = SequenceDataset::new(samples, 2);

        assert_eq!(
            SequenceDataSource::checksum(&dataset_a),
            SequenceDataSource::checksum(&dataset_b)
        );
    }

    #[test]
    fn checksum_changes_when_data_changes() {
        let mut sample = SequenceSample {
            steps: vec![vec![0.0, 1.0], vec![2.0, 3.0]],
        };
        let baseline = SequenceDataset::new(vec![sample.clone()], 2);

        sample.steps[0][0] = 42.0;
        let modified = SequenceDataset::new(vec![sample], 2);

        assert_ne!(
            SequenceDataSource::checksum(&baseline),
            SequenceDataSource::checksum(&modified)
        );
    }

    #[test]
    #[should_panic(expected = "Every timestep must have exactly 2 features")]
    fn dataset_validates_feature_width() {
        let samples = vec![SequenceSample {
            steps: vec![vec![0.0], vec![1.0, 2.0]],
        }];
        let _dataset = SequenceDataset::new(samples, 2);
    }
}
