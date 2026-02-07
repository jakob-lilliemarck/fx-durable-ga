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
    fn sample(&self, index: usize) -> Option<SequenceSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
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
    #[should_panic(expected = "Every timestep must have exactly 2 features")]
    fn dataset_validates_feature_width() {
        let samples = vec![SequenceSample {
            steps: vec![vec![0.0], vec![1.0, 2.0]],
        }];
        let _dataset = SequenceDataset::new(samples, 2);
    }
}
