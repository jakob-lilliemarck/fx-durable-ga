use crate::models::{Gene, Morphology};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    LatinHypercube { population_size: u32 },
    Random { population_size: u32 },
}

impl Distribution {
    pub fn latin_hypercube(population_size: u32) -> Self {
        Distribution::LatinHypercube { population_size }
    }

    pub fn random(population_size: u32) -> Self {
        Distribution::Random { population_size }
    }

    pub(crate) fn distribute(&self, morphology: &Morphology) -> Vec<Vec<Gene>> {
        match self {
            Distribution::LatinHypercube { population_size } => {
                latin_hypercube(*population_size as usize, morphology)
            }
            Distribution::Random { population_size } => {
                random_distribution(*population_size as usize, morphology)
            }
        }
    }
}

fn random_distribution(n_samples: usize, morphology: &Morphology) -> Vec<Vec<Gene>> {
    let mut genomes = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let genome = morphology.random();
        genomes.push(genome);
    }

    genomes
}

fn latin_hypercube(n_samples: usize, morphology: &Morphology) -> Vec<Vec<Gene>> {
    use rand::seq::SliceRandom;

    let n_dimensions = morphology.gene_bounds.len();
    let mut rng = rand::rng();

    // Create n_samples genomes
    let mut genomes = Vec::with_capacity(n_samples);

    // For each dimension, create Latin Hypercube sampling
    for dim_idx in 0..n_dimensions {
        let gene_bound = &morphology.gene_bounds[dim_idx];

        // 1. Create n_samples intervals at center points (deterministic)
        let mut intervals: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 + 0.5) / n_samples as f64) // Center of each interval
            .collect();

        // 2. Shuffle the intervals to decorrelate dimensions
        intervals.shuffle(&mut rng);

        // 3. Convert [0,1] samples to actual gene values using gene bounds
        let gene_values: Vec<Gene> = intervals
            .iter()
            .map(|&sample| gene_bound.from_sample(sample))
            .collect();

        // 4. Assign to genomes (transpose operation)
        for (genome_idx, &gene_value) in gene_values.iter().enumerate() {
            if dim_idx == 0 {
                // First dimension: create new genomes
                genomes.push(vec![gene_value]);
            } else {
                // Subsequent dimensions: append to existing genomes
                genomes[genome_idx].push(gene_value);
            }
        }
    }

    genomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GeneBounds;

    // Helper to create test morphology
    fn create_test_morphology(gene_bounds: Vec<GeneBounds>) -> Morphology {
        Morphology::new("test", 1, gene_bounds)
    }

    #[test]
    fn test_latin_hypercube_2d_simple() {
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 3, 4).unwrap(), // 4 steps: [0, 1, 2, 3]
            GeneBounds::integer(0, 3, 4).unwrap(), // 4 steps: [0, 1, 2, 3]
        ]);

        let dist = Distribution::latin_hypercube(4);
        let genomes = dist.distribute(&morphology);

        // Should have 4 genomes, each with 2 genes
        assert_eq!(genomes.len(), 4);
        assert!(genomes.iter().all(|genome| genome.len() == 2));

        // Extract dimensions for easier checking
        let dim1: Vec<i64> = genomes.iter().map(|g| g[0]).collect();
        let dim2: Vec<i64> = genomes.iter().map(|g| g[1]).collect();

        // Each dimension should contain exactly one of each value [0, 1, 2, 3]
        let mut dim1_sorted = dim1.clone();
        dim1_sorted.sort();
        assert_eq!(dim1_sorted, vec![0, 1, 2, 3]);

        let mut dim2_sorted = dim2.clone();
        dim2_sorted.sort();
        assert_eq!(dim2_sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_latin_hypercube_3d_simple() {
        // 3D test: 3 samples in 3 dimensions
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 2, 3).unwrap(), // 3 steps: [0, 1, 2]
            GeneBounds::integer(0, 2, 3).unwrap(), // 3 steps: [0, 1, 2]
            GeneBounds::integer(0, 2, 3).unwrap(), // 3 steps: [0, 1, 2]
        ]);

        let dist = Distribution::latin_hypercube(3);
        let genomes = dist.distribute(&morphology);

        // Should have 3 genomes, each with 3 genes
        assert_eq!(genomes.len(), 3);
        assert!(genomes.iter().all(|genome| genome.len() == 3));

        // Each dimension should contain exactly [0, 1, 2] in some order
        for dim in 0..3 {
            let mut values: Vec<i64> = genomes.iter().map(|g| g[dim]).collect();
            values.sort();
            assert_eq!(values, vec![0, 1, 2]);
        }
    }

    #[test]
    fn test_latin_hypercube_single_sample() {
        // Edge case: single sample should work
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 9, 10).unwrap(),
            GeneBounds::integer(0, 9, 10).unwrap(),
        ]);

        let dist = Distribution::latin_hypercube(1);
        let genomes = dist.distribute(&morphology);

        // Basic structure should work with single sample
        assert_eq!(genomes.len(), 1);
        assert_eq!(genomes[0].len(), 2);

        // Values should be within bounds
        assert!((0..=9).contains(&genomes[0][0]));
        assert!((0..=9).contains(&genomes[0][1]));
    }

    #[test]
    fn test_distribution_distribute_random() {
        let morphology = create_test_morphology(vec![
            GeneBounds::integer(0, 5000, 1000).unwrap(),
            GeneBounds::integer(0, 3000, 1000).unwrap(),
        ]);

        let dist = Distribution::random(5);

        // Generate multiple distributions to test randomness
        let genomes1 = dist.distribute(&morphology);
        let genomes2 = dist.distribute(&morphology);

        // Basic structure validation
        assert_eq!(genomes1.len(), 5);
        assert!(genomes1.iter().all(|genome| genome.len() == 2));

        // The key test: random distributions should be different
        assert_ne!(genomes1, genomes2);
    }
}
