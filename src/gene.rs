use rand::{Rng, rngs::ThreadRng};
use std::rc::Rc;
use uuid::Uuid;

pub type Gene = i64;

/// Defines valid value range and discretization for a Gene type
#[derive(Debug, Clone)]
pub struct GeneBounds {
    pub(crate) id: Uuid,
    pub(crate) start: i64,   // Start of range (inclusive)
    pub(crate) end: i64,     // End of range (inclusive)
    pub(crate) divisor: u32, // Number of discrete steps
}

#[derive(Debug, thiserror::Error)]
pub enum BoundsError {
    #[error("start is equal to end: start={start}, end={end}")]
    StartEqEnd { start: i64, end: i64 },
    #[error("start is greater than end: start={start}, end={end}")]
    StartGtEnd { start: i64, end: i64 },
    /// Divisor smaller than 2 is invalid, as that would be a constant unoptimizable value
    #[error("divisor is less than 2: divisor={divisor}")]
    DivisorLtTwo { divisor: u32 },
}

impl GeneBounds {
    pub fn new(start: i64, end: i64, divisor: u32) -> Result<Self, BoundsError> {
        if start == end {
            return Err(BoundsError::StartEqEnd { start, end });
        }

        if start > end {
            return Err(BoundsError::StartGtEnd { start, end });
        }

        if divisor < 2 {
            return Err(BoundsError::DivisorLtTwo { divisor });
        }

        Ok(Self {
            id: Uuid::now_v7(),
            start,
            end,
            divisor,
        })
    }

    /// Returns random gene value within bounds
    pub fn random(&self, rng: &mut ThreadRng) -> Gene {
        rng.random_range(0..self.divisor as i64)
    }
}

/// Collection of bounds defining possible gene values for a genome
#[derive(Debug)]
pub struct Morphology {
    pub(crate) id: Uuid,
    pub(crate) bounds: Vec<GeneBounds>,
}

impl Morphology {
    pub fn random(&self, rng: &mut ThreadRng) -> Vec<Gene> {
        self.bounds.iter().map(|b| b.random(rng)).collect()
    }
}

/// Instance of a genome with specific gene values
#[derive(Debug)]
pub struct Genome {
    id: Uuid,
    genes: Vec<Gene>,
    morphology: Rc<Morphology>,
}

#[derive(Debug, thiserror::Error)]
pub enum CrossoverError {
    #[error(
        "cannot crossover genomes with different morphologies: self={self_id}, other={other_id}"
    )]
    IncompatibleMorphology { self_id: Uuid, other_id: Uuid },
}

#[derive(Debug, thiserror::Error)]
pub enum PercentageError {
    #[error("{name} must be between 0 and 100, got {value}")]
    OutOfRange { name: &'static str, value: u8 },
}

impl Genome {
    /// Get the gene values
    pub fn genes(&self) -> &[Gene] {
        &self.genes
    }
    /// Creates new genome with random gene values
    pub fn random(rng: &mut ThreadRng, morphology: &Rc<Morphology>) -> Self {
        Self {
            id: Uuid::now_v7(),
            genes: morphology.random(rng),
            morphology: morphology.clone(),
        }
    }

    /// Creates child genome by mixing genes from two parents
    pub fn crossover(&self, rng: &mut ThreadRng, other: &Genome) -> Result<Genome, CrossoverError> {
        if !Rc::ptr_eq(&self.morphology, &other.morphology) {
            return Err(CrossoverError::IncompatibleMorphology {
                self_id: self.morphology.id,
                other_id: other.morphology.id,
            });
        }

        // Option 1: Simple uniform crossover (50/50 chance for each gene)
        let genes: Vec<Gene> = self
            .genes
            .iter()
            .zip(other.genes.iter())
            .map(|(&a, &b)| if rng.random_bool(0.5) { a } else { b })
            .collect();

        Ok(Genome {
            id: Uuid::now_v7(),
            genes,
            morphology: Rc::clone(&self.morphology),
        })
    }

    /// Mutates genes based on temperature (0-100) and rate (0-100)
    pub fn mutate(
        &mut self,
        rng: &mut ThreadRng,
        temperature: u8,
        rate: u8,
    ) -> Result<(), PercentageError> {
        if temperature > 100 {
            return Err(PercentageError::OutOfRange {
                name: "temperature",
                value: temperature,
            });
        }

        if rate > 100 {
            return Err(PercentageError::OutOfRange {
                name: "rate",
                value: rate,
            });
        }

        for (gene, bounds) in self.genes.iter_mut().zip(self.morphology.bounds.iter()) {
            // Should we mutate this gene?
            if rng.random_range(0..100) < rate {
                // Temperature controls mutation step: higher = larger jumps
                let max_step = (1 + (bounds.divisor * temperature as u32) / 100).max(1);

                // Choose direction and step size
                let direction = if rng.random_bool(0.5) { 1 } else { -1 };
                let step = rng.random_range(1..=max_step) as i64;

                // Apply mutation and clamp
                *gene = (*gene + direction * step).clamp(0, bounds.divisor as i64 - 1);
            }
        }
        Ok(())
    }
}

/// Collection of genomes forming one generation
#[derive(Debug)]
pub struct Population {
    pub(crate) id: Uuid,
    pub(crate) morphology: Rc<Morphology>,
    pub(crate) individuals: Vec<Genome>,
}

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("FitnessMismatch: expected {population_size}, for {fitness_count}")]
    FitnessMismatch {
        population_size: usize,
        fitness_count: usize,
    },
    #[error("crossover failed: {0}")]
    CrossoverError(#[from] CrossoverError),
    #[error("percentage out of range: {0}")]
    PercentageError(#[from] PercentageError),
}

/// Core trait for encoding/decoding an individual to/from genes
pub trait Individual: Send + Sync {
    /// Returns the bounds for each gene in this individual's genetic representation
    fn bounds(&self) -> Result<Vec<GeneBounds>, BoundsError>;

    /// Creates a new instance from a slice of gene values
    fn from_genes(&self, genes: &[f64]) -> Self
    where
        Self: Sized;

    /// Converts this individual into a vector of gene values
    fn to_genes(&self) -> Vec<f64>;
}

impl Population {
    /// Creates initial population with random genomes
    pub fn random(rng: &mut ThreadRng, size: usize, morphology: Rc<Morphology>) -> Population {
        let individuals = (0..size)
            .map(|_| Genome {
                id: Uuid::now_v7(),
                genes: morphology.random(rng),
                morphology: Rc::clone(&morphology),
            })
            .collect();

        Population {
            id: Uuid::now_v7(),
            morphology,
            individuals,
        }
    }
    /// Evolves population using provided fitness scores
    pub fn next_generation(
        &self,
        rng: &mut ThreadRng,
        fitness: Vec<f64>,
        temperature: u8,
        mutation_rate: u8,
    ) -> Result<Population, SelectionError> {
        // Validate that the number of fitness values matches the population size
        if fitness.len() != self.individuals.len() {
            return Err(SelectionError::FitnessMismatch {
                population_size: self.individuals.len(),
                fitness_count: fitness.len(),
            });
        }

        // Create selection weights from fitness
        let total_fitness: f64 = fitness.iter().sum();
        let weights: Vec<f64> = fitness.iter().map(|&f| f / total_fitness).collect();

        // Create new population through selection and crossover
        let individuals = (0..self.individuals.len())
            .map(|_| {
                // Select two parents weighted by fitness
                let parent1 = self.select_parent(rng, &weights);
                let parent2 = self.select_parent(rng, &weights);

                // Create child through crossover
                let mut child = parent1.crossover(rng, &parent2)?;

                // Maybe mutate
                child.mutate(rng, temperature, mutation_rate)?;

                Ok::<Genome, SelectionError>(child)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Population {
            id: Uuid::now_v7(),
            morphology: Rc::clone(&self.morphology),
            individuals,
        })
    }

    fn select_parent<'a>(&'a self, rng: &mut ThreadRng, weights: &[f64]) -> &'a Genome {
        // Simple weighted random selection
        let mut r = rng.random_range(0.0..1.0);
        for (genome, &weight) in self.individuals.iter().zip(weights) {
            r -= weight;
            if r <= 0.0 {
                return genome;
            }
        }
        // Fallback to last (shouldn't happen with normalized weights)
        self.individuals.last().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Cube {
        x: i64,
        y: i64,
        z: i64,
    }

    impl Individual for Cube {
        fn bounds(&self) -> Result<Vec<GeneBounds>, BoundsError> {
            let bounds = GeneBounds::new(0, 100, 101)?;
            Ok(vec![bounds.clone(), bounds.clone(), bounds])
        }

        // FIXME
        // I dont like that it is f64 here! genes should be i64! Discrete steps!
        fn from_genes(&self, genes: &[f64]) -> Self {
            Self {
                x: genes[0] as i64,
                y: genes[1] as i64,
                z: genes[2] as i64,
            }
        }

        // FIXME
        // I dont like that it is f64 here! genes should be i64! Discrete steps!
        fn to_genes(&self) -> Vec<f64> {
            vec![self.x as f64, self.y as f64, self.z as f64]
        }
    }

    #[test]
    fn test_evolve_to_origin() {
        let mut rng = rand::rng();

        // Create morphology for 3D coordinates
        let morphology = Rc::new(Morphology {
            id: Uuid::now_v7(),
            bounds: vec![GeneBounds::new(0, 100, 101).unwrap(); 3],
        });

        // Create initial population
        let mut pop = Population::random(&mut rng, 100, morphology);

        // Run evolution for n generations
        for generation in 0..10 {
            // Calculate fitness for each individual
            let fitness: Vec<f64> = pop
                .individuals
                .iter()
                .map(|ind| {
                    let genes = ind.genes();
                    // Scale from 0..divisor to 0..1 for distance calculation
                    let x = genes[0] as f64 / 100.0;
                    let y = genes[1] as f64 / 100.0;
                    let z = genes[2] as f64 / 100.0;
                    let dist = (x * x + y * y + z * z).sqrt();
                    // Positive, higher-is-better fitness
                    1.0 / (1.0 + dist)
                })
                .collect();

            let best_idx = fitness
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            let best_fitness = fitness[best_idx];
            let best_genes = pop.individuals[best_idx].genes();
            println!(
                "Generation {}: best fitness {} at position ({}, {}, {})",
                generation, best_fitness, best_genes[0], best_genes[1], best_genes[2]
            );

            if (best_fitness - 1.0).abs() < 1e-10 {
                println!("Found perfect solution!");
                break;
            }

            // Create next generation
            pop = pop.next_generation(&mut rng, fitness, 50, 10).unwrap();
        }
    }
}
