use super::GeneBounds;
use const_fnv1a_hash::fnv1a_hash_str_32;
use futures::future::BoxFuture;
use std::collections::HashMap;

// For the registry to function, we probably need to contain the fitness evaluation operation within it.
// The reason is that we will not be able to return an arbitary type from the registry, since we would not know what that that would be?
pub trait Encodeable {
    const NAME: &str;
    const HASH: i32 = fnv1a_hash_str_32(Self::NAME) as i32;

    type Phenotype;

    fn morphology() -> Vec<GeneBounds>;
    fn encode(&self) -> Vec<i64>;
    fn decode(genes: &[i64]) -> Self::Phenotype;
}

pub trait Evaluator<P> {
    fn fitness<'a>(&self, phenotype: P) -> BoxFuture<'a, Result<f64, String>>;
}

pub struct Registry<'a> {
    evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'a>>,
    morphologies: HashMap<i32, Vec<GeneBounds>>,
}

impl<'a> Registry<'a> {
    pub fn new() -> Self {
        Self {
            evaluators: HashMap::new(),
            morphologies: HashMap::new(),
        }
    }

    pub fn register<T, E>(mut self, type_hash: i32, evaluator: E) -> Self
    where
        T: Encodeable + 'a,
        E: Evaluator<T::Phenotype> + 'a,
    {
        let erased = ErasedEvaluator {
            evaluator,
            decode: T::decode,
            morphology: T::morphology(),
        };

        self.evaluators.insert(type_hash, Box::new(erased));
        self
    }

    pub fn evaluate<'b>(
        &'b self,
        type_hash: i32,
        genes: &[i64],
    ) -> BoxFuture<'b, Result<f64, String>> {
        match self.evaluators.get(&type_hash) {
            Some(evaluator) => evaluator.fitness(genes),
            None => Box::pin(futures::future::ready(Err(
                "No evaluator found for type".to_string()
            ))),
        }
    }

    pub fn morphology(&self, type_hash: i32) -> Result<&[GeneBounds], String> {
        match self.evaluators.get(&type_hash) {
            Some(evaluator) => Ok(evaluator.morphology()),
            None => Err("bullshiet".to_string()),
        }
    }
}

trait TypeErasedEvaluator {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, String>>;
    fn morphology(&self) -> &[GeneBounds];
}

struct ErasedEvaluator<P, E: Evaluator<P>> {
    evaluator: E,
    decode: fn(&[i64]) -> P,
    morphology: Vec<GeneBounds>,
}

impl<P, E: Evaluator<P>> TypeErasedEvaluator for ErasedEvaluator<P, E> {
    fn fitness<'a>(&self, genes: &[i64]) -> BoxFuture<'a, Result<f64, String>> {
        let phenotype = (self.decode)(genes);
        self.evaluator.fitness(phenotype)
    }

    fn morphology(&self) -> &[GeneBounds] {
        &self.morphology
    }
}

#[cfg(test)]
mod tests {
    use super::super::gene::{Morphology, Population};
    use super::*;
    use futures::future::ready;
    use std::rc::Rc;
    use uuid::Uuid;

    #[derive(Clone)]
    struct MyEvaluator;

    pub struct Rect {
        x: i64,
        y: i64,
    }

    impl Encodeable for Rect {
        const NAME: &str = "rect";

        type Phenotype = Rect;

        fn decode(genes: &[i64]) -> Self::Phenotype {
            Rect {
                x: genes[0],
                y: genes[1],
            }
        }

        fn encode(&self) -> Vec<i64> {
            vec![self.x, self.y]
        }

        fn morphology() -> Vec<GeneBounds> {
            vec![
                GeneBounds::new(-500, 500, 1000).unwrap(), // x: -50 to 50 in 101 steps
                GeneBounds::new(-500, 500, 1000).unwrap(), // y: -50 to 50 in 101 steps
            ]
        }
    }

    impl Evaluator<Rect> for MyEvaluator {
        fn fitness<'a>(&self, rect: Rect) -> BoxFuture<'a, Result<f64, String>> {
            // Simple fitness: distance from origin (closer is better)
            let dist = ((rect.x.pow(2) + rect.y.pow(2)) as f64).sqrt();
            // Convert to a maximization problem (higher is better)
            Box::pin(ready(Ok(1.0 / (1.0 + dist))))
        }
    }

    pub struct Cube {
        x: i64,
        y: i64,
        z: i64,
    }

    impl Encodeable for Cube {
        const NAME: &str = "cube";

        type Phenotype = Cube;

        fn decode(genes: &[i64]) -> Self::Phenotype {
            Cube {
                x: genes[0],
                y: genes[1],
                z: genes[2],
            }
        }

        fn encode(&self) -> Vec<i64> {
            vec![self.x, self.y, self.z]
        }

        fn morphology() -> Vec<GeneBounds> {
            let x_bounds = GeneBounds::new(0, 1000, 1000).unwrap();
            let y_bounds = GeneBounds::new(0, 1000, 1000).unwrap();
            let z_bounds = GeneBounds::new(0, 1000, 1000).unwrap();
            vec![x_bounds, y_bounds, z_bounds]
        }
    }

    impl Evaluator<Cube> for MyEvaluator {
        fn fitness<'a>(&self, cube: Cube) -> BoxFuture<'a, Result<f64, String>> {
            // Similar fitness: distance from origin (closer is better)
            let dist = ((cube.x.pow(2) + cube.y.pow(2) + cube.z.pow(2)) as f64).sqrt();
            Box::pin(ready(Ok(1.0 / (1.0 + dist))))
        }
    }

    #[tokio::test]
    async fn it_optimizes_different_types() {
        let mut rng = rand::rng();
        let my_evaluator = MyEvaluator;

        let registry = Registry::new()
            .register::<Cube, _>(Cube::HASH, my_evaluator.clone())
            .register::<Rect, _>(Rect::HASH, my_evaluator);

        // First optimize rectangle (find point closest to origin)
        println!("\nOptimizing Rectangle (2D):");
        let rect_morphology = Rc::new(Morphology {
            id: Uuid::now_v7(),
            bounds: Rect::morphology(),
        });
        let mut rect_pop = Population::random(&mut rng, 10, rect_morphology);

        for generation in 0..100 {
            let mut fitness = Vec::new();
            for individual in rect_pop.individuals.iter() {
                let fit = registry
                    .evaluate(Rect::HASH, individual.genes())
                    .await
                    .unwrap();
                fitness.push(fit);
            }

            let best_idx = fitness
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            let best_fitness = fitness[best_idx];
            let best_genes = rect_pop.individuals.iter().nth(best_idx).unwrap().genes();
            println!(
                "Generation {}: best fitness {} at position ({}, {})",
                generation, best_fitness, best_genes[0], best_genes[1]
            );

            if (best_fitness - 1.0).abs() < 1e-10 {
                println!("Found perfect solution!");
                break;
            }

            rect_pop = rect_pop.next_generation(&mut rng, fitness, 50, 10).unwrap();
        }

        // Then optimize cube (find point closest to origin)
        println!("\nOptimizing Cube (3D):");
        let cube_morphology = Rc::new(Morphology {
            id: Uuid::now_v7(),
            bounds: Cube::morphology(),
        });
        let mut cube_pop = Population::random(&mut rng, 10, cube_morphology);

        for generation in 0..100 {
            let mut fitness = Vec::new();
            for individual in cube_pop.individuals.iter() {
                let fit = registry
                    .evaluate(Cube::HASH, individual.genes())
                    .await
                    .unwrap();
                fitness.push(fit);
            }

            let best_idx = fitness
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            let best_fitness = fitness[best_idx];
            let best_genes = cube_pop.individuals.iter().nth(best_idx).unwrap().genes();
            println!(
                "Generation {}: best fitness {} at position ({}, {}, {})",
                generation, best_fitness, best_genes[0], best_genes[1], best_genes[2]
            );

            if (best_fitness - 1.0).abs() < 1e-10 {
                println!("Found perfect solution!");
                break;
            }

            cube_pop = cube_pop.next_generation(&mut rng, fitness, 50, 10).unwrap();
        }
    }
}
