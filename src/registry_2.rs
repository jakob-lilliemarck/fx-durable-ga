use futures::future::BoxFuture;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
};
use thiserror::Error;

pub trait Gene: Send + Sync + Any {
    /// Generate a random gene of this type
    fn generate_random(&self) -> Box<dyn Gene>;
    /// Mutate this gene in-place
    fn mutate(&mut self);
    /// Crossover with another gene of the same type
    fn crossover(&self, other: &dyn Gene) -> Box<dyn Gene>;
}

impl Gene for f64 {
    fn crossover(&self, other: &dyn Gene) -> Box<dyn Gene> {
        todo!()
    }

    fn mutate(&mut self) {
        todo!()
    }

    fn generate_random(&self) -> Box<dyn Gene> {
        todo!()
    }
}

impl Gene for bool {
    fn crossover(&self, other: &dyn Gene) -> Box<dyn Gene> {
        todo!()
    }

    fn mutate(&mut self) {
        todo!()
    }

    fn generate_random(&self) -> Box<dyn Gene> {
        todo!()
    }
}

pub type Genome = Vec<Box<dyn Gene>>;

// Trait for types that can be optimized
pub trait Phenotype: 'static + Send + Sync + Sized {
    /// Convert from genome to concrete phenotype
    fn from_genome(genome: &Genome) -> Result<Self, RegistryError>;

    /// Convert from concrete phenotype to genome
    fn to_genome(&self) -> Genome;
}

// Type-erased fitness function
pub trait FitnessFunction: Send + Sync {
    fn evaluate<'a>(&'a self, genome: &'a Genome) -> BoxFuture<'a, Result<f64, RegistryError>>;
    fn type_id(&self) -> TypeId;
}

// Concrete fitness function wrapper
pub struct TypedFitnessFunction<T, F> {
    func: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> FitnessFunction for TypedFitnessFunction<T, F>
where
    T: Phenotype,
    F: Fn(T) -> BoxFuture<'static, f64> + Send + Sync,
{
    fn evaluate<'a>(&'a self, genome: &'a Genome) -> BoxFuture<'a, Result<f64, RegistryError>> {
        Box::pin(async move {
            // Convert genome to phenotype first, THEN call user function
            let phenotype = T::from_genome(genome)?;
            let fitness = (self.func)(phenotype).await;
            Ok(fitness)
        })
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
}

pub struct Registry {
    fitness_functions: HashMap<TypeId, Box<dyn FitnessFunction>>,
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Could not convert phenotype")]
    Conversion,
    #[error("Type not registered: {type_name}")]
    TypeNotRegistered { type_name: &'static str },
}

impl Registry {
    pub fn new() -> Self {
        Self {
            fitness_functions: HashMap::new(),
        }
    }

    /// Register a phenotype type along with its fitness function
    /// The fitness function takes the concrete phenotype type, not genome
    pub fn register<T, F>(&mut self, fitness_fn: F) -> Result<(), RegistryError>
    where
        T: Phenotype,
        F: Fn(T) -> BoxFuture<'static, f64> + Send + Sync + 'static,
    {
        let type_id = TypeId::of::<T>();
        let fitness_function = TypedFitnessFunction {
            func: fitness_fn,
            _phantom: std::marker::PhantomData::<T>,
        };

        self.fitness_functions
            .insert(type_id, Box::new(fitness_function));
        Ok(())
    }

    /// Evaluate fitness for a genome using registered function
    /// Internally converts genome -> phenotype -> calls user function
    pub async fn evaluate_fitness(
        &self,
        type_id: TypeId,
        genome: &Genome,
    ) -> Result<f64, RegistryError> {
        let fitness_fn =
            self.fitness_functions
                .get(&type_id)
                .ok_or(RegistryError::TypeNotRegistered {
                    type_name: "unknown",
                })?;

        fitness_fn.evaluate(genome).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test phenotype 1: Simple struct with one f64 field
    #[derive(Debug, Clone, PartialEq)]
    struct SimpleFloat {
        value: f64,
    }

    impl Phenotype for SimpleFloat {
        fn from_genome(genome: &Genome) -> Result<Self, RegistryError> {
            if genome.len() != 1 {
                return Err(RegistryError::Conversion);
            }

            let gene = genome[0].as_ref();
            let any_ref = gene as &dyn Any;
            let value = any_ref
                .downcast_ref::<f64>()
                .ok_or(RegistryError::Conversion)?;

            Ok(SimpleFloat { value: *value })
        }

        fn to_genome(&self) -> Genome {
            vec![Box::new(self.value)]
        }
    }

    // Test phenotype 2: Struct with bool and f64 fields
    #[derive(Debug, Clone, PartialEq)]
    struct MixedType {
        flag: bool,
        weight: f64,
    }

    impl Phenotype for MixedType {
        fn from_genome(genome: &Genome) -> Result<Self, RegistryError> {
            if genome.len() != 2 {
                return Err(RegistryError::Conversion);
            }

            let flag_gene = genome[0].as_ref() as &dyn Any;
            let weight_gene = genome[1].as_ref() as &dyn Any;

            let flag = flag_gene
                .downcast_ref::<bool>()
                .ok_or(RegistryError::Conversion)?;
            let weight = weight_gene
                .downcast_ref::<f64>()
                .ok_or(RegistryError::Conversion)?;

            Ok(MixedType {
                flag: *flag,
                weight: *weight,
            })
        }

        fn to_genome(&self) -> Genome {
            vec![Box::new(self.flag), Box::new(self.weight)]
        }
    }

    #[tokio::test]
    async fn test_registry_with_both_types() {
        let mut registry = Registry::new();

        // Register SimpleFloat
        registry
            .register::<SimpleFloat, _>(|phenotype: SimpleFloat| {
                Box::pin(async move {
                    // Simple fitness: just return the value
                    phenotype.value
                })
            })
            .unwrap();

        let test = 1;
        // Register MixedType
        registry
            .register::<MixedType, _>(move |phenotype: MixedType| {
                Box::pin(async move {
                    let _test_2 = test + 1; // underscore prefix to avoid unused warning
                    // Fitness: if flag is true, return weight, otherwise return negative weight
                    if phenotype.flag {
                        phenotype.weight
                    } else {
                        -phenotype.weight
                    }
                })
            })
            .unwrap();

        // Test SimpleFloat
        let simple = SimpleFloat { value: 42.5 };
        let simple_genome = simple.to_genome();
        let simple_fitness = registry
            .evaluate_fitness(TypeId::of::<SimpleFloat>(), &simple_genome)
            .await
            .unwrap();
        assert_eq!(simple_fitness, 42.5);

        // Test MixedType with flag=true
        let mixed1 = MixedType {
            flag: true,
            weight: 10.0,
        };
        let mixed1_genome = mixed1.to_genome();
        let mixed1_fitness = registry
            .evaluate_fitness(TypeId::of::<MixedType>(), &mixed1_genome)
            .await
            .unwrap();
        assert_eq!(mixed1_fitness, 10.0);

        // Test MixedType with flag=false
        let mixed2 = MixedType {
            flag: false,
            weight: 5.0,
        };
        let mixed2_genome = mixed2.to_genome();
        let mixed2_fitness = registry
            .evaluate_fitness(TypeId::of::<MixedType>(), &mixed2_genome)
            .await
            .unwrap();
        assert_eq!(mixed2_fitness, -5.0);
    }

    #[tokio::test]
    async fn test_move_closure_with_external_resources() {
        let mut registry = Registry::new();

        let test = 1;
        // Register MixedType with move closure capturing external variable
        registry
            .register::<MixedType, _>(move |phenotype: MixedType| {
                Box::pin(async move {
                    let test_2 = test + 1;
                    // Fitness: use external resource in calculation
                    let base_fitness = if phenotype.flag {
                        phenotype.weight
                    } else {
                        -phenotype.weight
                    };
                    base_fitness + test_2 as f64
                })
            })
            .unwrap();

        let mixed = MixedType {
            flag: true,
            weight: 10.0,
        };
        let genome = mixed.to_genome();
        let fitness = registry
            .evaluate_fitness(TypeId::of::<MixedType>(), &genome)
            .await
            .unwrap();

        // Expected: 10.0 + (1 + 1) = 12.0
        assert_eq!(fitness, 12.0);
    }
}
