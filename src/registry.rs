use crate::gene::{GeneBounds, Individual};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use thiserror::Error;
use uuid::Uuid;

/// Registry entry for a concrete Individual type
struct Entry {
    individual: Box<dyn Individual>,
    fitness: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
}

/// Type registry to resolve Individual instances and fitness functions by type_id
pub struct Registry {
    entries: Mutex<HashMap<Uuid, Arc<Entry>>>,
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("type id not registered: {0}")]
    TypeNotRegistered(Uuid),
}

impl Registry {
    /// Create empty registry
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Register an Individual type with a fitness function
    ///
    /// - `type_id`: unique identifier for the type
    /// - `individual`: prototype instance used for bounds() and from_genes()
    /// - `fitness`: function evaluating fitness from gene values
    pub fn register(
        &self,
        type_id: Uuid,
        individual: Box<dyn Individual>,
        fitness: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    ) {
        let mut map = self.entries.lock().unwrap();
        map.insert(
            type_id,
            Arc::new(Entry {
                individual,
                fitness,
            }),
        );
    }

    /// Get the bounds for a registered individual type
    pub fn bounds(&self, type_id: &Uuid) -> Result<Vec<GeneBounds>, RegistryError> {
        let map = self.entries.lock().unwrap();
        map.get(type_id)
            .map(|e| e.individual.bounds())
            .ok_or_else(|| RegistryError::TypeNotRegistered(*type_id))
    }

    /// Evaluate fitness for gene values
    pub fn evaluate_fitness(&self, type_id: &Uuid, genes: &[f64]) -> Result<f64, RegistryError> {
        let map = self.entries.lock().unwrap();
        map.get(type_id)
            .map(|e| (e.fitness)(genes))
            .ok_or_else(|| RegistryError::TypeNotRegistered(*type_id))
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

    impl Cube {
        fn new() -> Self {
            Self { x: 0, y: 0, z: 0 }
        }
    }

    impl Individual for Cube {
        fn bounds(&self) -> Vec<GeneBounds> {
            let bounds = GeneBounds::new(0, 100, 101).unwrap();
            vec![bounds.clone(), bounds.clone(), bounds]
        }

        fn from_genes(&self, genes: &[f64]) -> Self {
            Self {
                x: genes[0] as i64,
                y: genes[1] as i64,
                z: genes[2] as i64,
            }
        }

        fn to_genes(&self) -> Vec<f64> {
            vec![self.x as f64, self.y as f64, self.z as f64]
        }
    }

    #[test]
    fn test_registry_register_and_use() {
        let registry = Registry::new();
        let type_id = Uuid::now_v7();

        // Create prototype cube and fitness function
        let prototype = Box::new(Cube::new());
        let fitness = Box::new(|genes: &[f64]| {
            // Fitness: negative distance from origin (maximize)
            -((genes[0] * genes[0] + genes[1] * genes[1] + genes[2] * genes[2]).sqrt())
        });

        // Register cube type
        registry.register(type_id, prototype, fitness);

        // Test cube at (30,40,0) -> distance=50 in gene space
        let test_genes = vec![30.0, 40.0, 0.0];
        let fit = registry.evaluate_fitness(&type_id, &test_genes).unwrap();

        // Should be -50.0 (negative distance)
        assert!((fit + 50.0).abs() < 1e-6);

        // Verify bounds
        let bounds = registry.bounds(&type_id).unwrap();
        assert_eq!(bounds.len(), 3);
        for bound in bounds {
            assert_eq!(bound.start, 0);
            assert_eq!(bound.end, 100);
            assert_eq!(bound.divisor, 101);
        }
    }
}
