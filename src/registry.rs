use std::any::{Any, TypeId};
use std::boxed::Box;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

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

#[derive(Debug)]
pub struct ConversionError(pub String);

/// A phenotype is the observable result of a genotype
pub trait Phenotype: Sized {
    /// Convert a genome into this phenotype
    fn from_genes<T>(genes: &[Box<dyn Gene>]) -> Result<Self, ConversionError>
    where
        Self: Sized;

    /// Define the genome template for this type
    fn to_genes(&self) -> Vec<Box<dyn Gene>>;
}

/// The collection of genes of a type
pub struct Genome<T: Phenotype> {
    pub genome: Vec<Box<dyn Gene>>,
    pub concrete: PhantomData<T>,
}

/// Type-erased trait for converting genes to concrete phenotype types
pub trait PhenotypeConverter: Send + Sync {
    /// Convert genes to a concrete phenotype type (boxed as Any)
    fn from_genes(&self, genes: &[Box<dyn Gene>]) -> Result<Box<dyn Any + Send + Sync>, String>;

    /// Get the type name for debugging
    fn type_name(&self) -> &'static str;
}

/// Concrete converter for a specific phenotype type
struct TypeConverter<P> {
    _phantom: PhantomData<P>,
}

impl<P> TypeConverter<P> {
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<P> PhenotypeConverter for TypeConverter<P>
where
    P: Phenotype + Send + Sync + 'static,
{
    fn from_genes(&self, genes: &[Box<dyn Gene>]) -> Result<Box<dyn Any + Send + Sync>, String> {
        let phenotype = P::from_genes::<P>(genes).map_err(|e| e.0)?; // ConversionError contains a String
        Ok(Box::new(phenotype))
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<P>()
    }
}

/// Registry for phenotype type converters
pub struct PhenotypeRegistry {
    converters: HashMap<u64, Box<dyn PhenotypeConverter>>,
}

impl PhenotypeRegistry {
    pub fn new() -> Self {
        Self {
            converters: HashMap::new(),
        }
    }

    /// Register a phenotype type for conversion
    pub fn register<P>(&mut self)
    where
        P: Phenotype + Send + Sync + 'static,
    {
        let type_hash = Self::type_hash::<P>();
        self.converters
            .insert(type_hash, Box::new(TypeConverter::<P>::new()));
    }

    /// Convert genes to a specific phenotype type
    pub fn convert<P: 'static>(&self, genes: &[Box<dyn Gene>]) -> Result<P, String> {
        let type_hash = Self::type_hash::<P>();
        let converter = self
            .converters
            .get(&type_hash)
            .ok_or("Type not registered")?;

        let any_box = converter.from_genes(genes)?;
        let concrete = any_box.downcast::<P>().map_err(|_| "Type mismatch")?;

        Ok(*concrete)
    }

    /// Convert genes using type hash (when you don't know the type at compile time)
    pub fn convert_by_hash(
        &self,
        type_hash: u64,
        genes: &[Box<dyn Gene>],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        let converter = self
            .converters
            .get(&type_hash)
            .ok_or("Type not registered")?;
        converter.from_genes(genes)
    }

    /// Generate consistent hash for type
    pub fn type_hash<T: 'static>() -> u64 {
        let type_id = TypeId::of::<T>();
        let mut hasher = DefaultHasher::new();
        type_id.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    pub struct Example {
        x: f64,
        y: f64,
        flip: bool,
    }

    impl Phenotype for Example {
        fn from_genes<T>(genes: &[Box<dyn Gene>]) -> Result<Self, ConversionError>
        where
            Self: Sized,
        {
            if genes.len() != 3 {
                return Err(ConversionError("Expected 3 genes".to_string()));
            }

            // Use Any downcasting to extract values (fx-event-bus pattern)
            let any_ref = genes[0].as_ref() as &(dyn Any + '_);
            let x = any_ref
                .downcast_ref::<f64>()
                .ok_or_else(|| ConversionError("Gene 0 is not f64".to_string()))?;

            let any_ref = genes[1].as_ref() as &(dyn Any + '_);
            let y = any_ref
                .downcast_ref::<f64>()
                .ok_or_else(|| ConversionError("Gene 1 is not f64".to_string()))?;

            let any_ref = genes[2].as_ref() as &(dyn Any + '_);
            let flip = any_ref
                .downcast_ref::<bool>()
                .ok_or_else(|| ConversionError("Gene 2 is not bool".to_string()))?;

            Ok(Example {
                x: *x,
                y: *y,
                flip: *flip,
            })
        }

        fn to_genes(&self) -> Vec<Box<dyn Gene>> {
            let mut genome: Vec<Box<dyn Gene>> = Vec::with_capacity(3);
            genome.push(Box::new(self.x));
            genome.push(Box::new(self.y));
            genome.push(Box::new(self.flip));

            genome
        }
    }

    #[test]
    fn it_transforms_phenotype_to_genotype_to_phenotype() {
        // Create original phenotype
        let original = Example {
            x: 1.5,
            y: -2.3,
            flip: true,
        };

        // Register the type
        let mut registry = PhenotypeRegistry::new();
        registry.register::<Example>();

        // Transform: phenotype -> genotype
        let genes = original.to_genes();

        // Transform: genotype -> phenotype (via registry)
        let reconstructed: Example = registry.convert(&genes).unwrap();

        // Verify they match
        assert_eq!(original.x, reconstructed.x);
        assert_eq!(original.y, reconstructed.y);
        assert_eq!(original.flip, reconstructed.flip);
    }
}
