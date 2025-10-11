use uuid::Uuid;

/// An optimization was requested
pub struct OptimizationRequested {
    optimization_id: Uuid,
}

/// A genotypes was generated
pub struct GenotypeGenerated {
    optimization_id: Uuid,
    genotype_id: Uuid,
}

/// A phenotype was successfully evaluated
pub struct PhenotypeEvaluated {
    optimization_id: Uuid,
    genotype_id: Uuid,
}

/// The optimization reached its optimization goal or ran termination condition
pub struct OptimizationCompleted {
    optimization_id: Uuid,
}
