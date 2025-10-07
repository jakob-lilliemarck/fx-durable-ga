use uuid::Uuid;

/// An optimization was requested
pub struct OptimizationRequested {
    optimization_id: Uuid,
}

/// A generation of genotypes was generated
pub struct GenerationGenerated {
    optimization_id: Uuid,
}

/// The fitness of a phenotype was successfully computed
pub struct PhenotypeEvaluated {
    optimization_id: Uuid,
}

/// The optimization reached its optimization goal, or ran until the termination condition was met
pub struct OptimizationCompleted {
    optimization_id: Uuid,
}
