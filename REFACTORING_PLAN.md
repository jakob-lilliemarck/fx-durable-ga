# Refactoring Plan: Transition to Generic Evolutionary Framework

**Status:** Phase 1 & 2 Complete (Foundation & Persistence Layer)

## Overview
This plan outlines a **breaking change** to refactor `fx-durable-ga` from a specialized numeric optimization engine into a generic evolutionary framework. The goal is to support any user-defined genotype (GP trees, hybrid types, etc.) by removing the dependency on `Vec<i64>` genomes.

The existing `genotypes` table, `morphology` system, and associated repository code will be completely replaced.

## Core Philosophy
Shift from **Data-Driven Definition** (Morphology rules in DB) to **Code-Driven Definition** (Rust's type system defines rules at compile time). The framework becomes ignorant of the concrete types it manages.

## Architectural Changes

### 1. The `GenotypeManager` Trait: A Single Point of Integration
The core of the architecture is a single, type-erased trait that the framework uses to perform all evolutionary operations. This is the sole integration point for the user.

*   **User's Responsibility**: The user provides one stateless implementation of this trait for each concrete genotype they wish to evolve.
*   **Framework's Responsibility**: The framework interacts *only* with this trait via a registry, remaining completely ignorant of the user's concrete data types.

```rust
// Implemented by the user as a stateless "manager" for their type.
// All methods operate on `serde_json::Value` to hide the concrete type from the framework.
use futures::future::BoxFuture;
use anyhow::Result;

pub trait GenotypeManager: Send + Sync {
    /// Generate a new, random genome as a JSON Value.
    fn random(&self, rng: &mut impl Rng) -> Value;
    
    /// Perform crossover on two JSON values, returning a new JSON child.
    fn crossover(&self, parent1: &Value, parent2: &Value, rng: &mut impl Rng) -> Result<Value>;

    /// Mutate a JSON genome in place.
    fn mutate(&self, genotype: &mut Value, rng: &mut impl Rng, mutation_rate: f64, temperature: f64) -> Result<()>;
    
    /// Evaluate the fitness of a JSON genome.
    fn evaluate<'a>(&'a self, genotype: &'a Value) -> BoxFuture<'a, Result<f64>>;
}
```
The user is free to implement the logic inside these methods however they choose.

### 2. Refactored `OptimizationService` and Registry
*   The existing **`OptimizationService` will be directly refactored** and will remain **non-generic**.
*   It will hold a single registry: `HashMap<i32, Box<dyn GenotypeManager>>`.
*   The `service_builder` will be updated with a `with_genotype_manager(...)` method to populate this registry.
*   When a job is received, the service uses the `type_hash` to look up the correct `GenotypeManager` from the registry and calls its type-erased methods.

### 3. Unified, Type-Erased Persistence
*   **`Genotype` Struct**: The primary `Genotype` struct will be refactored to store the genome as `genome_data: serde_json::Value`.
*   **Database**: The `genotypes` table will be replaced with a new version that has a `genome_data JSONB` column.
*   **Repository**: The existing `genotypes` repository will be replaced with a new implementation that works with the refactored `Genotype` struct and the new table schema.

## Implementation Steps & Progress

### Phase 1: Foundation (Complete)
- [x] **Define Trait**: `src/models/evolution.rs` created with the `GenotypeManager` trait.
- [x] **Type-Erased Model**: `src/models/generic_genotype.rs` created as a prototype.
- [x] **Migration**: A new migration has been created for a parallel `generic_genotypes` table.

### Phase 2: Persistence (Complete)
- [x] **Parallel Repository**: A new, separate `generic_genotypes` repository has been implemented.
- [x] **Validation**: A roundtrip test has been written and passed, proving the persistence strategy.

### Phase 3: Unification & Refactoring (Next Steps)
1.  **Rename & Replace**:
    *   Rename the `generic_genotypes` migration, table, model (`GenericGenotype` -> `Genotype`), and repository to become the new primary `genotypes` implementation.
    *   Delete the old `genotypes` repository, the old `Genotype` struct, and create a migration to drop the old `genotypes` table.
2.  **Delete Obsolete Code**:
    *   Remove the `Morphology` model, repository, and database table.
    *   Remove the `Crossover` and `Mutagen` models' internal logic, retaining them only as configuration DTOs.
3.  **Refactor `OptimizationService`**:
    *   Update the service to use the new `genotypes` repository.
    *   Implement the `GenotypeManager` registry.
    *   Rewrite the core methods (`generate_initial_population`, `breed_next_generation`, etc.) to dispatch all evolutionary logic to the appropriate `GenotypeManager` from the registry.
4.  **Update Tests & Examples**: Update all existing tests and examples to use the new `GenotypeManager`-based workflow.

## Key Insights
*   **Breaking Change Simplifies**: By not supporting the legacy system, the refactoring is cleaner and avoids a complex transition period.
*   **Type Erasure is Key**: The `GenotypeManager` trait, operating on `serde_json::Value`, is the essential abstraction that allows a non-generic service to handle multiple, user-defined types.
*   **Framework Boundary**: The framework handles "Persistence" and "Orchestration". The consuming application, via its `GenotypeManager` implementation, handles all "Structure" and "Evolution Logic".
