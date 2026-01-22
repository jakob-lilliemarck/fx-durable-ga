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
use const_fnv1a_hash::fnv1a_hash_str_32;

pub trait GenotypeManager: Send + Sync {
    /// Unique name identifier for the type being managed.
    fn name(&self) -> &'static str;
    
    /// Hash derived from the name for efficient type identification.
    /// Default implementation provided using FNV-1a.
    fn hash(&self) -> i32 {
        fnv1a_hash_str_32(self.name()) as i32
    }

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

This phase is a **breaking change** that refactors the `OptimizationService` and its dependencies to be driven by the `GenotypeManager` registry.

---

#### **Step 3.1: Update `GenotypeManager` Trait**
*   **File:** `src/models/evolution.rs`
*   **Action:** Add `name()` and `hash()` methods to the `GenotypeManager` trait. This encapsulates the type identity logic previously handled by the `Encodeable` trait, allowing us to eventually remove `Encodeable`.

---

#### **Step 3.2: Refactor Core Models for Generality**

*   **File:** `src/models/request.rs`
    *   **Action:** Remove `crossover` and `distribution` fields. The `new()` function will need to be adjusted.
        ```diff
        pub struct Request {
            // ...
        -   pub(crate) crossover: Crossover,
        -   pub(crate) distribution: Distribution,
            // ...
        }
        ```
    *   **Rationale:** These configurations are obsolete in the new architecture. `GenotypeManager` owns the logic for crossover and initial population generation (`random`), so the `Request` no longer needs to carry this configuration.

*   **File:** `src/models/selector.rs`
    *   **Action:** Refactor `select_parents` to be agnostic of the `Genotype` struct. It should operate on a simpler slice.
        ```rust
        // Define a simple struct for selection
        pub struct FitnessRecord {
            pub id: Uuid,
            pub fitness: f64,
        }

        // Change the signature
        pub fn select_parents<'a>(
            &self,
            num_pairs: usize,
        -   candidates: &'a [(Genotype, Option<f64>)],
        +   candidates: &'a [FitnessRecord],
            goal: &FitnessGoal,
        ) -> Result<Vec<(&'a FitnessRecord, &'a FitnessRecord)>, SelectionError>
        ```
    *   **Rationale:** The selection algorithm only needs the ID and fitness. This decouples `Selector` from any specific `Genotype` implementation.

*   **File:** `src/models/breeder.rs`
    *   **Action:** **Delete this file.**
    *   **Rationale:** The logic in `Breeder` is a simple loop that calls crossover and mutate. This logic will be moved directly into a private method within the `OptimizationService` and will dispatch to the `GenotypeManager`.

---

#### **Step 3.3: Refactor the `OptimizationService`**

*   **File:** `src/services/optimization/service_builder.rs`
    *   **Action 1:** Add the `GenotypeManager` registry.
        ```diff
        pub struct ServiceBuilder {
            // ...
            pub(super) evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
        +   pub(super) genotype_managers: HashMap<i32, Box<dyn GenotypeManager + 'static>>,
            // ...
        }
        ```
    *   **Action 2:** Create the `with_genotype_manager` registration method.
        ```rust
        #[instrument(level = "debug", skip(self, manager), fields(type_name = manager.name(), type_hash = manager.hash()))]
        pub fn with_genotype_manager<M>(mut self, manager: M) -> Self
        where
            M: GenotypeManager + 'static,
        {
            self.genotype_managers.insert(manager.hash(), Box::new(manager));
            self
        }
        ```
    *   **Action 3:** Update the `build()` method to pass the new manager registry to the `Service`.
    *   **Action 4:** Deprecate the old `.register()` method in favor of the new manager.

*   **File:** `src/services/optimization/service.rs`
    *   **Action 1:** Update the `Service` struct.
        ```diff
        pub struct Service {
            // ...
            pub(super) genotypes: genotypes::Repository, // This will be the NEW repository
        -   pub(super) evaluators: HashMap<i32, Box<dyn TypeErasedEvaluator + 'static>>,
        +   pub(super) genotype_managers: HashMap<i32, Box<dyn GenotypeManager + 'static>>,
            // ...
        }
        ```
    *   **Action 2: Refactor `generate_initial_population`**
        *   Remove the call to `morphologies.get_morphology`.
        *   Look up the `type_hash` in `self.genotype_managers`.
        *   Call `manager.random()` in a loop based on `request.distribution`.
        *   Create `GenericGenotype` instances from the returned `Value`.
        *   Save them using the new `genotypes` repository.
    *   **Action 3: Refactor `breed_genotypes`**
        *   This method will now contain the logic previously in `Breeder`.
        *   It will fetch parent `GenericGenotype`s.
        *   It will look up the `GenotypeManager`.
        *   In a loop, it will deserialize parent `genome_data` to `Value`, call `manager.crossover` and `manager.mutate`, and create new `GenericGenotype` children to be saved.
    *   **Action 4: Refactor `evaluate_genotype`**
        *   Remove the lookup in `self.evaluators`.
        *   Look up the `GenotypeManager` using the `type_hash`.
        *   Call `manager.evaluate(&genotype.genome_data)`.
        *   The concept of `TypeErasedEvaluator` is now obsolete and can be removed.

---

#### **Step 3.4: Unification and Deletion**

1.  **Rename & Replace**:
    *   Delete `src/repositories/genotypes`.
    *   Rename `src/repositories/generic_genotypes` to `src/repositories/genotypes`.
    *   Delete `src/models/genotype.rs`.
    *   Rename `src/models/generic_genotype.rs` to `src/models/genotype.rs` and `GenericGenotype` to `Genotype`.
2.  **Delete Obsolete Code**:
    *   Delete `src/models/morphology.rs`, `src/repositories/morphologies`, and the database table.
    *   Delete the `Encodeable` trait.
    *   Refactor `Crossover` and `Mutagen` to be simple DTOs with no logic, or remove them entirely if the `GenotypeManager` is expected to handle all configuration internally.
3.  **Update Tests & Examples**: Update all broken tests and examples to use the new `GenotypeManager`-based workflow.

## Key Insights
*   **Breaking Change Simplifies**: By not supporting the legacy system, the refactoring is cleaner and avoids a complex transition period.
*   **Type Erasure is Key**: The `GenotypeManager` trait, operating on `serde_json::Value`, is the essential abstraction that allows a non-generic service to handle multiple, user-defined types.
*   **Framework Boundary**: The framework handles "Persistence" and "Orchestration". The consuming application, via its `GenotypeManager` implementation, handles all "Structure" and "Evolution Logic".
