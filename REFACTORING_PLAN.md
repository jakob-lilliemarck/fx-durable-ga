# Refactoring Plan: Transition to Generic Evolutionary Framework

**Status:** Phase 1 & 2 Complete (Foundation & Persistence Layer)

## Overview
This plan outlines a **breaking change** to refactor `fx-durable-ga` from a specialized numeric optimization engine into a generic evolutionary framework. The goal is to support any user-defined genotype (GP trees, hybrid types, etc.) by removing the dependency on `Vec<i64>` genomes.

The existing `genotypes` table, `morphology` system, and associated repository code will be completely replaced; the canonical table remains `genotypes` (no separate `generic_genotypes` table).

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
    
    /// Evaluate the fitness of a JSON genome with termination support.
    fn evaluate<'a>(
        &'a self,
        genotype: &'a Value,
        terminated: &'a dyn Terminated,
    ) -> BoxFuture<'a, Result<f64>>;
}
```
The user is free to implement the logic inside these methods however they choose.

### 2. Refactored `OptimizationService` and Registry
*   The existing **`OptimizationService` will be directly refactored** and will remain **non-generic**.
*   It will hold a single registry: `HashMap<i32, Box<dyn GenotypeManager>>`.
*   The `service_builder` will be updated with a `with_genotype_manager(...)` method to populate this registry.
*   When a job is received, the service uses the `type_hash` to look up the correct `GenotypeManager` from the registry and calls its type-erased methods.

### 3. Unified, Type-Erased Persistence
*   **`Genotype` Struct**: The primary `Genotype` struct will be refactored to store the genome as `genome: serde_json::Value` (same field name, new JSON representation).
*   **Database**: The `genotypes` table will be altered in place so `genome` becomes `JSONB`.
*   **Repository**: The existing `genotypes` repository will be updated to the new schema; the prototype `generic_genotypes` repository will be removed.

## Implementation Steps & Progress

### Phase 1: Foundation (Complete)
- [x] **Define Trait**: `src/models/evolution.rs` created with the `GenotypeManager` trait.
- [x] **Type-Erased Model**: `src/models/generic_genotype.rs` created as a prototype.
- [x] **Migration**: A new migration has been created for a parallel `generic_genotypes` table.

### Phase 2: Persistence (Complete)
- [x] **Repository Prototype**: A JSONB-backed repository was implemented (initially under `generic_genotypes`) and validated via roundtrip test.
- [x] **Migration Prototype**: An initial migration created `generic_genotypes`; this will now be rewritten to alter `genotypes` in place to the JSONB schema.

### Phase 3: Unification & Refactoring (Next Steps)

This phase is a **breaking change** that refactors the `OptimizationService` and its dependencies to be driven by the `GenotypeManager` registry.

---

#### **Step 3.1: Validate `GenotypeManager` Trait**
*   **File:** `src/models/evolution.rs`
*   **Action:** Ensure `name()`/`hash()` are present (already added) so type identity lives in the manager. Extend `evaluate` to accept a termination handle (`&dyn Terminated`) so cancellation remains supported. No other trait changes planned for this refactor.

---

#### **Step 3.2: Core Models**

*   **Request** (`src/models/request.rs`): **No changes.** Keep existing fields (`crossover`, `distribution`, `mutagen`, etc.). Even if `distribution` is unused initially, we leave the API intact for now.
*   **Selector** (`src/models/selector.rs`): **No signature change.** Keep access to genotypes; we are not introducing `FitnessRecord` indirection in this refactor.
*   **Breeder** (`src/models/breeder.rs`): Keep the module. Refactor it to operate on JSON-based `Genotype`, taking parent references, `request.mutagen`, and a `GenotypeManager` to perform crossover/mutate. This keeps breeding logic isolated and testable without the full service.

---

#### **Step 3.3: Refactor the `OptimizationService`**

*   **File:** `src/services/optimization/service_builder.rs`
    *   Add a `genotype_managers: HashMap<i32, Box<dyn GenotypeManager>>`.
    *   Provide `with_genotype_manager` (preferred registration API). Remove/replace `.register()` and the `TypeErasedEvaluator` pathway.
    *   Drop dependencies on `Encodeable`, `Evaluator`, and `Morphology`.
    *   Pass the manager registry into `Service::new`.

*   **File:** `src/services/optimization/service.rs`
    *   Replace evaluator map with `genotype_managers`.
    *   `generate_initial_population`: use `GenotypeManager::random` via `type_hash`; respect `request.distribution` even if internally unused by the manager. Persist via the genotypes repo.
    *   `breed_genotypes`: delegate to refactored `Breeder`, passing parents, `GenotypeManager`, `request.mutagen`, and a shared `&mut rng`.
    *   `evaluate_genotype`: call `GenotypeManager::evaluate(&genotype.genome, terminated_handle)` directly (no `TypeErasedEvaluator`), preserving termination support.
    *   Ensure instrumentation with `#[instrument(level = \"debug\")]` and log only significant business events (`info!`/`warn!`/`error!` as appropriate).

---

#### **Step 3.4: Unification and Deletion**

1.  **Schema & Migration**
    * Rewrite the existing migration that introduced `generic_genotypes` to instead alter the existing `genotypes` table in place: change `genome` column type from `BIGINT[]` to `JSONB NOT NULL` (keep the column name), keep `genome_hash BIGINT` and existing indexes. Do **not** create `generic_genotypes` and do not touch `fitness` or `populations` view/indexes. Down migration restores `genome BIGINT[]` and removes the JSONB change.
2.  **Repositories & Models**
    * Keep `src/repositories/genotypes`; update its queries/models to use `genome` as JSONB. Remove the prototype `generic_genotypes` repository.
    * Keep `src/models/genotype.rs` but refactor its `genome` field to `serde_json::Value` and update helpers/hash accordingly. Remove `src/models/generic_genotype.rs`.
3.  **Obsolete Components**
    * Delete `Encodeable`, `Evaluator`, `TypeErasedEvaluator`, `Morphology`, their repositories, and related modules (`morphologies`).
4.  **Tests & Examples**
    * Rewrite existing tests to target the new `GenotypeManager` flow; prefer rewriting over removal when meaningful.
5.  **RNG & Logging**
    * Prefer passing a shared `&mut rng`; only create new RNGs when sharing is impractical across threads.
    * Follow existing instrumentation patterns (`#[instrument(level = \"debug\")]`; log only significant business events).

## Key Insights
*   **Breaking Change Simplifies**: Legacy pipeline is removed entirely; major-version bump will signal the break.
*   **Single Integration Point**: `GenotypeManager` replaces `Encodeable`/`Evaluator`/`Morphology` as the only integration surface.
*   **Type Erasure**: Operating on `serde_json::Value` keeps the service non-generic while enabling arbitrary genotypes (GA or GP).
*   **Framework Boundary**: Framework owns persistence/orchestration; consuming app owns structure and evolutionary logic in its `GenotypeManager`.
*   **Data Reset**: Clean break is acceptable; migration recreates `genotypes` with the new schema.
