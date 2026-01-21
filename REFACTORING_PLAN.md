# Refactoring Plan: Transition to Generic Evolutionary Framework

**Status:** Phase 1 & 2 Complete

## Overview
Transform `fx-durable-ga` from a specialized numeric optimization engine into a generic evolutionary framework capable of handling any user-defined genotype (including GP trees, hybrid structures, and neural network configurations).

This will be achieved by building a **parallel, isolated persistence layer** first, allowing the new generic system to be developed and tested without interfering with the existing `BIGINT[]`-based GA.

## Core Philosophy
Shift from **Data-Driven Definition** (Morphology rules in DB) to **Code-Driven Definition** (Rust's type system defines rules at compile time). The framework becomes ignorant of the concrete types it manages.

## Architectural Changes

### 1. The `Evolvable` Trait (Complete)
The core contract for any type that can be optimized by the framework. It is constrained to ensure types are persistable, clonable, and thread-safe.

```rust
pub trait Evolvable: Serialize + DeserializeOwned + Clone + Send + Sync + Debug {
    // ... methods: mutate, crossover
}
```

### 2. Type-Erased Persistence (Complete)
Instead of making the repository generic, we use type erasure at the storage layer.

*   **`GenericGenotype` Struct**: A single, non-generic struct stores the genome as a `serde_json::Value`. This is the model that the repository works with.
*   **Database**: A new, separate table `generic_genotypes` with a `genome_data JSONB` column has been created.
*   **Repository**: A new, non-generic `generic_genotypes` repository handles the `GenericGenotype` struct, serializing to and from the database.

### 3. Service Layer Generics
*   **The service layer will be the generic component.** A new service, e.g., `GenericOptimizationService<G: Evolvable>`, will be created.
*   This service will orchestrate the evolutionary loop:
    1. Load parents as `GenericGenotype` from the repository.
    2. Deserialize the `genome_data` `Value` into the concrete type `G`.
    3. Call `crossover` and `mutate` on the concrete `G` instances.
    4. Serialize the new child `G` back into a `serde_json::Value`.
    5. Store the new child in a `GenericGenotype` struct via the repository.

### 4. Deprecations (Long-Term)
*   **Morphology**: This concept is superseded by the `Evolvable` implementation on the user's type.
*   **Mutagen/Crossover Models**: These remain as configuration DTOs (holding rates/params) but their logic is no longer used by the generic services.

## Implementation Steps & Progress

### Phase 1: Foundation (Complete)
- [x] **Define Trait**: `src/models/evolution.rs` created with the `Evolvable` trait.
- [x] **Type-Erased Model**: `src/models/generic_genotype.rs` created with `genome_data: serde_json::Value`.
- [x] **Migration**: A new migration has been created and applied for the `generic_genotypes` table.

### Phase 2: Persistence (Complete)
- [x] **Parallel Repository**: A new, separate `generic_genotypes` repository has been implemented.
- [x] **Validation**: A roundtrip test has been written and is passing, proving that a custom `Evolvable` type can be successfully written to and read from the new table.

### Phase 3: Service Refactoring (Next Steps)
1.  **Create a new, generic optimization service** (`GenericOptimizationService<G>`) that is parallel to the existing service.
2.  Implement the main evolutionary loop within this new service, orchestrating the `generic_genotypes` repository and the `Evolvable` trait methods.
3.  Create an example or integration test that uses this service to run a full, end-to-end optimization for a sample `Evolvable` type.

### Phase 4: Integration & Cleanup (Future)
1.  **Legacy Support**: Implement `Evolvable` for a wrapper around `Vec<i64>` to provide a migration path for existing use cases.
2.  **Unify Services**: Once the generic service is proven, decide whether to migrate the old service or maintain both.
3.  **Cleanup**: Remove unused `Morphology` logic and old models if a full migration occurs.

## Key Insights
*   **Type Erasure is Key**: The repository and database should be unaware of the concrete types. Storing genomes as `JSONB` and making the *service* layer generic is the correct architecture.
*   **Parallel Development**: Building a separate `generic_genotypes` table and repository is a low-risk strategy that prevents disruption to the existing system.
*   **Stateful Nodes**: GP nodes can hold state, but it must be ephemeral. `#[serde(skip)]` and a custom `Clone` implementation that resets state is the correct pattern.
*   **Framework Boundary**: The framework handles "Persistence" and "Orchestration". The consuming application handles "Structure" and "Evolution Logic".
