# Service Refactoring Status: Move Domain Logic to Models

## Goal
Keep `src/services/optimization/service.rs` focused on orchestration (IO, transactions, locks, events) while moving business logic into `src/models`.

## Guiding Principles
- Models encapsulate domain logic; services orchestrate repositories and events
- No repository access from models
- Deterministic, testable model APIs with injected RNG
- Service methods become thin, readable, and composable

## Current Status

### ✅ COMPLETED

**1. Breeder** - `src/models/breeder.rs`
- **Status**: ✅ Complete with 100% test coverage
- **Implementation**: Pure static functions (cleaner than originally planned struct)
- **API**: `Breeder::breed_batch(request, morphology, parent_pairs, generation_id, progress, rng)`
- **Service integration**: ✅ Replaced inline breeding logic in `breed_genotypes`
- **Tests**: 6 comprehensive tests covering all scenarios including genetic diversity verification

**2. Clean Parameter Passing**
- **Status**: ✅ Already implemented correctly
- **Implementation**: `new_optimization_request` takes `mutagen` and `crossover` as parameters
- **No action needed**: Current design is cleaner than originally planned Operators builder

### 🚧 REMAINING WORK

**PopulationEvolver** - `src/models/evolver.rs` (Not started)
- **Problem**: Complex deduplication + iterative selection loop still lives in `breed_genotypes`
- **Current location**: Lines ~301-358 in `breed_genotypes` method
- **Logic to extract**:
  - While loop with `final_genotypes.len() < num_offspring`
  - `MAX_ZERO_PROGRESS` retry policy
  - Batch creation, hash collection, database intersection checks
  - Duplicate filtering with `generated_hashes` HashSet
  - Zero progress counter and iteration management

**Proposed PopulationEvolver API**:
```rust
struct EvolutionResult {
    offspring: Vec<Genotype>,
    duplicates_detected: usize,
    iterations: u32,
}

struct EvolutionPolicy {
    max_zero_progress_iterations: i32,
}

impl PopulationEvolver {
    fn evolve(
        policy: &EvolutionPolicy,
        request: &Request,
        morphology: &Morphology, 
        candidates: &[(Genotype, Option<f64>)],
        target_offspring: usize,
        next_generation_id: i32,
        is_unique: impl Fn(i64) -> bool,  // provided by service using repo + local HashSet
        rng: &mut impl Rng,
    ) -> Result<EvolutionResult, SelectionError>
}
```

**Service responsibilities after refactor**:
- Lock management and early exit checks
- Fetch candidates and morphology from repositories
- Compute uniqueness via `repo.get_intersection` + local `HashSet`
- Persist final results and publish events
- Log warnings about insufficient unique genotypes

## Migration Plan
1. ✅ ~~Create Breeder with full tests~~
2. ✅ ~~Integrate Breeder into service~~
3. **Next**: Create PopulationEvolver with comprehensive tests
4. **Final**: Integrate PopulationEvolver into `breed_genotypes`, removing complex loop

## Testing Strategy
- ✅ Breeder: 100% coverage with deterministic seeded RNG tests
- **Next**: PopulationEvolver unit tests with mocked uniqueness predicates
- Service tests focus on orchestration, not domain logic

## Progress: ~75% Complete
- ✅ Domain logic extraction: 1 of 2 components complete
- ✅ Service integration: 1 of 2 major methods refactored
- **Remaining**: Extract the most complex piece (PopulationEvolver) to complete the refactoring
