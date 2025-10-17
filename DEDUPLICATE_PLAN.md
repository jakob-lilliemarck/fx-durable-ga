# Genome Deduplication Implementation Plan

## Objective
Eliminate duplicate genome evaluations to achieve 35% reduction in computational cost (based on analysis showing 1.5x average duplication rate).

## Current Metrics (Baseline)
- **Average duplicates per genome**: 1.5x
- **Average evaluations to goal**: 688
- **Target reduction**: 35% (saving ~241 evaluations per optimization)

## Approach Overview

### Core Strategy
1. **Single database query** to fetch all existing genome hashes for the request
2. **In-memory duplicate detection** using HashSet for O(1) lookups
3. **Iterative breeding with zero-progress detection** to avoid infinite loops
4. **Graceful degradation** (accept smaller populations rather than bias)

### Workflow
```
existing_hashes = query_database_once(request_id)  // Single DB call
generated_hashes = HashSet::new()                  // In-memory cache
zero_progress_counter = 0
unique_offspring = []

while (need_more_offspring && zero_progress_counter < 5) {
    batch = breed_offspring_from_same_parents()
    new_unique = filter_duplicates(batch, existing_hashes, generated_hashes)
    
    if (new_unique.count() == 0) {
        zero_progress_counter++
    } else {
        zero_progress_counter = 0
        unique_offspring.extend(new_unique)
        generated_hashes.extend(hashes_of_new_unique)
    }
}

save_to_database(unique_offspring)  // Existing logic
```

## Required Changes

### 1. Database Schema Changes
- **Add `genome_hash` column** to `fx_durable_ga.genotypes` table (BIGINT)
- **Add index** on `(request_id, genome_hash)` for fast lookups
- **Migration** to compute hashes for existing genomes

### 2. Code Restructuring

#### A. Extract Pure Breeding Logic
**File**: `src/service/service.rs`

Extract from `breed_new_genotypes()` (lines 270-319):
```rust
fn breed_offspring_batch(
    request: &Request,
    parent_genotype_pairs: &[(Genotype, Genotype)],
    morphology: &Morphology,
    num_offspring: usize,
    next_generation_id: i32,
) -> Vec<Genotype> {
    // Pure breeding logic - no database operations
    // Returns Vec<Genotype> ready for deduplication
}
```

#### B. Add Genome Hashing
**File**: `src/models/genotype.rs`

```rust
impl Genotype {
    pub fn compute_hash(&self) -> i64 {
        // Fast hash of genome Vec<i64>
        // Consider using FxHash or SipHash for speed
    }
}
```

#### C. Create Deduplication Wrapper
**File**: `src/service/service.rs`

```rust
async fn breed_new_genotypes_with_deduplication(
    &self,
    request: &Request,
    num_offspring: usize,
    next_generation_id: i32,
) -> Result<(), Error> {
    // 1. Query existing hashes ONCE
    // 2. Perform parent selection ONCE
    // 3. Iterative breeding with duplicate filtering
    // 4. Save unique offspring (existing database logic)
}
```

#### D. Add Hash Query Method
**File**: `src/repositories/genotypes/queries.rs`

```rust
pub async fn get_existing_genome_hashes<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: Uuid,
) -> Result<HashSet<i64>, Error> {
    // SELECT genome_hash FROM genotypes WHERE request_id = $1
}
```

### 3. Integration Points

#### Replace Current Call
**File**: `src/service/service.rs` in `maintain_population()`

Change:
```rust
self.breed_new_genotypes(&request, num_offspring, next_generation_id).await
```

To:
```rust
self.breed_new_genotypes_with_deduplication(&request, num_offspring, next_generation_id).await
```

## Implementation Steps

### Phase 1: Database Changes
1. Create migration for `genome_hash` column
2. Add index on `(request_id, genome_hash)`
3. Populate existing genomes with computed hashes

### Phase 2: Core Logic
1. Implement `Genotype::compute_hash()`
2. Extract `breed_offspring_batch()` from existing code
3. Add `get_existing_genome_hashes()` query

### Phase 3: Deduplication Wrapper
1. Implement `breed_new_genotypes_with_deduplication()`
2. Add duplicate filtering logic
3. Integrate zero-progress detection

### Phase 4: Integration & Testing
1. Replace calls to use new deduplication method
2. Test with existing optimization runs
3. Measure performance improvement

## Success Criteria

### Performance Metrics
- **Duplicate rate reduction**: From 1.5x to ~1.05x
- **Evaluation reduction**: From 688 to ~450 average evaluations to goal
- **Overall savings**: 35% fewer fitness evaluations

### Quality Assurance
- **No bias introduced**: Same selection process, just duplicate filtering
- **Deterministic results**: Same parents produce same offspring distribution
- **Graceful handling**: No infinite loops, accept smaller populations when needed

## Edge Cases Handled

1. **Hash collisions**: Extremely rare with good hash function, can be ignored
2. **Zero progress**: Counter prevents infinite loops
3. **Small search spaces**: Graceful degradation with smaller populations
4. **Empty parent pools**: Existing error handling remains

## Risks & Mitigations

### Risk: Performance regression from additional processing
**Mitigation**: In-memory hash operations are extremely fast (O(1)), offset by massive evaluation savings

### Risk: Infinite loops in degenerate cases  
**Mitigation**: Zero-progress counter with hard limit (5 attempts)

### Risk: Breaking existing functionality
**Mitigation**: Incremental implementation with existing tests, rollback capability

## Future Enhancements

1. **Cross-request deduplication**: Share genomes across different optimization requests
2. **Approximate deduplication**: Consider Hamming distance for very similar genomes
3. **Hash function optimization**: Profile different hash algorithms for performance
4. **Persistent cache**: Store hashes in Redis for multi-process scenarios

## Expected Impact

With current duplication rate of 1.5x:
- **Before**: 688 evaluations average to reach goal
- **After**: ~450 evaluations average to reach goal  
- **Savings**: 238 evaluations (35% reduction)
- **Multiplicative benefit**: Savings compound with population size and generation count