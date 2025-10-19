# Service Test Plan

## Test Strategy

- Use real repositories with sqlx testing tooling (no mocks)
- Focus on service orchestration logic and integration
- Test database interactions and event publishing
- Keep test data simple and minimal
- Use the same testing patterns as repository tests

## Test Cases (Priority Order)

### Priority 1: Core Happy Path Tests

#### 1.1 `new_optimization_request` - Happy Path
**Why**: Core entry point, must work for everything else to function
- Create a valid optimization request
- Verify request is stored in database
- Verify `OptimizationRequestedEvent` is published

#### 1.2 `generate_initial_population` - Happy Path  
**Why**: Critical for starting optimization, relatively isolated
- Create a request with known morphology
- Generate initial population
- Verify genotypes are created in database
- Verify `GenotypeGenerated` events are published for each genotype

#### 1.3 `evaluate_genotype` - Happy Path
**Why**: Core evaluation loop, needs to work for optimization to proceed  
- Create genotype and register simple evaluator
- Evaluate genotype
- Verify fitness is recorded in database
- Verify `GenotypeEvaluatedEvent` is published

### Priority 2: Core Error Handling

#### 2.1 `evaluate_genotype` - Unknown Type Error
**Why**: Common error condition, should fail gracefully
- Try to evaluate genotype with unregistered type hash
- Verify returns `UnknownTypeError`

#### 2.2 `evaluate_genotype` - Genotype Not Found
**Why**: Database consistency error, should be handled
- Try to evaluate non-existent genotype ID
- Verify error is handled gracefully

#### 2.3 `generate_initial_population` - Request Not Found
**Why**: Job system could call with invalid request ID
- Try to generate population for non-existent request
- Verify error is handled gracefully

### Priority 3: Population Management

#### 3.1 `maintain_population` - Request Completion
**Why**: Optimization needs to terminate when goal is reached
- Create request with low fitness goal
- Create population with genotype that exceeds goal
- Call maintain_population
- Verify `RequestCompletedEvent` is published

#### 3.2 `maintain_population` - Schedule Decisions
**Why**: Core optimization flow control
- Test Wait decision: verify no action taken
- Test Terminate decision: verify termination event published  
- Test Breed decision: verify breeding is triggered (stub breed_genotypes)

#### 3.3 `maintain_population` - No Fitness Yet
**Why**: Edge case that occurs early in optimization
- Create request with genotypes but no fitness recorded
- Call maintain_population
- Verify early return (no events published)

### Priority 4: Advanced Features

#### 4.1 `conclude_request` - Happy Path
**Why**: Request lifecycle completion
- Create request conclusion
- Call conclude_request
- Verify conclusion is stored in database

#### 4.2 `conclude_request` - Idempotent
**Why**: Multiple workers might try to conclude same request
- Conclude same request twice
- Verify second call doesn't duplicate or error

#### 4.3 `publish_terminated` - Happy Path
**Why**: Termination event publishing
- Call publish_terminated with request ID
- Verify `RequestTerminatedEvent` is published

### Priority 5: Race Conditions & Edge Cases

#### 5.1 `generate_initial_population` - Race Condition
**Why**: Multiple workers might try to generate same initial population
- Simulate race condition (empty insert result)
- Verify no events are published but no error occurs

#### 5.2 `breed_genotypes` - Generation Already Exists
**Why**: Multiple workers might try to breed same generation  
- Create generation manually in database
- Call breed_genotypes for same generation
- Verify early return without breeding

#### 5.3 `breed_genotypes` - Deduplication Logic
**Why**: Complex deduplication logic needs verification
- Set up scenario that generates duplicate genotypes
- Verify deduplication attempts counter works
- Verify warning is logged when max attempts reached

## Test Implementation Notes

### Database Setup
- Use `#[sqlx::test]` attribute
- Each test gets fresh database
- Use repository test patterns for setup

### Event Testing
- Use event bus testing utilities
- Verify event types and payload content
- Check event ordering when multiple events published

### Test Data
- Create minimal test morphology (e.g., 2 genes, 10 steps each)
- Use simple evaluator that returns constant fitness
- Use human-readable UUIDs in tests: `00000000-0000-0000-0000-000000000001`

### Error Testing
- Don't just test that errors occur, verify specific error types
- Test error propagation through async boundaries

## Implementation Order

Start with Priority 1 tests to establish testing patterns and infrastructure. Each priority level builds on the previous ones, with later tests being able to reuse setup code and testing utilities developed for earlier tests.

The Priority 1 tests will validate the core optimization flow works end-to-end, while higher priority tests focus on edge cases and error conditions.