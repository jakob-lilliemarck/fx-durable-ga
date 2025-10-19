# Refactor plan: Make optimization service pure orchestration

Goal
- Move business/decision logic into models under src/models
- Keep src/services/optimization/service.rs focused on orchestration: IO, transactions, locks, and event publishing

Guiding principles
- Models encapsulate domain logic; services orchestrate repositories and events
- No repository access from models
- Deterministic, testable model APIs (inject RNG or accept seeds where useful)
- Service methods become thin, readable, and composable

Current hotspots to extract from the service
1) Operator construction in new_optimization_request
   - Problem: hardcoded mutagen/crossover; ignores function inputsre
   - Fix: create operator constructors in models and let service pass inputs through

2) Breeding logic in breed_offspring_batch
   - Problem: crossover/mutation and progress placeholder (0.0) live in service
   - Fix: move to a model responsible for breeding children

3) Dedup + iterative selection loop in breed_genotypes
   - Problem: loop control, uniqueness policy, and selection retries live in service
   - Fix: extract as an evolver that produces unique offspring using domain policies

Proposed model additions/changes
1) Operators builder (new)
   - File: src/models/operators.rs (or fold into existing types if preferred)
   - Responsibility: encapsulate creation/validation of Mutagen and Crossover from parameters
   - API ideas:
     - Operators::new(mutation_rate: f64, temperature: f64, crossover_prob: f64) -> Result<Self, MutagenError | ProbabilityOutOfRangeError>
     - fn into_parts(self) -> (Mutagen, Crossover)
   - Service impact: new_optimization_request constructs Operators then passes parts to Request::new

2) Breeder (new)
   - File: src/models/breeder.rs
   - Responsibility: pure creation of child genotypes from two parents
   - API idea:
     - Breeder::new(request: &Request, morphology: &Morphology) -> Self
     - fn breed_child(&self, parent1: &Genotype, parent2: &Genotype, next_generation_id: i32, progress: f64, rng: &mut impl Rng) -> Genotype
     - fn breed_batch(&self, parent_pairs: &[(usize, usize)], candidates: &[(Genotype, Option<f64>)], next_generation_id: i32, progress: f64, rng: &mut impl Rng) -> Vec<Genotype>
   - Note: progress remains a parameter; service or a later Progress model decides its value

3) PopulationEvolver (new)
   - File: src/models/evolver.rs
   - Responsibility: iterate selection-breed-dedup until enough unique offspring
   - Inputs:
     - request, morphology, selector, target_offspring, next_generation_id
     - candidates_with_fitness: &[(Genotype, Option<f64>)]
     - uniqueness predicate: Fn(&i64) -> bool (provided by service using repo intersection results)
     - limits/policy (e.g., max_zero_progress)
   - Outputs:
     - EvolutionResult { offspring: Vec<Genotype>, duplicates_detected: usize, iterations: u32 }
   - API ideas:
     - PopulationEvolver::new(policy: EvolutionPolicy)
     - fn evolve(&self, breeder: &Breeder, selector: &Selector, candidates: &[(Genotype, Option<f64>)], unique: impl Fn(&i64) -> bool, target: usize, next_generation_id: i32, rng: &mut impl Rng) -> EvolutionResult
   - This centralizes selection retries and dedup accounting

4) Optional: ProgressCalculator (new, later)
   - File: src/models/progress.rs
   - Responsibility: compute optimization progress in [0,1] given request/population context
   - The service passes progress to Breeder; initial value can remain 0.0 until defined

Service refactor tasks (scoped steps)
1) new_optimization_request
   - Replace hardcoded Mutagen/Crossover with Operators::new(temperature, mutation_rate, crossover_prob)
   - Pass operators.into_parts() to Request::new
   - Keep transaction + event publishing as-is

2) generate_initial_population
   - Keep orchestration (fetch request/morphology, distribution, persist, events)
   - Optional: call Genotype::from_genomes(request, genomes, generation_id) if you choose to add a helper constructor

3) breed_offspring_batch → Breeder::breed_batch
   - Create Breeder in service using (&request, &morphology)
   - Replace inlined crossover/mutation with breeder.breed_batch(..., progress)
   - Remove local RNG duplication if breeder takes rng by &mut

4) breed_genotypes loop → PopulationEvolver::evolve
   - Service responsibilities:
     - lock key
     - early exit if generation exists
     - fetch candidates, morphology
     - compute uniqueness via repo.get_intersection and a local HashSet for already generated
     - persist results and publish events
   - Evolution responsibilities:
     - parent selection attempts, batching, retries, and zero-progress policy
     - return final unique offspring; track stats for logs
   - Replace in-service constant MAX_ZERO_PROGRESS with EvolutionPolicy in models

5) maintain_population
   - Keep orchestration: fetch request and population, delegate to Request/Schedule for decisions
   - Call breed_genotypes (which now uses models) as before

6) Terminator
   - Keep as infra glue (service/repo check). No model dependency

Testing plan
- Unit tests for Operators, Breeder, PopulationEvolver (pure logic, deterministic with seeded RNG)
- Service tests become slimmer: verify orchestration, transactions, and event publishing with mocked repos/bus
- Existing model tests remain; add new tests for new model modules

Migration plan
1) Introduce new model modules with full tests
2) Refactor service to use models without changing public API
3) Remove duplicated logic and constants from the service
4) Run cargo check and targeted cargo test; update coverage baselines

Open questions
- Where should progress be computed? For now pass 0.0 (existing behavior) via parameter; later introduce ProgressCalculator
- Place Operators next to Mutagen/Crossover or standalone module? Start standalone for cohesion
- Do we want a model-level uniqueness strategy abstraction (hash vs. genome) for future evolvers?
