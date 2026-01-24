# Refactor requests to single user_defined JSON field
## Problem
Crossover, mutagen, and distribution are stored and serialized as separate typed fields on Request/DbRequest and the requests table. With GenotypeManager now owning GA operators, these fixed schemas are unnecessary and cause JSON shape mismatches. We want a single opaque JSONB field (instructions) carried through the API/DB and passed to GenotypeManager.
### Current state
* requests table columns: crossover JSONB, mutagen JSONB, distribution JSONB
* Request struct fields: mutagen: Mutagen, crossover: Crossover, distribution: Distribution
* DbRequest mirrors these; queries insert/select them
* Tests/examples construct requests with these typed fields
* No production data to preserve; schema changes can drop columns
## Proposed changes
### Migrations
SQL (new migration):
```sql path=null start=null
-- drop GA-specific columns
ALTER TABLE fx_durable_ga.requests
  DROP COLUMN IF EXISTS crossover,
  DROP COLUMN IF EXISTS mutagen,
  DROP COLUMN IF EXISTS distribution;
-- add opaque user-defined blob
-- add opaque instructions blob
ALTER TABLE fx_durable_ga.requests
  ADD COLUMN user_defined JSONB NOT NULL;
```
No default; column is required.
### Models
`src/models/request.rs`:
```rust path=./src/models/request.rs start=null
pub struct Request {
    pub(crate) id: Uuid,
    pub(crate) requested_at: DateTime<Utc>,
    pub(crate) type_name: String,
    pub(crate) type_hash: i32,
    pub(crate) goal: FitnessGoal,
    pub(crate) selector: Selector,
    pub(crate) schedule: Schedule,
    pub(crate) user_defined: serde_json::Value, // new, replaces mutagen/crossover/distribution
    pub data: Option<serde_json::Value>,
}

pub(crate) fn new(
    type_name: &str,
    type_hash: i32,
    goal: FitnessGoal,
    selector: Selector,
    schedule: Schedule,
    user_defined: impl Serialize,          // replaces mutagen/crossover/distribution params
    data: Option<impl Serialize>,
) -> Result<Self, RequestValidationError> {
    let user_defined = serde_json::to_value(user_defined)?;
    let data = data.map(serde_json::to_value).transpose()?;
    Ok(Self { /* ... */ user_defined, data })
}
```
`DbRequest` mirrors this shape (user_defined field instead of three GA fields). Remove the mutagen/crossover/distribution fields entirely.

Mutagen model: delete `src/models/mutagen.rs` (and Temperature/MutationRate helpers) after refactor; remove exports/usages.
```rust path=./src/services/optimization/service.rs start=null
pub async fn new_optimization_request(
    &self,
    type_name: &str,
    type_hash: i32,
    goal: FitnessGoal,
    schedule: Schedule,
    selector: Selector,
    user_defined: impl Serialize,          // new parameter
) -> Result<Uuid, Error> {
    let request = Request::new(
        type_name,
        type_hash,
        goal,
        selector,
        schedule,
        user_defined,
        data, // unchanged
    )?;
    // existing flow continues
}
```
Breeder/GenotypeManager wiring (signature changes):
```rust path=./src/models/evolution.rs start=null
pub trait GenotypeManager {
    fn random(&self, rng: &mut dyn RngCore, user_defined: &Value) -> Result<Value>;
    fn crossover(&self, p1: &Value, p2: &Value, rng: &mut dyn RngCore, user_defined: &Value) -> Result<Value>;
    fn mutate(&self, genome: &mut Value, rng: &mut dyn RngCore, progress: f64, user_defined: &Value) -> Result<()>;
    fn evaluate<'a>(&'a self, genome: &'a Value, terminated: &'a dyn Terminated, user_defined: &'a Value) -> BoxFuture<'a, Result<f64>>;
}
```
`Breeder` uses:
```rust path=./src/models/breeder.rs start=null
let mut child_genome = manager.crossover(&p1, &p2, rng, &request.user_defined)?;
manager.mutate(&mut child_genome, rng, progress, &request.user_defined)?;
```
Mutation rate/temperature are derived inside the manager from `user_defined`; the Mutagen type is removed from Request.
### Queries
`src/repositories/requests/queries.rs` insert/select bindings change to a single JSONB column:
```rust path=./src/repositories/requests/queries.rs start=null
INSERT INTO fx_durable_ga.requests (
    id, requested_at, type_name, type_hash,
    goal, schedule, selector,
    user_defined,  -- new
    data
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
RETURNING id, requested_at, type_name, type_hash,
          goal, schedule, selector,
          user_defined, data;
```
DbRequest mapping updated accordingly.
### Tests
- `repositories::requests::models` tests: build requests with `user_defined = json!({...})`; assert round-trip equals that JSON.
- `requests::queries` SQLx tests: update insert/select expectations to include `user_defined`; drop assertions on crossover/mutagen/distribution.
- `breeder` tests: pass `user_defined` JSON; adjust GenotypeManager test stub to accept the new signature.
- Remove/adjust tests that assert `{"Uniform":{...}}` shapes.
### Examples
- `examples/point_search.rs`: when creating requests, supply one `user_defined` JSON that holds whatever knobs the PointManager needs (e.g., mutation/crossover probabilities, bounds).
- `examples/regression_model.rs` and `feature_engineering.rs`: same pattern—embed their configs in `user_defined`; managers deserialize as needed.
- GenotypeManager implementations deserialize `user_defined` in `random/crossover/mutate/evaluate`.
### Cleanup
- Remove `src/models/crossover.rs`, mutagen/distribution types, and exports from `models/mod.rs` if no longer referenced after signature changes.
- Run `cargo fmt` and full test suite.
## Risks / open questions
* Any remaining code paths expecting Mutagen/Distribution types (e.g., legacy tests/examples) must be migrated or deleted
* If we still need temperature/mutation rate convenience helpers, consider providing helper builders that return JSON snippets rather than typed structs
* Coordination with SQLx offline cache: regenerate after migration
## Rollout plan
1) Implement migration (drop three columns, add instructions)
2) Update Request/DbRequest structs and conversions
3) Update Service/new_optimization_request and Breeder/GenotypeManager call sites
4) Update queries and tests
5) Update examples
6) cargo fmt && SQLX_OFFLINE=true cargo test
7) Regenerate sqlx-data.json if used
