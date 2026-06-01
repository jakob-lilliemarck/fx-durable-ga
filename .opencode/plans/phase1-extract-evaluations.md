# Phase 1: Move Evaluation Write Path to Evaluation Service

## Goal

The evaluation service owns all evaluation writes. No changes to the database schema.
All existing JOINs continue to work. The genotypes repository retains temporary write
access via `genotypes::WriteTx::store_evaluations` for the optimization service's breed
cycle (resolved in Phase 3).

## Steps

### 1.1 Create evaluation repository directory structure

```
src/services/evaluation/repositories/
├── mod.rs
├── errors.rs
└── queries/
    ├── mod.rs
    └── store_evaluations.rs
```

### 1.2 Create files

#### `repositories/errors.rs`

```rust
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Not found: {0}")]
    NotFound(Uuid),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
```

#### `repositories/queries/store_evaluations.rs`

Copy from `src/repositories/genotypes/queries/store_evaluations.rs`, changing:
- `use super::super::Error as RepositoryError` → `use super::super::Error as RepositoryError`
- Import path for `Evaluation` → point to `super::super::super::Evaluation`
- Keep `pub(crate)` visibility
- Tests need updating — import path changes. For now, mark tests with `#[cfg(test)]` but the test may need adjustments. Since the Evaluation model is being moved, the tests can import from the evaluation crate.

#### `repositories/queries/mod.rs`

```rust
mod store_evaluations;

pub(crate) use store_evaluations::store_evaluations;
```

#### `repositories/mod.rs`

```rust
mod errors;
pub(crate) mod queries;

use crate::infrastructure::db;
use crate::services::evaluation::Evaluation;
use sqlx::PgTransaction;
use tracing::instrument;

#[derive(Debug, Clone)]
pub struct Read {
    ro: db::ReadPool,
}

#[derive(Debug, Clone)]
pub struct Write {
    wr: db::WritePool,
}

pub struct WriteTx<'tx> {
    tx: &'tx mut PgTransaction<'static>,
}

impl db::Tx for Write {
    type Error = super::errors::Error;

    fn tx(self) -> db::TxFut<Self::Error> {
        let pool = self.wr.pool.clone();
        Box::pin(async move {
            let tx = pool.begin().await?;
            Ok(tx)
        })
    }
}

impl Read {
    pub fn new(ro: db::ReadPool) -> Self {
        Self { ro }
    }
}

impl Write {
    pub fn new(wr: db::WritePool) -> Self {
        Self { wr }
    }
}

impl<'tx> WriteTx<'tx> {
    pub fn new(tx: &'tx mut PgTransaction<'static>) -> Self {
        Self { tx }
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn store_evaluations(
        &mut self,
        evaluations: &[Evaluation],
    ) -> Result<Vec<Evaluation>, errors::Error> {
        queries::store_evaluations(&mut **self.tx, evaluations).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migrations;
    use sqlx::PgPool;

    #[sqlx::test(migrations = false)]
    async fn it_stores_evaluations(pool: PgPool) -> anyhow::Result<()> {
        migrations::run_default_migrations(&pool).await?;
        // TODO: Add proper seed + assertion
        Ok(())
    }
}
```

### 1.3 Move `Evaluation` model to evaluation service

In `src/services/evaluation/mod.rs`, add the `Evaluation` struct (moved from
`src/repositories/genotypes/models.rs`). The model should be re-exported from
the evaluation module so the genotypes repository can still access it during the
transitional phase (Phase 3 removes this dependency).

Add to `src/services/evaluation/mod.rs`:

```rust
mod errors;
mod events;
mod registrations;
mod repositories;
mod service;

pub use errors::Error;
pub use events::GenotypeEvaluatedEvent;
pub use registrations::register;
pub use service::Evaluator;
pub use service::Service;
pub use service::Evaluation;  // <--- NEW
```

And in `service.rs`, add the `Evaluation` struct, or create a separate `models.rs`.
Simplest: add it to `service.rs` as `pub use` from somewhere, or keep `Evaluation`
in `mod.rs`.

Actually, better: create `src/services/evaluation/models.rs`:

```rust
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Evaluation {
    pub(crate) id: Uuid,
    pub(crate) genotype_id: Uuid,
    pub(crate) fitness: f64,
    pub(crate) started_at: Option<DateTime<Utc>>,
    pub(crate) completed_at: Option<DateTime<Utc>>,
    pub(crate) evaluated_by: Option<Uuid>,
}

impl Evaluation {
    pub fn new(
        genotype_id: Uuid,
        fitness: f64,
        started_at: Option<DateTime<Utc>>,
        completed_at: Option<DateTime<Utc>>,
        evaluated_by: Option<Uuid>,
    ) -> Self { /* same as current implementation */ }

    // accessors
    pub fn id(&self) -> Uuid { self.id }
    pub fn genotype_id(&self) -> &Uuid { &self.genotype_id }
    pub fn fitness(&self) -> f64 { self.fitness }
    pub fn started_at(&self) -> &Option<DateTime<Utc>> { &self.started_at }
    pub fn completed_at(&self) -> &Option<DateTime<Utc>> { &self.completed_at }
    pub fn evaluated_by(&self) -> &Option<Uuid> { &self.evaluated_by }
}
```

Update `mod.rs` to declare and re-export `Evaluation`:

```rust
mod models;
pub use models::Evaluation;
```

### 1.4 Update `evaluation::Service` to use its own WriteTx

In `src/services/evaluation/service.rs`:

- Change `genotypes_wr: genotypes::Write` to `evaluations_wr: repositories::Write`
- Update `evaluate_genotype` to use `repositories::WriteTx::store_evaluations` instead of `genotypes::WriteTx::store_evaluations`
- The `genotypes_ro: genotypes::Read` stays (needed for genotype data like `type_name`)

Before:
```rust
db::begin(self.genotypes_wr.clone(), |tx| {
    Box::pin(async move {
        genotypes::WriteTx::new(tx)
            .store_evaluations(&[Evaluation::new(...)])
            .await?;
        // ... publish event
    })
}).await?;
```

After:
```rust
db::begin(self.evaluations_wr.clone(), |tx| {
    Box::pin(async move {
        super::repositories::WriteTx::new(tx)
            .store_evaluations(&[Evaluation::new(...)])
            .await?;
        // ... publish event (same tx)
    })
}).await?;
```

### 1.5 Update evaluation DI registrations

In `src/services/evaluation/registrations.rs`:

- Remove injection of `genotypes::Write`
- Add injection of `super::repositories::Write`

Before:
```rust
let genotypes_ro = c.get::<genotypes::Read>().await?;
let genotypes_wr = c.get::<genotypes::Write>().await?;
let service = super::Service::new(host_id, synchronization, genotypes_ro, genotypes_wr);
```

After:
```rust
let genotypes_ro = c.get::<genotypes::Read>().await?;
let evaluations_wr = c.get::<super::repositories::Write>().await?;
let service = super::Service::new(host_id, synchronization, genotypes_ro, evaluations_wr);
```

Also register the evaluation repositories:
```rust
pub fn register(c: &mut Container) {
    c.provide(provide_evaluation_service);
    c.provide(provide_evaluations_repository_ro);
    c.provide(provide_evaluations_repository_wr);
}

fn provide_evaluations_repository_ro(c: &mut Container) -> BoxFuture<'_, ProviderResult<super::repositories::Read>> {
    Box::pin(async {
        let ro = c.get::<db::ReadPool>().await?;
        Ok(super::repositories::Read::new(ro))
    })
}

fn provide_evaluations_repository_wr(c: &mut Container) -> BoxFuture<'_, ProviderResult<super::repositories::Write>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;
        Ok(super::repositories::Write::new(wr))
    })
}
```

### 1.6 Keep optimization breed cycle using genotypes WriteTx (temporary)

The `optimization::Service` breed cycle at `src/services/optimization/service.rs:500–516`
continues to use `genotypes::WriteTx::store_evaluations` because it stores genotypes
AND evaluations atomically in the same transaction alongside job publishing.
This is acceptable for Phase 1 and will be resolved in Phase 3.

### 1.7 Forward-compatibility: add `request_id` and `generated_at` to Evaluation

In preparation for Phase 2, add `request_id: Option<Uuid>` and
`generated_at: Option<DateTime<Utc>>` fields to the `Evaluation` struct
as optional fields (default to `None`). This way the evaluation service
(and optimization service) can begin populating them at write time without
requiring the schema migration yet.

These fields will be used in Phase 2 when the evaluations table moves to
its own schema and needs to filter by `request_id` independently.

### 1.8 Remove `Evaluation` model from genotypes repository

After the `Evaluation` struct has been moved to the evaluation service and
re-exported, update:
- `src/repositories/genotypes/mod.rs` — change `pub use models::{..., Evaluation, ...}` to import from the evaluation service: `pub use crate::services::evaluation::Evaluation`
- All genotypes files that reference `Evaluation` should now get it via the re-export

This avoids code duplication and maintains a single source of truth for the model.

### 1.9 Run verification

```bash
cargo check
cargo test -p fx-durable-ga
```

Expected: all tests pass, no compiler errors or warnings.
The key behaviors to verify:
- `evaluation::Service::evaluate_genotype` writes evaluations via its own repo
- `optimization::Service` breed cycle continues to write evaluations atomically with genotypes
- All existing read queries continue to work unchanged
