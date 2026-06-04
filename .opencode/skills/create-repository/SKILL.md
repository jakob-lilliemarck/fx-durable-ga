---
name: create-repository
description: |
  Use when creating a new repository aggregate inside an existing service's
  repositories/ directory. Scaffolds the complete repository file structure
  including Read/Write/WriteTx struct, models, DI registrations, and an empty
  queries/ directory with a stub mod.rs. Do NOT use for services, controllers,
  or infrastructure code.
---

# Create Repository

## Step 1 — Scaffold boilerplate

Run the `scaffold-repository` tool:

```
serviceName: evaluation       # directory under src/services/
aggregateName: evaluations    # plural, underscore-separated directory name
singularName: Evaluation      # PascalCase domain model struct name
updateModFiles: true          # (default) updates parent module files
```

This creates:

```
src/services/<service>/repositories/<aggregate>/
├── mod.rs
├── errors.rs
├── repository.rs
├── registrations.rs
├── models.rs
└── queries/
    └── mod.rs              # stub — add concrete query files per repository concern
```

If parent modules already exist, the tool appends to `repositories/mod.rs` and service `mod.rs`. If not, it creates them.

## Step 2 — Customize the generated files

### models.rs

Add real fields to the `{Singular}` struct:
- Fields are `pub(crate)`
- Accessor methods are `pub`
- Use `Uuid::now_v7()` in constructors

### queries/ — add concrete query files

Each query is its own file in `queries/`. For example, `queries/search.rs`:

```rust
use super::super::Error as RepositoryError;
use super::super::{Singular};
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Default)]
pub struct SearchFilter {
    ids: Option<Vec<Uuid>>,
    limit: Option<i64>,
}

impl SearchFilter {
    pub fn with_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.ids = Some(ids);
        self
    }

    pub fn with_limit(mut self, limit: i64) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Performs a filtered search.
#[instrument(level = "debug", skip(tx))]
pub async fn search<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchFilter,
) -> Result<Vec<Singular>, RepositoryError> {
    let rows = sqlx::query_as::<_, Singular>(
        r#"
        SELECT id, ...
        FROM schema.table
        WHERE ($1::uuid[] IS NULL OR id = ANY($1))
        LIMIT $2
        "#,
    )
    .bind(filter.ids.as_deref())
    .bind(filter.limit)
    .fetch_all(tx)
    .await?;

    Ok(rows)
}
```

Don't forget to declare each query file in `queries/mod.rs` and re-export the filter.

### repository.rs

Add methods that delegate to the query functions. For example:

```rust
impl Read {
    #[instrument(level = "debug", skip(self))]
    pub async fn search(&self, filter: &SearchFilter) -> Result<Vec<Singular>, Error> {
        super::queries::search(&self.ro.pool, filter).await
    }
}
```

Also add `pub(crate)` mutation methods on `WriteTx` for insert/update/delete as needed.

### Service's `registrations.rs`

The tool does NOT update the service's `registrations.rs`. You must manually add:

```rust
c.provide(
    <aggregate>::registrations::provide_<aggregate>_repository_ro,
);
c.provide(
    <aggregate>::registrations::provide_<aggregate>_repository_wr,
);
```

## Naming conventions (reference)

| Aspect | Convention |
|---|---|
| Directory | lowercase, plural, underscore-separated |
| Error enum | `pub enum Error` |
| Repository structs | `Read`, `Write`, `WriteTx` |
| Provider fns | `provide_<name>_repository_ro` / `_wr` |
| `Read` visibility | `pub` — injectable into any caller |
| `Write`/`WriteTx` | `pub(crate)` — never re-exported past the service |
| `queries` module | `pub(super)` — tests access queries through repo methods, not directly |
| Domain model fields | `pub(crate)` with `pub` accessor methods |

## Query rules (reference)

- Use `$1 IS NULL OR col = $1` pattern for optional filter fields
- Filter types use builder pattern: `with_*` methods returning `Self`
- Always require `LIMIT`; use cursor pagination, never `OFFSET`
- Use `RETURNING` on all INSERT/UPDATE queries
- Pass non-`Copy` args as references; the function body clones if needed
- Inline tests: `#[sqlx::test(migrations = false)]` + `crate::migrations::run_default_migrations`
