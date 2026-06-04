---
name: create-repository
description: |
  Use when creating a new repository aggregate inside an existing service's
  repositories/ directory. Scaffolds the complete repository file structure
  including Read/Write/WriteTx struct, models, queries, and DI registrations.
  Do NOT use for services, controllers, or infrastructure code.
---

# Create Repository

Follow these steps when creating a new repository aggregate.

## Directory structure

Create inside `src/services/<service-name>/repositories/<plural-aggregate-name>/`:

```
repositories/<plural-aggregate-name>/
├── mod.rs
├── errors.rs
├── repository.rs
├── registrations.rs
├── models.rs
└── queries/
    ├── mod.rs
    ├── query_file_1.rs
    ├── query_file_2.rs
    └── ...
```

Also update:
- `repositories/mod.rs` — add `pub mod <plural-aggregate-name>`
- Service's `registrations.rs` — add provider registrations

## Naming

- Directory name: lowercase, plural, underscore-separated (e.g. `evaluations`)
- Error enum: `pub enum Error`
- Repository structs: `Read`, `Write`, `WriteTx`

## mod.rs

```rust
mod errors;
mod models;
pub(super) mod queries;
mod repository;

pub mod registrations;

pub use errors::Error;
pub use models::DomainModel;
pub use queries::SearchFilter;

pub use repository::Read;
pub(crate) use repository::Write;
pub(crate) use repository::WriteTx;
```

- Domain models and filters are `pub` — consumable by other services
- `Read` is `pub` — injectable into any caller
- `Write`/`WriteTx` are `pub(crate)` — the owning service's `mod.rs` controls what leaves the service boundary; Write/WriteTx must never be re-exported past the service
- `queries` module is `pub(super)` — visible to the repository parent module; tests access query functions through repository methods, not directly

## errors.rs

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
```

- `Internal` variant only needed when the repository's owning service uses `db::begin`

## repository.rs

```rust
use crate::infrastructure::db;
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
    type Error = Error;

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

    #[instrument(level = "debug", skip(self))]
    pub async fn search(&self, filter: &SearchFilter) -> Result<Vec<DomainModel>, Error> {
        queries::search(&self.ro.pool, filter).await
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
    pub(crate) async fn insert(&mut self, model: &DomainModel) -> Result<DomainModel, Error> {
        queries::insert(&mut **self.tx, model).await
    }
}
```

- Read methods: `pub` — accessible from any caller via the Read struct
- WriteTx mutation methods: `pub(crate)` — only the owning service calls these
- Methods delegate to `queries::*` functions

## registrations.rs

```rust
use crate::infrastructure::db;
use crate::infrastructure::di::{Container, ProviderResult};
use futures::future::BoxFuture;

pub fn provide_<name>_repository_ro(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Read>> {
    Box::pin(async {
        let ro = c.get::<db::ReadPool>().await?;
        Ok(super::Read::new(ro))
    })
}

pub fn provide_<name>_repository_wr(
    c: &mut Container,
) -> BoxFuture<'_, ProviderResult<super::Write>> {
    Box::pin(async {
        let wr = c.get::<db::WritePool>().await?;
        Ok(super::Write::new(wr))
    })
}
```

- Provider names follow: `provide_<name>_repository_ro` and `provide_<name>_repository_wr`

## models.rs

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainModel {
    pub(crate) id: Uuid,
    pub(crate) name: String,
}

impl DomainModel {
    pub fn id(&self) -> Uuid { self.id }
    pub fn name(&self) -> &str { &self.name }
}
```

- Fields are `pub(crate)` — accessible within the service
- Accessor methods are `pub` — public read interface
- Use `#[derive(Debug, Clone, Serialize, Deserialize)]`

## queries/mod.rs

```rust
mod insert;
mod search;

pub use insert::{insert, InsertFilter};
pub use search::{search, SearchFilter};
```

- Export query functions and their filter structs
- Write query functions (like `insert`) use `pub(crate)` if they should not leak

## queries/<name>.rs

```rust
use super::super::Error as RepositoryError;
use super::super::DomainModel;
use sqlx::PgExecutor;
use tracing::instrument;

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

/// Description of the query
#[instrument(level = "debug", skip(tx))]
pub async fn search<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &SearchFilter,
) -> Result<Vec<DomainModel>, RepositoryError> {
    let rows = sqlx::query_as!(
        DomainModel,
        r#"
        SELECT id, name
        FROM schema.table
        WHERE ($1::uuid[] IS NULL OR id = ANY($1))
        LIMIT $2
        "#,
        filter.ids.as_deref(),
        filter.limit,
    )
    .fetch_all(tx)
    .await?;

    Ok(rows)
}
```

- Use `sqlx::query_as!` / `sqlx::query!` for compile-time verification
- Use `RETURNING` on mutations
- Always require `LIMIT` on collection queries
- Use `$1 IS NULL OR col = $1` pattern for optional filters
- Filter types use a builder pattern with `with_*` methods
- Arguments are references (`&str`, `&Uuid`) — not owned values
- Always include `#[cfg(test)]` inline tests with `#[sqlx::test(migrations = false)]`
