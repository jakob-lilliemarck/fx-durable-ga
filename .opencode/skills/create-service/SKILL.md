---
name: create-service
description: |
  Use when creating a new service module under src/services/. Scaffolds the
  complete service file structure following the project's modular monolith
  conventions. Do NOT use for repositories, controllers, or infrastructure code.
---

# Create Service

Follow these steps when creating a new service.

## Directory structure

Create the service under `src/services/<name>/`:

```
src/services/<name>/
├── mod.rs
├── errors.rs
├── service.rs
├── registrations.rs
├── events.rs          # optional — only if the service publishes events
├── jobs.rs            # optional — only if the service processes background jobs
└── repositories/      # optional — only if the service owns write access to data
    └── mod.rs
```

## Naming

- Directory name: lowercase, singular, underscore-separated (e.g. `genotype_indexing`)
- Service struct: `pub struct Service`
- Error enum: `pub enum Error` in `errors.rs`
- Event file: `events.rs`, job file: `jobs.rs`

## mod.rs

```rust
mod errors;
// mod events;     // uncomment if events.rs exists
// mod jobs;       // uncomment if jobs.rs exists
mod registrations;
mod service;
// pub mod repositories;  // uncomment if repositories/ exists

pub use errors::Error;
pub use registrations::register;
pub use service::Service;

// Re-export event payloads
// pub use events::SomethingHappenedEvent;

// Re-export read repositories (never Write/WriteTx)
// pub use repositories::evaluations::Read as EvaluationsRead;
// pub use repositories::evaluations::Evaluation;
// pub use repositories::evaluations::SearchEvaluationsFilter;
```

Declare only the modules that exist.

**Key visibility rule:** The service's `mod.rs` is the gatekeeper. It must **only re-export `Read`** from its sub-repositories — never `Write` or `WriteTx`. This ensures that only the owning service can mutate its data while any caller can read it.

## errors.rs

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Human-readable description
    #[error("message: {field}")]
    DomainSpecific { field: String },

    /// Convert from a sub-repository
    #[error("...")]
    SomeRepo(#[from] some_repository::Error),

    /// Catch-all for services that use db::begin
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
```

- Domain-specific variants for meaningful failure modes
- `#[from]` for unambiguous conversions; manual `From` impl for conflicting names
- Omit `Internal` if the service does not use `db::begin`

## service.rs

```rust
use tracing::instrument;

pub struct Service {
    some_repo_ro: some_repository::Read,
    some_repo_wr: some_repository::Write,
}

impl Service {
    pub(crate) fn new(
        some_repo_ro: some_repository::Read,
        some_repo_wr: some_repository::Write,
    ) -> Self {
        Self { some_repo_ro, some_repo_wr }
    }

    #[instrument(level = "debug", skip(self))]
    pub async fn do_something(&self) -> Result<(), super::Error> {
        db::begin(self.some_repo_wr.clone(), |tx| {
            Box::pin(async move {
                let mut wr = some_repository::WriteTx::new(tx);
                // ... mutations ...

                let mut publisher = fx_event_bus::Publisher::new_tx(wr.tx());
                publisher.publish(SomeEvent { ... }).await?;
                Ok(())
            })
        }).await?;
        Ok(())
    }
}
```

- Constructor is `pub(crate)` — called only through DI
- `#[instrument(level = "debug", skip(self))]` on non-trivial methods
- Events are published inside the transaction (transactional outbox)

## Atomicity

Service methods that mutate state must be atomic — every write transaction must leave the database in a fully consistent state. Never write to one table without also writing related data or publishing the corresponding event/job within the same transaction. Use `db::begin` to wrap all related mutations:

```rust
db::begin(self.repo_wr.clone(), |tx| {
    Box::pin(async move {
        let mut wr = WriteTx::new(tx);
        wr.do_one_thing(...).await?;
        wr.do_another_thing(...).await?;
        // events/jobs inside the same transaction
        Ok(())
    })
}).await?;
```

Multi-step writes across separate transactions are acceptable (e.g., writing a progress row first, then the full record later), provided each individual transaction is self-consistent on its own.

## registrations.rs

```rust
use crate::infrastructure::di::{Container, ProviderResult};
use futures::future::BoxFuture;
use std::sync::Arc;

fn provide_service(c: &mut Container) -> BoxFuture<'_, ProviderResult<Arc<super::Service>>> {
    Box::pin(async {
        let some_repo_ro = c.get::<some_repository::Read>().await?;
        let some_repo_wr = c.get::<some_repository::Write>().await?;
        Ok(Arc::new(super::Service::new(some_repo_ro, some_repo_wr)))
    })
}

pub fn register(c: &mut Container) {
    c.provide(some_repository::registrations::provide_some_repository_ro);
    c.provide(some_repository::registrations::provide_some_repository_wr);
    c.provide(provide_service);

    // If event/job handlers need wiring:
    // c.invokable(invoke_event_handler_registration);
}
```

- Register sub-repository providers FIRST, then the service
- Handler registration via `c.invokable(...)` when the service consumes events or processes jobs

## events.rs (optional)

Only include when the service publishes events.

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomethingHappenedEvent {
    pub entity_id: Uuid,
}

impl fx_event_bus::Event for SomethingHappenedEvent {
    const NAME: &'static str = "SomethingHappened";
}

impl SomethingHappenedEvent {
    pub fn new(entity_id: Uuid) -> Self {
        Self { entity_id }
    }
}
```

Naming rules:
- Event name (const): PascalCase, past tense — `"SomethingHappened"`
- Payload struct: `{NAME}Event` — `SomethingHappenedEvent`
- Handler struct: `{NAME}Handler` — `SomethingHappenedHandler`

## jobs.rs (optional)

Only include when the service processes background jobs.

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DoSomethingMessage {
    pub entity_id: Uuid,
}

impl fx_mq_jobs::Message for DoSomethingMessage {
    const NAME: &str = "DoSomething";
}

pub(super) struct DoSomethingHandler {
    service: Arc<super::Service>,
}

impl fx_mq_jobs::Handler for DoSomethingHandler {
    type Message = DoSomethingMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, _lease_renewer))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> futures::future::BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            self.service.do_something(message.entity_id).await
        })
    }

    fn max_attempts(&self) -> i32 { 5 }
}

pub(super) fn register_job_handlers(
    builder: fx_mq_jobs::RegistryBuilder,
    service: &Arc<super::Service>,
) -> fx_mq_jobs::RegistryBuilder {
    builder.with_handler(DoSomethingHandler { service: service.clone() })
}
```

Naming rules:
- Message name (const): PascalCase, imperative command — `"DoSomething"`
- Payload struct: `{NAME}Message` — `DoSomethingMessage`
- Handler struct: `{NAME}Handler` — `DoSomethingHandler`

## repositories/ directory (optional)

Only include when the service owns write access to its own database tables. Delegate to the **create-repository** skill for each aggregate.
