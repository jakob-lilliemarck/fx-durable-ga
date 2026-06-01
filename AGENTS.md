# Agent Development Guidelines

This document outlines the process, conventions, and patterns to follow when developing and modifying this project.

## Working Methodology

1. Always re-read a file prior to mutating its contents — the contents may have changed.
2. Always run `cargo check` before completing a task (only changed files needed). Check for:
   - no errors
   - no deprecation warnings
   - no unused imports
3. Always run `cargo test` before completing a task (only affected tests).

## Documentation Conventions

- Documentation should be present but **brief** and **simple**. No lengthy doc comments or code comments.
- **Doc comments** `///` are intended for the *client user*. They document the contract and clearly outline how the user may interact with the code and what to expect from it.

**Documentation by visibility level:**
- `fn` (private) — No doc comments needed.
- `pub(super)`, `pub(self)`, `pub(crate)` — Simple, brief doc comments describing what the method does, how to call it, and what to expect.
- `pub fn` (public API) — More comprehensive doc comments describing what the method does, how to call it, what to expect, and examples for complex calls.

**Doc comments** `///`
- Are intended for external consumers. Should focus on *what the method does*, *how to call it*, and *what to expect*. They must NOT reveal implementation details — those belong in code comments.

**Code comments** `//`
- Are intended for the *developer*. They capture important implementation details and make the code more approachable, especially for logic conditions that may be hard for humans to understand.

After writing documentation, always run doc tests for the modified file to ensure correctness.

## Architecture

This project follows a **modular monolith** pattern — organized as a set of services that communicate through events. No service depends on another service's internal implementation.

**Services** are the sole carrier of domain logic. They expose public business-method interfaces for other services to call.

**Repositories** are pure data-access and mutation layers — they contain **zero business logic**. Every repository belongs to exactly one service.

**Write repositories** are responsible for data mutations (insert, update, delete). They MUST:
- Be kept **private** to their owning service module — never re-exported or made `pub`
- Only be injectable into the owning service

**Read repositories** are responsible for queries. They are `pub` and can be injected into any service or controller that needs read access.

This split ensures:
- Only the owning service can write to its data
- Other services can read data without coupling to write internals
- Business logic stays in services, never in repositories

### Database Schemas

Each service gets its own database schema. Repositories within the same service share that schema. Foreign keys and referential integrity are constrained to the service boundary — never across services. Services are **bounded contexts**.

### Service Communication

Services communicate in exactly two ways:

1. **Direct invocation** — A service can nest another service and call its public methods.
2. **Events** — Events are public. Any service can register a handler for events published by another service.

### Repository Boundaries

A repository models an **aggregate** — the tightly coupled set of tables that form a single domain concept. Joins across tables within the same repository are fine. Cross-repository joins are forbidden — never join to tables owned by another repository.

## Module Structure

Every service follows a consistent file layout:

| File | Purpose |
|---|---|
| `mod.rs` | Module declarations + `pub use` re-exports |
| `errors.rs` | Error enum (see Error Handling) |
| `service.rs` | Main service struct + impl |
| `registrations.rs` | DI registration function |
| `events.rs` | Event structs + handlers (optional) |
| `jobs.rs` | Job messages + handlers (optional) |
| `repositories/` | Aggregate-based repository directories (optional) |

**Visibility rules:**
- **Events** are `pub` — they are exported from the service module so other services can register handlers for them.
- **Jobs** are `pub(super)` — they are private to the service module. Jobs are an internal async execution mechanism and must never be triggered from outside the service boundary. Instead, other services should call the service's public methods or handle its events.

### Repository Module Structure

Every repository (sitting inside a `repositories/` directory within a service, or as a top-level `repositories/` module) follows a consistent file layout:

| File | Purpose |
|---|---|
| `mod.rs` | Module declarations + `pub use` re-exports |
| `errors.rs` | Error enum (see Error Handling) |
| `repository.rs` | `Read`, `Write`, `WriteTx` struct definitions (see Repository structure) |
| `registrations.rs` | DI provider functions |
| `queries.rs` or `queries/` | Raw SQL query functions (for repositories with write access). A single file for 1-2 queries; a directory module when many queries exist |
| `models.rs` | Domain model types (alternative to `queries.rs` for read-only or model-focused repositories) |

### Registrations Conventions

Every repository MUST contain a `registrations.rs` file with one provider function per repository struct:

- `provide_[name]_repository_ro` — provides the `Read` struct
- `provide_[name]_repository_wr` — provides the `Write` struct
- Additional providers for any other structs defined within the repository module

Repository `registrations.rs` MUST NOT contain a helper `register` method. Provider functions are registered individually by the owning service.

Service `registrations.rs` MAY contain a `pub fn register(c: &mut Container)` method. When present, it MUST register the service itself and all provider functions from the service module's sub-repositories.

## Error Handling

Every service defines its own error enum in `errors.rs` using `thiserror`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Domain-specific error
    #[error("resource not found: {0}")]
    NotFound(String),

    /// Catch-all for any unclassified errors
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
```

- Domain-specific variants capture meaningful failure modes with descriptive `#[error("...")]` messages.
- The `Internal(#[from] anyhow::Error)` catch-all variant converts unexpected errors. The `#[from]` attribute auto-generates `From` impls.
  **Note:** Only include `Internal` when the service uses `db::begin` (which wraps errors into `anyhow::Error`). Services that don't use `db::begin` may omit it.
- Repository-level errors follow a simpler pattern — just `Database(#[from] sqlx::Error)` and optionally a generic error variant.

## Testing Conventions

1. **No ad-hoc SQL** — Do not write raw SQL queries directly in tests for seeding data or making assertions. Import and use the existing query methods from the repositories. All necessary queries are exposed for testing purposes.

2. **Centralized seeding** — Each test module should contain a single `seed` function responsible for setting up all necessary data within that module, called at the start of each test. To better mimic a real-world environment, the `seed` function should create a comprehensive set of data, even if a specific test does not use all of it.

3. **Clear Arrange, Act, Assert**:
   - **Arrange** — It is acceptable to use helper methods for data construction and seeding.
   - **Act & Assert** — Must be written as pure, inline Rust code. Avoid abstracting this logic into helper functions so the core test logic is easy to read top to bottom.

4. **No AAA comments** — Do not add comments like `// ARRANGE`, `// ACT`, or `// ASSERT`. The structure of the test should make these sections self-evident.

5. **Avoid arbitrary sleeps** — Tests should be deterministic. Avoid `sleep` calls for asynchronous operations. Use synchronous tests or event-driven mechanisms (notifications, polling with timeout) to wait for a specific state.

6. **Human-readable UUIDs** — When writing string UUIDs in tests, keep them readable like `"00000000-0000-0000-0000-000000000001"`, `"00000000-0000-0000-0000-000000000002"`, etc.

7. **Test coverage requirements** — Every public function in a service and every function in a repository's `queries` module must have tests. Repository struct wrappers (Read/Write/WriteTx) are thin wrappers that delegate to `queries` — they do not need separate testing.

8. **`#[sqlx::test]` with manual migrations** — Tests use `#[sqlx::test(migrations = false)]` to get a fresh database pool. The first line of the test function calls `crate::migrations::run_default_migrations(&pool).await?` to apply schema.

9. **DI container in tests** — Build a full `Container` by calling `crate::register(&mut c)`, then override pools with the test pool:
    ```rust
    let mut c = Container::new();
    crate::register(&mut c);
    c.provide(|_| Box::pin(async { Ok(db::WritePool { pool: pool.clone() }) }));
    c.provide(|_| Box::pin(async { Ok(db::ReadPool { pool: pool.clone() }) }));
    c.invoke().await?;
    let app = c.get::<Arc<App>>().await?;
    ```

## Repository and Query Patterns

### Repository structure

Every repository with write access follows a three-struct pattern:

- **`Read`** — Holds a read pool (`db::ReadPool`). Contains query methods that delegate to the `queries` module.
- **`Write`** — Holds a write pool (`db::WritePool`). Implements `db::Tx` to begin transactions.
- **`WriteTx<'tx>`** — Holds `&'tx mut PgTransaction`. Contains mutation methods that delegate to the `queries` module.

Read-only repositories (no mutations) only need the `Read` struct.

### Queries module

Raw SQL functions live in a dedicated `queries.rs` (or `queries/` directory) per aggregate. Repository structs are thin wrappers that delegate to these functions — they contain **no business logic**. The query functions are the units that should be tested.

### Transactional outbox

To guarantee eventual consistency, events and jobs are published **inside the same database transaction** that performs the data mutation:

```rust
db::begin(self.writer.clone(), |tx| {
    Box::pin(async move {
        let mut wr = WriteTx::new(tx);
        wr.insert_something(...).await?;
        let mut publisher = fx_event_bus::Publisher::new_tx(wr.tx());
        publisher.publish(SomeEvent { ... }).await?;
        Ok(())
    })
}).await?;
```

This ensures the mutation and the published event/job are committed atomically. If the transaction fails, the event is never published.

### Filter patterns

Query filter types use a builder pattern with `with_*` methods returning `Self`:

```rust
#[derive(Default)]
pub struct Filter {
    field: Option<Type>,
}

impl Filter {
    pub fn with_field(mut self, value: Type) -> Self {
        self.field = Some(value);
        self
    }
}
```

### Query rules

1.  **Statically checked queries** — Prioritize `sqlx::query_as!` and `sqlx::query!` to leverage compile-time SQL verification.

2.  **Use `RETURNING` for mutations** — When inserting or updating data, use a `RETURNING` clause to fetch the data that was just written. This confirms the operation succeeded and returns the actual persisted state.

3.  **Clean SQL formatting** — Format SQL queries for readability. Place clauses (`INSERT`, `VALUES`, `RETURNING`, etc.) on new lines and indent columns for clarity.

4.  **Specific error handling** — Map generic database errors to specific, meaningful application-level errors. For example, `sqlx::Error::RowNotFound` should become a domain-specific `Error::NotFound`.

5.  **Accessor methods and private fields** — Keep struct fields `pub(super)`. Expose data via public accessor methods (e.g., `budget.id()`) rather than direct field access (e.g., `budget.id`).

6.  **Use references for function arguments** — For non-`Copy` types (`String`, `Uuid`, etc.), pass arguments as references (e.g., `&str`, `&Uuid`). The function body is responsible for cloning if it needs ownership.

## Logging and Instrumentation

- Use `#[instrument(level = "debug")]` to instrument methods. The default level is `"info"`, so explicitly specify `"debug"`.
- **Most methods should be instrumented**, except trivial methods:
  - Simple getters that do not compute anything
  - Simple constructors that just pass values through
  - Getters that compute values or constructors with logic **should** be instrumented
- Review existing `#[instrument]` annotations to ensure they cover all loggable arguments and use the correct log level.
- Add `#[instrument(level = "debug")]` to methods that are missing it.
- Events of special **business concern** should be logged with `tracing::info!`.
- Events that **should never occur** should be logged with `tracing::warn!`.
- Errors that are **swallowed or handled** should be logged with `tracing::error!`. Do not log errors that are returned to the caller or otherwise handled in the calling code.

## Other Guidelines

- Use `Uuid::now_v7()`, not `Uuid::new_v4()`.
