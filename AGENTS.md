# Agent Development Guidelines

This document outlines the process, conventions, and patterns to follow when developing and modifying this project.

## Working Methodology

1. Always re-read a file prior to mutating its contents — the contents may have changed.
2. Always run `cargo fmt` before completing a task (only changed files needed).
3. Always run `cargo test` before completing a task (only affected tests). Check for:
   - no errors
   - no deprecation warnings
   - no unused imports

## Documentation Conventions

- Documentation should be present but **brief** and **simple**. No lengthy doc comments or code comments.
- Doc comments `///` document the *contract*: arguments, return value, notable errors. They must NOT reveal implementation details.
- Code comments `//` capture implementation details and make the code more approachable.

**Detail by visibility:**
- `fn` (private) — No doc comments needed.
- `pub(super)`, `pub(self)`, `pub(crate)` — Single-line summary of what the method does.
- `pub fn` — Slightly more detail; brief examples for complex calls.

After writing documentation, always run doc tests for the modified file to ensure correctness.

## Import Conventions

Never alias imports. Use qualified namespaces directly in code.

```rust
// GOOD — use qualified namespaces
use super::repositories::a;
use super::repositories::b;

let x = a::Repository { ... };
let y = b::Repository { ... };

// BAD — aliasing hides the type origin
use super::repositories::a::Repository as ARepo;
use super::repositories::b::Repository as BRepo;
```

When two modules export types with the same name, import the parent namespace and qualify:

```rust
use crate::services::budgets;
use crate::services::requests;

let err = budgets::Error { ... };
let err = requests::Error { ... };
```

## Architecture

This project follows a **modular monolith** pattern — organized as a set of services that communicate through events. No service depends on another service's internal implementation.

**Services** are the sole carrier of domain logic. They expose public business-method interfaces for other services to call.

**Repositories** are pure data-access and mutation layers — they contain **zero business logic**. Every repository belongs to exactly one service.

**Write repositories** are responsible for data mutations (insert, update, delete). They are only injectable into the owning service.

**Read repositories** are responsible for queries. They can be injected into any service or controller that needs read access.

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

**Event conventions:** Events are named in past tense (`SomethingHappened`). They carry entity IDs as the primary payload. Denormalized data may be included when it avoids N+1 queries at the consumer side.

**Event naming rules:**
- The associated const `NAME: &str` on the event struct is the event name in PascalCase (e.g. `"GenotypeEvaluated"`).
- The event payload struct is named `{NAME}Event` (e.g. `GenotypeEvaluatedEvent`).
- Event handler structs are named `{NAME}Handler` (e.g. `GenotypeEvaluatedHandler`).

**Job naming rules:**
- Jobs are named as imperative commands (e.g. `DoSomething`).
- The associated const `Message::NAME: &str` follows the same PascalCase convention.
- Job payload structs are named `{NAME}Message` (e.g. `DoSomethingMessage`).
- Job handler structs are named `{NAME}Handler` (e.g. `DoSomethingHandler`).

### Repository Boundaries

A repository models an **aggregate** — the tightly coupled set of tables that form a single domain concept. Joins across tables within the same repository are fine. Cross-repository joins are forbidden — never join to tables owned by another repository.

### Service-Repository Boundary

Services orchestrate business logic. A service method that merely delegates to a
read repository with fixed filter parameters is an anti-pattern — inject the read
repository directly into the caller instead. Service methods should only exist
when they coordinate multiple repositories, enforce business rules, publish events,
dispatch jobs, or manage transactions.

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

| What | Visibility |
|---|---|
| `Service` struct | `pub` |
| Events | `pub` — exported so other services register handlers |
| Jobs | `pub(super)` — private to the service; never triggered from outside |
| Read repositories | `pub` — injectable into any caller |
| Write / WriteTx repositories | private to the owning service |
| Filter structs | `pub` |
| Business types (e.g. `Genotype`, `Request`, `Evaluation`) | `pub` |
| Implementation models (e.g. `DBSomeModel`, internal registries) | private to module |

### Repository Module Structure

Every repository (sitting inside a `repositories/` directory within a service) follows a consistent file layout:

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

### Dependency Injection

This project uses a manual DI container (`src/infrastructure/di.rs`). Every registered type is constructed on first request and cached as a singleton.

Services request dependencies via `c.get::<T>()` in their provider closure:

```rust
pub fn register(c: &mut Container) {
    c.provide::<MyService, _>(|c| {
        Box::pin(async move {
            let read_repo = c.get::<some_repository::Read>().await?;
            Ok(MyService::new(read_repo))
        })
    });
}
```

Each service's `registrations.rs` registers the service struct itself and all provider functions from its sub-repositories.

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
- When converting errors across service boundaries, prefer `#[from]` on `thiserror::Error` variants where the source type is unambiguous. For conflicting conversions (e.g. two repositories both export `Error`), implement `From<ForeignError> for Error` manually or map through the `Internal` variant.

## Testing Conventions

1. **No ad-hoc SQL** — Do not write raw SQL queries directly in tests for seeding data or making assertions. Import and use the existing query methods from the repositories. All necessary queries are exposed for testing purposes. If no existing method supports your assertion, raise the issue before writing a new query — it may belong in a different repository or need design input.

2. **Centralized seeding** — Each test module should contain a single `seed` function responsible for setting up all necessary data within that module, called at the start of each test. To better mimic a real-world environment, the `seed` function should create a comprehensive set of data, even if a specific test does not use all of it. The rationale: a thin seed only targets current test paths; when new code touches shared data, it may silently break. A realistic baseline catches these cross-concern conflicts early.

3. **Clear Arrange, Act, Assert**:
   - **Arrange** — It is acceptable to use helper methods for data construction and seeding.
   - **Act & Assert** — Must be written as pure, inline Rust code. Avoid abstracting this logic into helper functions so the core test logic is easy to read top to bottom.

4. **No AAA comments** — Do not add comments like `// ARRANGE`, `// ACT`, or `// ASSERT`. The structure of the test should make these sections self-evident.

5. **Avoid arbitrary sleeps** — Tests should be deterministic. Avoid `sleep` calls for asynchronous operations. Use synchronous tests or event-driven mechanisms (notifications, polling with timeout) to wait for a specific state.

6. **Human-readable UUIDs** — When writing string UUIDs in tests, keep them readable like `"00000000-0000-0000-0000-000000000001"`, `"00000000-0000-0000-0000-000000000002"`, etc.

7. **Test placement** — Tests live in the same file as the code they test or in a
   `tests/` directory within that module:

   | Code in | Tests in |
   |---|---|
   | `queries.rs` (or `queries/` file) | same file (`#[cfg(test)]` inline) |
   | `repository.rs` | same file or `repositories/[name]/tests/` |
   | `service.rs` | same file or `tests/` directory in the service module |

   A service test must not reach into a child repository to test repository logic;
   test each layer independently. Repository tests construct the repository struct
   directly (e.g. `Read::new(db::ReadPool { pool })`) without a DI container.

   Thin pass-through methods on `Read`/`Write`/`WriteTx` that delegate to a single
   query with no additional logic do not need separate tests (the query is already
   tested). Composition methods that call other repository methods with specific
   filter parameters DO need tests.

8. **`#[sqlx::test]` with manual migrations** — Tests use `#[sqlx::test(migrations = false)]` to get a fresh database pool. The first line of the test function calls `crate::migrations::run_default_migrations(&pool).await?` to apply schema.

9. **Never relax tests due to unexplained failures** — If an assertion fails
   inconsistently, investigate until the root cause is understood. Do not change
   `==` to `>=`, add loose tolerances, or remove assertions to make a test pass.
   A non-deterministic test is a bug report. Either fix the underlying issue or,
   if the test exercises inherently async/event-driven behavior, use deterministic
   synchronization (semaphores, timeouts with polling) to wait for the expected
   state before asserting, as described in point 5. The test must pass reliably
   in isolation AND in the full suite.

## Repositories

### Repository structure

Every repository with write access follows a three-struct pattern:

- **`Read`** — Holds a read pool (`db::ReadPool`). Contains query methods that delegate to the `queries` module.
- **`Write`** — Holds a write pool (`db::WritePool`). Implements `db::Tx` to begin transactions.
- **`WriteTx<'tx>`** — Holds `&'tx mut PgTransaction`. Contains mutation methods that delegate to the `queries` module.

Read-only repositories (no mutations) only need the `Read` struct.

**Accessor methods and private fields** — Keep struct fields `pub(super)`. Expose data via public accessor methods (e.g., `budget.id()`) rather than direct field access (e.g., `budget.id`).

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

### Schema conventions

1.  **No `DEFAULT` values** — The database should be a dumb store, not make assumptions about data. Every column must be explicitly provided on insert.

2.  **Prefer `NOT NULL`** — Avoid nullable columns unless there is a strong reason. If an aggregate has fields that may be populated later, split it into two tables — one written first, the other written once the data exists. As a rule of thumb, strive for zero `NULL` columns.

3.  **UUID over INTEGER / SERIAL for IDs** — Use `UUID` for primary keys. Composite primary keys are acceptable, but never `SERIAL` or `INTEGER` IDs. ID values must be provided by the application, not generated by the database.

### Query rules

1.  **Statically checked queries** — Prioritize `sqlx::query_as!` and `sqlx::query!` to leverage compile-time SQL verification.

2.  **Use `RETURNING` for mutations** — When inserting or updating data, use a `RETURNING` clause to fetch the data that was just written. This confirms the operation succeeded and returns the actual persisted state.

3.  **Clean SQL formatting** — Format SQL queries for readability. Place clauses (`INSERT`, `VALUES`, `RETURNING`, etc.) on new lines and indent columns for clarity.

4.  **Specific error handling** — Map generic database errors to specific, meaningful application-level errors. For example, `sqlx::Error::RowNotFound` should become a domain-specific `Error::NotFound`.

5.  **Use references for function arguments** — For non-`Copy` types (`String`, `Uuid`, etc.), pass arguments as references (e.g., `&str`, `&Uuid`). The function body is responsible for cloning if it needs ownership.

6.  **Always require `LIMIT` on collection queries** — Never query for an unbounded number of rows. Callers must paginate.

7.  **Use key-set (cursor) pagination, never `OFFSET`** — `OFFSET` is inefficient and inconsistent under concurrent writes.

8.  **Single search query per entity** — Never create separate queries for different filter combinations. Use a single `search_[entity_type]` function with a filter parameter that uses conditional SQL (`$1 IS NULL OR column = $1`). Filter fields default to `None`; builder methods follow `with_[attribute_name]` naming but are not required to match field names exactly (e.g., `with_group_ids` for a `group_id` field).

## Controllers

Controllers are HTTP handlers that expose service functionality via REST endpoints. They use **Axum** with the **aide** crate for OpenAPI documentation.

### Module structure

```
src/controllers/
├── mod.rs                  # Router composition + request/response re-exports
├── {resource}.rs           # Handlers + route definitions
└── models/
    ├── mod.rs
    ├── form_or_json.rs      # FormOrJson custom extractor
    └── namespaced_query.rs  # NamespacedQuery custom extractor
```

Each controller file follows a consistent layout:

```
mod.rs                   — module decls + re-exports
{resource}.rs            — handlers + route definitions
models/                  — custom Axum extractors (shared across controllers)
```

### Router pattern

Each controller module exports a `pub fn router(app: Arc<App>) -> ApiRouter` that registers its routes:

```rust
pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .api_route(
            "/resource",
            post_with(create_handler, |op| {
                op.operation_id("create_resource")
                    .tag("Resources")
                    .response_with::<500, Json<ErrorResponse>, _>(
                        |r| r.description("Internal server error"),
                    )
            }),
        )
        .api_route(
            "/resource/{id}",
            get_with(get_handler, |op| {
                op.operation_id("get_resource")
                    .tag("Resources")
                    .response_with::<500, Json<ErrorResponse>, _>(
                        |r| r.description("Internal server error"),
                    )
            }),
        )
        .with_state(app)
}
```

Routes are composed in `controllers/mod.rs`:

```rust
pub fn router(app: Arc<App>) -> ApiRouter {
    ApiRouter::new()
        .merge(resource_a::router(app.clone()))
        .merge(resource_b::router(app))
}
```

### Handler signature conventions

Handlers follow a consistent argument order:

| Parameter | When to use |
|---|---|
| `State(app): State<Arc<App>>` | Always — provides access to services and repositories |
| `Path(id): Path<Uuid>` | Route segment contains `{id}` |
| `NamespacedQuery(params): NamespacedQuery<Q>` | GET with query parameters on HTMX pages |
| `input: FormOrJson<P>` | POST with body (accepts both form and JSON) |
| `Json(payload): Json<P>` | POST with JSON-only body (API endpoints) |
| `headers: HeaderMap` | Almost always — used for content negotiation |
| `uri: Uri` | Needed for building relative URLs within the page |

All handlers return `Response` (not `impl IntoResponse`). Annotate handlers with `#[axum::debug_handler]` for better compile-time error messages.

### Custom extractors

Two custom Axum extractors live in `controllers/models/`:

**`FormOrJson<T>`** — Accepts either `application/json` or URL-encoded form data. Used for endpoints that serve both HTMX forms and API clients. The OpenAPI schema always shows the JSON variant. Access the inner value via `.into_inner()`.

**`NamespacedQuery<T>`** — Strips `{namespace}--` prefixes from query parameter keys. Used on HTMX pages where multiple components on the same page issue requests with overlapping parameter names. Both namespaced keys (`component--search=foo`) and plain keys (`search=foo`) resolve to the same deserialized field.

### Content negotiation (`RenderService`)

Each controller module defines a private `RenderService` struct that dispatches based on request headers:

| Condition | Response body |
|---|---|
| `Accept: application/json` | JSON — the view is serialized via `serde` |
| `HX-Request` header present | HTMX fragment — the view is rendered as partial HTML |
| Neither | Full HTML page — wrapped in `<html><head>` with HTMX, Chart.js, and stylesheets |

```rust
fn respond<T>(&self, headers: &HeaderMap, status: StatusCode, view: &T) -> Response
where
    T: maud::Render + serde::Serialize;
```

Error responses follow the same pattern via `respond_error`:

```rust
fn respond_error(&self, headers: &HeaderMap, status: StatusCode, message: impl std::fmt::Display) -> Response;
```

### API-only endpoints

For endpoints that serve only API clients (no browser UI), use `Json<T>` directly and return typed results:

```rust
use crate::views::errors::ErrorResponse;

async fn create_api_endpoint(
    State(app): State<Arc<App>>,
    Json(payload): Json<Payload>,
) -> Result<(StatusCode, Json<Response>), (StatusCode, Json<ErrorResponse>)> {
    // ...
    Ok((StatusCode::CREATED, Json(response)))
}
```

The error response format is consistent across all endpoints (see `src/views/errors.rs`):

```rust
pub struct ErrorResponse {
    error: String,
}
```

### Request/Response type re-exports

`controllers/mod.rs` re-exports payload and response types for use by API clients:

```rust
pub use resource_a::CreatePayload;
pub use resource_a::CreateResponse;
```

### Payload/query struct conventions

- Derive `Debug, Deserialize, schemars::JsonSchema` on all request payloads and query structs
- Use `#[serde(rename_all = "snake_case")]` on enums
- Use `#[serde(default)]` with custom deserializers for optional fields with non-standard formats
- Query struct fields should be `Option<T>` for optional parameters

## Views

Views are the rendering layer paired with controllers. Each view type implements both `maud::Render` for HTML and `serde::Serialize` for JSON, enabling the content negotiation pattern.

### Module structure

```
src/views/
├── mod.rs
├── diversity.rs
├── errors.rs
├── fitness.rs
├── genotypes.rs
├── optimization.rs
└── population.rs
```

### View pattern

```rust
#[derive(serde::Serialize, schemars::JsonSchema)]
pub struct SomeView {
    id: String,
    name: String,
}

impl SomeView {
    pub fn new(id: String, name: String) -> Self { // ...
    }
    pub fn with_some_flag(mut self, flag: bool) -> Self { // ...
    }
}

impl maud::Render for SomeView {
    fn render(&self) -> Markup {
        html! {
            // Full or partial HTML markup
        }
    }
}
```

### Construction conventions

- Use builder-pattern `with_*` methods for optional fields
- Support partial rendering via an `is_partial` flag — when true, render only the fragment for HTMX swap; when false, render the full layout
- The `Serialize` impl should produce the JSON representation of the data (typically a simplified version of the full view)
- For list views, the `Serialize` impl often delegates to a simplified item type (e.g., `OptimizationListView` serializes as `Vec<Optimization>`)
- Derive `schemars::JsonSchema` on all view types for OpenAPI schema generation

### HTMX conventions

- Lazy-load sections via `hx-get` with `hx-trigger="load"` or `hx-trigger="intersect once"`
- Sub-route URLs are constructed from the current page's URI via `base_url()` + path segment
- Namespace overlapping query parameters using the `{section}--` prefix pattern (handled by `NamespacedQuery`)

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

## Validation Conventions

1. **Validate at boundaries** — Data is validated when it enters the system. Once persisted, business rules are not rechecked on read — `sqlx`'s compile-time checked queries (`query_as!`, `query!`) handle structural integrity on the read path.

2. **Controllers** validate incoming data by type — deserialization of JSON/form data into structs handles this. No explicit validation should be needed in controllers.

3. **Services** validate business rules. If a method requires an argument to be an int between 1 and 17, the service enforces that — it's a business rule, not a type constraint.

4. **Repositories** typically do not need to validate.

## Other Guidelines

- Use `Uuid::now_v7()`, not `Uuid::new_v4()`.
