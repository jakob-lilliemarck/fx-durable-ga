---
name: create-service-tests
description: |
  Use when writing tests for service business methods. Follows the project's
  testing conventions from AGENTS.md. Do NOT use for repository-level tests
  or query-level tests.
---

# Create Service Tests

## Placement

Per AGENTS.md:

| Code in | Tests in |
|---|---|
| `service.rs` | same file or `tests/` directory in the service module |

Service tests must test business logic, not repository internals — test each layer independently.

## Boilerplate

```rust
#[sqlx::test(migrations = false)]
async fn it_describes_behavior(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;
    // ...
    Ok(())
}
```

- Always use `#[sqlx::test(migrations = false)]`
- Always call `run_default_migrations` first
- Return `anyhow::Result<()>`

## Instantiation

Always use the DI container via `TestConfig`:

```rust
let mut c = crate::test_tools::TestConfig::new(pool.clone())
    .with_optimizer(SomeOptimizer)
    .with_indexer(SomeIndexer)
    .build().await?;
let svc = c.get::<Arc<super::Service>>().await?;
```

- `TestConfig::new(pool)` builds a minimal container with all services registered via their normal `register` functions
- `.with_optimizer(...)` / `.with_indexer(...)` are optional — only needed when the test exercises optimizer or indexer code. Many tests can skip them and just use `.build().await?`
- `.with_listeners()` if the service processes events during the test
- Call `c.get::<Arc<Service>>()` to extract the service under test

## Seed function

One `seed` function per test module, called at the start of each test. Uses existing query and repository methods — never raw SQL.

```rust
async fn seed(pool: &PgPool) -> anyhow::Result<SeedData> {
    let request = store_request(pool, some_request).await?;
    let genotypes = store_genotypes(pool, &genotypes).await?;
    Ok(SeedData { request_id: request.id, genotype_ids: ... })
}
```

Create a comprehensive baseline — a thin seed misses cross-concern conflicts.

## What to test per service method

**Business rules / validation errors:**

```rust
#[sqlx::test(migrations = false)]
async fn it_rejects_invalid_input(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;
    let svc = build_service(&pool).await?;
    let result = svc.do_something(invalid_input).await;
    assert!(result.is_err());
    Ok(())
}
```

**Happy path:**

```rust
#[sqlx::test(migrations = false)]
async fn it_creates_the_expected_state(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;
    let svc = build_service(&pool).await?;
    let result = svc.do_something(valid_input).await?;
    assert_eq!(result.some_field(), expected_value);
    Ok(())
}
```

**Event published — count unacknowledged events:**

```rust
#[sqlx::test(migrations = false)]
async fn it_publishes_an_event(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;
    let svc = build_service(&pool).await?;
    let before = get_unacknowledged_events(&pool).await?;
    svc.do_something(input).await?;
    let after = get_unacknowledged_events(&pool).await?;
    assert_eq!(after - before, 1);
    Ok(())
}
```

**Job dispatched — query the job messages table:**

```rust
#[sqlx::test(migrations = false)]
async fn it_dispatches_a_job(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;
    let svc = build_service(&pool).await?;
    svc.do_something(input).await?;
    let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
    let mut tx = pool.begin().await?;
    let jobs = mq.get_all_messages(&mut tx).await?;
    let matching: Vec<_> = jobs.iter().filter(|j| j.name == "SomeJobName").collect();
    assert_eq!(matching.len(), expected_count);
    Ok(())
}
```

**Atomicity — transaction rollback:**

When a service method calls `db::begin`, a failure inside the closure rolls back all mutations. The test should verify that partial state is not persisted.

## Rules

- No raw SQL for seeding or assertions — use existing repository/query methods
- No `// ARRANGE` / `// ACT` / `// ASSERT` comments
- No `sleep` calls — deterministic only
- Human-readable UUIDs: `Uuid::nil()` or `Uuid::parse_str("00000000-0000-0000-0000-000000000001")`
- Service tests must NOT reach into sub-repositories to test repository logic
- Thin pass-through methods that only delegate to a repository do not need service-level tests

## What NOT to test at the service level

- Repository query logic — test that in repository tests
- Repository CRUD wrappers — those are thin delegates, already tested at the query level
- Internal scheduler details
