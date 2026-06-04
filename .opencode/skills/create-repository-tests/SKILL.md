---
name: create-repository-tests
description: |
  Use when writing tests for repository query functions or repository struct
  methods. Follows the project's testing conventions from AGENTS.md. Do NOT
  use for service-level or integration tests.
---

# Create Repository Tests

## Placement

Per AGENTS.md:

| Code in | Tests in |
|---|---|
| `queries.rs` (or `queries/` file) | same file (`#[cfg(test)]` inline) |
| `repository.rs` | same file or `repositories/[name]/tests/` |

Query functions are the primary unit to test — repository structs are thin wrappers that usually don't need their own tests.

## Test boilerplate

```rust
#[sqlx::test(migrations = false)]
async fn it_describes_behavior(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;
    // ...
    Ok(())
}
```

- Always use `#[sqlx::test(migrations = false)]` — fresh DB per test
- Always call `run_default_migrations` as the first line
- Return `anyhow::Result<()>` so you can use `?`

## Seed function

One `seed` function per test module, called at the start of each test (or lazily by tests that need it). Uses existing query and repository methods to set up data — never raw SQL.

```rust
async fn seed(pool: &PgPool) -> anyhow::Result<TestData> {
    let request = store_request(pool, some_request).await?;
    let genotypes = store_genotypes(pool, &genotypes).await?;
    let evaluations = store_evaluations(pool, &evals).await?;
    Ok(TestData { request_id: request.id, genotype_ids: ... })
}
```

Create a comprehensive set of data — a thin seed misses cross-concern conflicts that a realistic baseline would catch.

## What to test for each query function

Test each option independently plus the combined case:

- **Default** — no filters returns all matching rows
- **Each filter** — single filter narrows results correctly
- **No match** — filter that matches nothing returns empty
- **Ordering** — ascending / descending where applicable
- **Limit** — returns at most N rows
- **Combined** — filters + ordering + limit together
- **Cursor/pagination** — cursor boundary cases

One test function per scenario. Tests are pure, inline Rust code — no AAA comments, no helper abstractions for Act/Assert.

## Example

```rust
#[cfg(test)]
mod tests_search {
    use super::{SearchFilter, search};
    // ... imports ...

    async fn seed(pool: &PgPool) -> anyhow::Result<(Uuid, Vec<Uuid>)> {
        // ... set up data using existing query functions ...
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_all_without_filters(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        seed(&pool).await?;
        let results = search(&pool, &SearchFilter::default()).await?;
        assert_eq!(results.len(), 6);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_by_group(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (group_id, _) = seed(&pool).await?;
        let results = search(&pool, &SearchFilter::default().with_group_id(group_id)).await?;
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.group_id() == group_id));
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_when_no_match(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let results = search(&pool, &SearchFilter::default().with_group_id(Uuid::nil())).await?;
        assert!(results.is_empty());
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_limits_results(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        seed(&pool).await?;
        let results = search(&pool, &SearchFilter::default().with_limit(2)).await?;
        assert_eq!(results.len(), 2);
        Ok(())
    }
}
```

## Repository struct tests

Thin pass-through methods on `Read`/`Write`/`WriteTx` that delegate to a single query function with no additional logic do NOT need separate tests (the query is already tested).

Only test repository methods that compose other repository methods or apply specific filter parameter logic.

When testing repository structs, construct them directly without a DI container:

```rust
let read = Read::new(db::ReadPool { pool: pool.clone() });
let write = Write::new(db::WritePool { pool: pool.clone() });
```

## UUIDs

Use human-readable UUIDs in assertions:
- Parse via `Uuid::parse_str("00000000-0000-0000-0000-000000000001")`
- Or `Uuid::nil()` for the zero UUID

## What NOT to do

- No raw SQL in tests — use existing query functions
- No `// ARRANGE` / `// ACT` / `// ASSERT` comments
- No `sleep` calls — tests must be deterministic
