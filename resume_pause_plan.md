# Resume / Pause — Completion Plan

## Current state

Resume works implicitly through the event bus:

```
budgeting::append(request.account_id, ..., "OptimizationCharged")
    → TransactionCreatedEvent
    → TransactionCreatedHandler (optimization/events.rs)
    → dispatches BreedGenotypesMessage
```

The `TransactionCreatedHandler` already dispatches `BreedGenotypesMessage` for
transactions with reason `"OptimizationCharged"`. Since the optimization
pipeline is entirely job-driven, adding budget to an exhausted account is
sufficient to resume breeding.

What's missing: a **public API** to trigger this. External callers don't know
the `account_id` — they only know the `request_id`.

## Scope

No event-sourced state tables, no budget delta tables, no explicit pause
mechanism. Just two thin public methods on `optimization::Service`.

## Work items

### 1. `add_budget(request_id, amount)` → `Result<(), Error>`

A public method on `optimization::Service`:

1. Looks up the request via `requests_ro.get_request(request_id)`
2. Calls `self.budgeting.append(request.account_id, "OptimizationBudget", amount, "OptimizationCharged")`
3. The event bus picks up the `TransactionCreatedEvent` and dispatches
   `BreedGenotypesMessage`

**Why `"OptimizationCharged"` as reason?** The `TransactionCreatedHandler`
checks for this exact string and dispatches breeding only when it matches.
Using `"OptimizationCharged"` ensures the breeding pipeline is triggered.

### 2. `resume(request_id, amount)` → `Result<(), Error>` (convenience)

A thin wrapper around `add_budget`. Semantically clearer for callers. If
you prefer to keep the API surface minimal, we can skip this and expose only
`add_budget`.

## Dependencies

- `budgeting::Service` — already injected as `self.budgeting` in the
  optimization service
- `requests_ro` — already a field on `optimization::Service`
- `budgeting.append()` — already a public method

## Files to change

| File | Change |
|------|--------|
| `src/services/optimization/service.rs` | Add `add_budget(request_id, amount)` (and optionally `resume`) public methods |
| `src/services/optimization/service.rs` (tests) | Add tests for `add_budget` |

## Usage

```rust
// After an optimization has exhausted its budget:
let request_id = Uuid::parse_str("...")?;
svc.add_budget(request_id, 1000).await?;
// → budget is credited, event bus fires, breeding resumes
```

## Non-goals (explicitly excluded)

- No `OptimizationState` table or state machine
- No `budget_deltas` table
- No explicit `pause` method (semaphores already handle exhaustion)
- No event-sourced rebuild
