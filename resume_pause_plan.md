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

What was missing: a **public API** to trigger this. External callers don't know
the `account_id` — they only know the `request_id`.

## Scope

No event-sourced state tables, no budget delta tables, no explicit pause
mechanism. Just a thin public method on `optimization::Service`.

## Completed work

### `add_budget(request_id, amount)` → `Result<(), Error>`

- [x] Public method on `optimization::Service` that looks up request by ID and
      calls `budgeting.append()` with the correct `account_id`
- [x] Uses `REASON_OPTIMIZATION_BUDGET_ADDED` — distinct from `CREATED` (initial
      request) and `CHARGED` (debit), ensuring no semantic collision with either
- [x] `TransactionCreatedHandler` processes `BUDGET_ADDED` identically to `CREATED`
      (dispatches `ChargeOptimizationBudgetMessage`)
- [x] Tests: error on unknown request, transaction created with correct
      reason/amount

### Removed from scope
- `resume()` convenience wrapper was skipped per feedback — `add_budget` is
  the only method needed

## Known Issue — Event propagation from direct `append()` calls

Events published via `budgeting::Service::append()` (the path used by
`add_budget`) are **not** picked up by the event listener, while the same events
published through the job chain (`AddOptimizationBudgetMessage` → job handler →
`append()`) work correctly. This means `add_budget` correctly credits the
account but the charge-then-breed pipeline is not triggered.

**Symptoms:** `sync.wait_for()` (which awaits the budget-exhaustion semaphore)
blocks indefinitely after `add_budget`. The balance increases then stays flat —
no charge transaction is created.

**Likely root cause:** The two paths differ in transaction/connection context
when writing to the event outbox, causing the PG NOTIFY to not reach the
listener. The job-dispatched path goes through a `db::begin` within a job
handler (which has its own connection context), while the direct path uses a
`db::begin` on the budgeting write pool with no wrapping job context.

**Fix needed:** Investigate the outbox notification mechanism — it may be that
the event outbox's NOTIFY trigger or the listener's channel subscription depends
on the originating connection or schema search path. Once fixed, the integration
test (polls balance increase then decrease after `add_budget`) should pass.

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
