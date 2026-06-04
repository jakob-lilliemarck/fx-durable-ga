---
name: create-service
description: |
  Use when creating a new service module under src/services/. Scaffolds the
  complete service file structure following the project's modular monolith
  conventions. Do NOT use for repositories, controllers, or infrastructure code.
---

# Create Service

## Step 1 — Scaffold boilerplate

Run the `scaffold-service` tool:

```
serviceName: genotype_indexing    # directory under src/services/
withEvents: true                  # (default: false) generate events.rs stub
withJobs: false                   # (default: false) generate jobs.rs stub
withRepositories: false           # (default: false) generate repositories/mod.rs stub
updateParentMod: true             # (default: true) update src/services/mod.rs
```

This creates:

```
src/services/<name>/
├── mod.rs                # module decls — active for selected options, commented otherwise
├── errors.rs             # Error enum with Internal variant
├── service.rs            # Service struct + constructor + placeholder method
├── registrations.rs      # DI provider + register()
├── events.rs             # (if withEvents) stub event struct
├── jobs.rs               # (if withJobs) stub message + handler
└── repositories/
    └── mod.rs            # (if withRepositories) empty stub
```

`mod.rs` has `mod events;` / `mod jobs;` / `pub(super) mod repositories;` active
when the corresponding flag is true, and commented out otherwise. Re-export
lines for events and repo types are always commented — the developer
uncomments and customizes them.

## Step 2 — Customize

### errors.rs

Add domain-specific error variants. Use `#[from]` for unambiguous
conversions. The `Internal` variant is included by default — remove it if
the service doesn't use `db::begin`.

### service.rs

- Replace the `do_something` placeholder with actual business methods.
- Add field dependencies (read repos, other services) to the struct and
  constructor.

### registrations.rs

- Add `c.get::<T>()` calls in the provider function for each dependency.
- Register repository providers before the service:

```rust
pub fn register(c: &mut Container) {
    c.provide(<aggregate>::registrations::provide_<aggregate>_repository_ro);
    c.provide(<aggregate>::registrations::provide_<aggregate>_repository_wr);
    c.provide(provide_<name>_service);
}
```

- For event/job handling, uncomment the `c.invokable(...)` line and
  implement the actual handler registration.

### events.rs / jobs.rs

Replace the stub types with real event/job types following the naming rules
below.

### repositories/

Use the **create-repository** skill guidance and the `scaffold-repository`
tool for each aggregate.

## Naming conventions (reference)

| Aspect | Convention |
|---|---|
| Directory | lowercase, underscore-separated |
| Service struct | `pub struct Service` |
| Error enum | `pub enum Error` in `errors.rs` |
| Event payload | `{PascalEventName}Event` — published as event bus event |
| Event handler | `{PascalEventName}Handler` |
| Job message | `{PascalCommandName}Message` — implements `fx_mq_jobs::Message` |
| Job handler | `{PascalCommandName}Handler` — implements `fx_mq_jobs::Handler` |
| Job visibility | `pub(super)` — private to the service |
| Provider fn | `provide_{name}_service` |
| `register()` | `pub fn register(c: &mut Container)` |

## Service-Repository Boundary

A service method that merely delegates to a read repository with fixed filter
parameters is an anti-pattern — inject the read repository directly into the
caller instead. Service methods should only exist when they coordinate
multiple repositories, enforce business rules, publish events, dispatch jobs,
or manage transactions.
