# fx-durable-ga
A durable, auditable optimization library built on PostgreSQL.

---
## What is this?

fx-durable-ga is designed for **long-running genetic algorithm optimizations** where durability and auditability matter more than framework speed. It's built for scenarios where fitness evaluations are expensive (seconds to hours) and you need:

- **Crash recovery**: Resume optimizations exactly where they left off
- **Full audit trails**: Every evaluation, generation, and decision is recorded
- **Concurrent execution**: Multiple workers can contribute to the same optimization

---
## When to use

**Well suited for:**
- AI model hyperparameter optimization
- Neural architecture search
- Feature selection for ML models
- Any optimization where evaluation takes much longer than the GA framework overhead and where parameters can be represented as discrete numbers.

**Not ideal for:**
- Fast, in-memory optimizations (use traditional GA libraries)
- Real-time applications requiring sub-second responses

---
## What are the technical characteristics?

This project is made up from set of modular services communicating through events. It leverages PostgreSQL ACID transactions for both persistence and message transport which enables robust eventual consistency, crash recovery and effortless exactly-once semantics. To allow scaled deployments it makes use of listener multiplexing between events, jobs and synchronization messages, each of which serving its own purpose. This project interprets those concepts as follows:
- **Events**: Provides means of communicating across services. A single event is always handled exactly once, within a single transaction on a single host machine, but will fan out to each registered handler. The event is considered handled once each handler has succeeded, meaning that if one handler fails the whole transaction will fail and roll back. Event handling should be kept lightweight and typically just dispatch jobs.
- **Jobs**: Provides means of running expensive, long running background tasks. Jobs are handled exactly once on a single host machine making use of a lease-based model not to keep long-lived transactions to the database. Jobs do no fan out, a single job has a single handler. A job belonging to one service should never be triggered from outside the service boundary, but should be triggered through the services methods or through any event handlers the service registers.
- **Synchronization**: Provides means of synchronizing across host machines. Example use cases includes cancelling in progress async method calls. Synchronization semaphores use at-least once semantics under the hood, but the public interface of the synchronization service may yet provide exactly-once semantics to its consumers.

Events, jobs and synchronization all make use of PostgreSQL LISTEN/NOTIFY as well as a fallback polling mechanism. However each of the three uses a multiplexed listener such that one host machine only need a single connection for listening.

### Resource usage

This project aims to be massively scalable to tens or hundreds of host machines that can collaboratively solve optimization problems. As such in the face of trade offs resource usage should be prioritized as follows, where 1 is most important to conserve:
1. **Database connections**
2. **Network requests**
3. **Memory usage**
4. **CPU**

Database connection is the scarcest resource and the one that is most important to conserve since the database is the key synchronization mechanism. Inserts should be batched whenever possible and any queries run as part of jobs should be light and fast.

---
## Usage and integration

fx-durable-ga is a library crate designed to be integrated into your application. It provides three components:

1. **Event bus** - Coordinates optimization progress through PostgreSQL NOTIFY/LISTEN
2. **Job queue** - Executes fitness evaluations across distributed workers
3. **HTTP API** (optional) - REST interface for creating and monitoring optimizations

You're responsible for spawning the event listener and job workers in your application. This design gives you flexibility to run all components in a single process or scale workers independently across multiple machines.

### Quick start
This library is designed to be integrated into your application. See the examples directory for complete working integrations:

**Basic integration:** See [`examples/point_search.rs`](examples/point_search.rs) for a complete working example showing:
- Implementing the `Optimizer` trait for your domain
- Building the App with your optimization service
- Spawning the event bus listener for coordination
- Spawning job queue workers for evaluation execution
- Creating and monitoring optimization requests

**Advanced features:** See [`examples/gp_function_indexing.rs`](examples/gp_function_indexing.rs) for an integration demonstrating:
- Genotype indexing with embeddings for similarity search
- Training custom LSTM autoencoders on program behavior
- Feature engineering capabilities for complex optimization spaces

---
## Examples

- **`point_search.rs`** - Basic 3D point optimization, good starting point
- **`regression_model.rs`** - Hyperparameter optimization for ML models
- **`gp_point_search.rs`** - Genetic programming for evolving mathematical expressions
- **`gp_function_indexing.rs`** - Advanced indexing with LSTM autoencoders
- **`feature_engineering.rs`** - Feature selection for machine learning

---
## Documentation

- **API documentation**: https://docs.rs/fx-durable-ga or run `cargo doc --open`

---
## Migrations

Set `DATABASE_URL` in your environment and create the database (e.g., `sqlx database create`). Run migrations with:

```bash
SQLX_OFFLINE=true cargo run --bin migrate --features migration
```

This runs migrations for fx-durable-ga and its dependencies (`fx-event-bus` and `fx-mq-jobs`). The migration binary uses the feature flag `"migration"` to exclude code that is statically type-checked by sqlx. Once migrations complete, you can set `SQLX_OFFLINE=false`.

**In your application:** Use `fx_durable_ga::migrations::run_migrations()` to run migrations programmatically. Note that this uses the default schema name for `fx-mq-jobs`. If you need a custom schema, call each migrator separately.

## Contributing

Contributions are welcome! Please feel free to submit pull requests for bug fixes, improvements, or new features.
