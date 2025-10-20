# fx-durable-ga

A durable, auditable genetic algorithm optimization library built on PostgreSQL.

## What is this?

fx-durable-ga is designed for **long-running genetic algorithm optimizations** where durability and auditability matter more than framework speed. It's built for scenarios where fitness evaluations are expensive (seconds to hours) and you need:

- **Crash recovery**: Resume optimizations exactly where they left off
- **Full audit trails**: Every evaluation, generation, and decision is recorded
- **Concurrent execution**: Multiple workers can contribute to the same optimization
- **Parameter tracking**: Complete history of what was tried and why

## When to use this

**Perfect for:**
- AI model hyperparameter optimization
- Neural architecture search
- Feature selection for ML models
- Any optimization where evaluation takes much longer than the GA framework overhead

**Not ideal for:**
- Fast, in-memory optimizations (use traditional GA libraries)
- Real-time applications requiring sub-second responses
- Simple parameter sweeps (use grid search)

## How it works

The library uses PostgreSQL as both storage and coordination layer:

1. **Durable state**: All populations, genotypes, and evaluations persist in the database
2. **Event-driven**: Optimizations progress through database events, enabling crash recovery
3. **Deduplication**: Identical genomes are never evaluated twice, even across restarts
4. **Smart initialization**: Latin Hypercube Sampling and Sobol sequences for better coverage
5. **Early termination**: Automatic stopping when populations converge or plateau

Network latency to the database is the primary overhead, but this is negligible when fitness evaluations take seconds or longer.

## Quick start

```rust
use fx_durable_ga::{bootstrap::bootstrap, models::*};

// 1. Define your optimization target
#[derive(Debug, Clone)]
struct MyParams {
    learning_rate: f64,
    batch_size: i64,
}

// 2. Implement genetic encoding
impl Encodeable for MyParams {
    const NAME: &'static str = "MyParams";
    type Phenotype = MyParams;

    fn morphology() -> Vec<GeneBounds> {
        vec![
            GeneBounds::decimal(0.001, 1.0, 1000, 3).unwrap(),
            GeneBounds::integer(16, 512, 32).unwrap(),
        ]
    }

    // encode() and decode() implementations...
}

// 3. Implement fitness evaluation
struct MyEvaluator;
impl Evaluator<MyParams> for MyEvaluator {
    fn fitness(&self, params: MyParams, _terminated: &Box<dyn Terminated>) -> BoxFuture<Result<f64, anyhow::Error>> {
        Box::pin(async move {
            // Your expensive evaluation here
            let accuracy = train_model(params).await?;
            Ok(accuracy)
        })
    }
}

// 4. Start optimization
let service = bootstrap(pool).await?
    .register::<MyParams, _>(MyEvaluator).await?
    .build();

let request = OptimizationRequest::builder()
    .population_size(50)
    .max_generations(100)
    .build();

service.start_optimization::<MyParams>(request).await?;
```

## Documentation and examples

- **API documentation**: Run `cargo doc --open` for comprehensive API docs
- **Examples**: See `examples/optimize_cube.rs` for a complete working example
- **Code documentation**: All public APIs include detailed usage examples

## Development setup

1. Set up PostgreSQL and configure `.env.local` with your database URL
2. Run migrations: The library handles schema setup automatically
3. Generate SQLx cache: `cargo sqlx prepare` (optional, for offline compilation)
4. Run examples: `cargo run --example point-search`
5. Run tests: `cargo test`
