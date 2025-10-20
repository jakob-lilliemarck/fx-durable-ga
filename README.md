# fx-durable-ga

A durable, auditable genetic algorithm optimization library built on PostgreSQL.

## What is this?

fx-durable-ga is designed for **long-running genetic algorithm optimizations** where durability and auditability matter more than framework speed. It's built for scenarios where fitness evaluations are expensive (seconds to hours) and you need:

- **Crash recovery**: Resume optimizations exactly where they left off
- **Full audit trails**: Every evaluation, generation, and decision is recorded
- **Concurrent execution**: Multiple workers can contribute to the same optimization
- **Parameter tracking**: Complete history of what was tried

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
3. **Deduplication**: Identical genomes are never evaluated twice for the same request
4. **Smart initialization**: Latin Hypercube Sampling for better space coverage
5. **Fitness-based termination**: Automatic stopping when target fitness thresholds are reached

Network latency to the database is the primary overhead, but this is negligible when fitness evaluations take seconds or longer.

## Quick start

```rust
use fx_durable_ga::{bootstrap::bootstrap, models::*};
use futures::future::BoxFuture;

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

    fn encode(&self) -> Vec<i64> {
        let bounds = Self::morphology();
        vec![
            bounds[0].from_sample((self.learning_rate - 0.001) / (1.0 - 0.001)),
            bounds[1].from_sample((self.batch_size - 16) as f64 / (512 - 16) as f64),
        ]
    }

    fn decode(genes: &[i64]) -> Self::Phenotype {
        let bounds = Self::morphology();
        MyParams {
            learning_rate: bounds[0].to_f64(genes[0]),
            batch_size: (bounds[1].to_f64(genes[1]) * (512 - 16) as f64 + 16.0) as i64,
        }
    }
}

// 3. Implement fitness evaluation
struct MyEvaluator;
impl Evaluator<MyParams> for MyEvaluator {
    fn fitness<'a>(&self, params: MyParams, _terminated: &'a Box<dyn Terminated>) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
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

service.new_optimization_request(
    MyParams::NAME,
    MyParams::HASH,
    FitnessGoal::maximize(0.95)?, // Stop at 95% accuracy
    Schedule::generational(100, 10), // 100 generations, 10 parallel
    Selector::tournament(3, 50),  // Tournament selection
    Mutagen::new(
        Temperature::constant(0.5)?,
        MutationRate::constant(0.1)?,
    ),
    Crossover::uniform(0.5)?,
    Distribution::latin_hypercube(50), // Better than random for most cases
).await?;
```

## Documentation and examples

- **API documentation**: Run `cargo doc --open` for comprehensive API docs
- **Examples**: See `examples/point_search.rs` for a complete working example
- **Code documentation**: All public APIs include detailed usage examples

## Development setup

1. Set up PostgreSQL and configure `DATABASE_URL` with your database URL
2. Run migrations using sqlx cli `sqlx migrate run`. If you run into issues with missing relations from jobs or events, use offline mode and prepare query cache.
3. Generate SQLx cache: `cargo sqlx prepare` (optional, for offline compilation)
4. Run examples: `cargo run --example point-search`
5. Run tests: `cargo test`
6. Generate test coverage: `cargo llvm-cov --html --output-dir coverage`

## Contributing

Contributions are welcome! Please feel free to submit pull requests for bug fixes, improvements, or new features.
