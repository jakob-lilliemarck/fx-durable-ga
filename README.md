# fx-durable-ga

A durable, auditable genetic algorithm optimization library built on PostgreSQL.

## What is this?

fx-durable-ga is designed for **long-running genetic algorithm optimizations** where durability and auditability matter more than framework speed. It's built for scenarios where fitness evaluations are expensive (seconds to hours) and you need:

- **Crash recovery**: Resume optimizations exactly where they left off
- **Full audit trails**: Every evaluation, generation, and decision is recorded
- **Concurrent execution**: Multiple workers can contribute to the same optimization

## When to use

**Well suited for:**
- AI model hyperparameter optimization
- Neural architecture search
- Feature selection for ML models
- Any optimization where evaluation takes much longer than the GA framework overhead and where parameters can be represented as discrete numbers.

**Not ideal for:**
- Fast, in-memory optimizations (use traditional GA libraries)
- Real-time applications requiring sub-second responses

## How does it work?

The library uses PostgreSQL for both storage and coordination:

1. **Durable state**: All populations, genotypes, and evaluations persist in the database
2. **Event-driven**: Optimizations progress through database events, enabling crash recovery
3. **Deduplication**: Identical genomes are never evaluated twice for the same request
4. **Smart initialization**: Latin Hypercube Sampling for better space coverage
5. **Fitness-based termination**: Automatic stopping when target fitness thresholds are reached

Network latency to the database is the primary overhead, but this is negligible when fitness evaluations take seconds or longer.

## Quick start

```rust
use fx_durable_ga::{bootstrap, models::*};
use futures::future::BoxFuture;

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
            bounds[0].encode_f64(self.learning_rate).expect("learning_rate within bounds"),
            bounds[1].encode_f64(self.batch_size as f64).expect("batch_size within bounds"),
        ]
    }

    fn decode(genes: &[i64]) -> Self::Phenotype {
        let bounds = Self::morphology();
        MyParams {
            learning_rate: bounds[0].decode_f64(genes[0]),
            batch_size: bounds[1].decode_f64(genes[1]) as i64,
        }
    }
}

// 3. Implement fitness evaluation
struct MyEvaluator;
impl Evaluator<MyParams> for MyEvaluator {
    fn fitness<'a>(&self, params: MyParams, _terminated: &'a Box<dyn Terminated>) -> BoxFuture<'a, Result<f64, anyhow::Error>> {
        Box::pin(async move {
            // Your expensive evaluation goes here - make it non-blocking, or run the listener on a separate thread.
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
    FitnessGoal::maximize(0.95)?, // Fitness value to stop at while trying to maximize
    Schedule::generational(100, 10), // 100 generations, 10 per generation.
    Selector::tournament(3, 25),  // Tournament selection
    Mutagen::new(
        Temperature::constant(0.5)?,
        MutationRate::constant(0.1)?,
    ),
    Crossover::uniform(0.5)?,
    Distribution::latin_hypercube(50), // Better than random for most cases
).await?;
```

## Documentation and examples

- **API documentation**: https://docs.rs/fx-durable-ga/0.1.5/fx_durable_ga/index.html or run `cargo doc --open`
- **Examples**:
  1. `examples/point_search.rs` - basic example
  2. `examples/regression_model.rs` - hyperparameter optimization for a regression model

## Migrations
Run migrations using the provided sqlx migrator.
- To run the migrations of this crate only, use `fx_durable_ga::migrations::run_migrations`
- To run migrations of dependencies and the migrations of this crate, use `fx_durable_ga::migrations::run_migrations`. Note that this will use the default schema name for `fx-mq-jobs`, if you wish to use another schema, you will need call each migrator.

### Running migrations for development
Set `DATABASE_URL` in your env and create the database. If you're using `sqlx` cli call `sqlx database create`. Then `SQLX_OFFLINE=true cargo run --bin migrate --features migration` to run migrations for this crate and its dependencies `fx-event-bus` and `fx-mq-jobs`. The migration binary uses the feature flag `"migration"` to exclude all code that is statically typechecked by `sqlx`. Once it has been run you may set `SQLX_OFFLINE=false` and everything should work as normally.

Set `DATABASE_URL` in your environment and create the database (e.g., `sqlx database create`). Run migrations with:

```bash
SQLX_OFFLINE=true cargo run --bin migrate --features migration
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests for bug fixes, improvements, or new features.
