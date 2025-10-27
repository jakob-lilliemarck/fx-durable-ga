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
Run migrations using the migrator: https://docs.rs/fx-durable-ga/0.1.5/fx_durable_ga/migrations/fn.run_migrations.html

You may also need to run migrations for
 - fx-mq-jobs: https://docs.rs/fx-event-bus/0.1.7/fx_event_bus/fn.run_migrations.html
 - fx-event-bus: https://docs.rs/fx-mq-jobs/0.1.8/fx_mq_jobs/fn.run_migrations.html
See examples for more detail.

### Running migrations for development
Running `cargo sqlx prepare` may require setting search path options for the `DATABASE_URL` variable:
```sh
DATABASE_URL="postgres://postgres:postgres@localhost:5432/fx-durable-ga?options=-c%20search_path%3Dfx_mq_jobs%2Cfx_event_bus%2Cfx_durable_ga"
```
Keeping the search path parameters may however make tests fail, so some juggling is currently required.

## Contributing

Contributions are welcome! Please feel free to submit pull requests for bug fixes, improvements, or new features.
