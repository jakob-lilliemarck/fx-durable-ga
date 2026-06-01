#![cfg(feature = "test-tools")]

use chrono::Utc;
use futures::lock::Mutex;
use fx_durable_ga::infrastructure::di::Container;
use fx_durable_ga::services::evaluation;
use fx_durable_ga::services::optimization::{self as foreign_service, OptimizerRegistry};
use fx_durable_ga::services::optimization::{FitnessGoal, Schedule, Selector};
use fx_durable_ga::{configuration, migrations};
use fx_mq_building_blocks::testing_tools::{get_all_messages, is_failed};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

// Tests that stopping the app marks in-progress jobs as failed
// NOTE: Currently ignored because app.stop() is not yet implemented (see FIXME in bootstrap.rs)
#[sqlx::test(migrations = false)]
#[ignore]
async fn test_stopping_marks_jobs_as_failed(
    pool_opts: PgPoolOptions,
    connect_opts: PgConnectOptions,
) -> anyhow::Result<()> {
    fx_durable_ga::test_tools::init_test_tracing();

    let pool = pool_opts
        .clone()
        .max_connections(12)
        .connect_with(connect_opts.clone())
        .await?;

    migrations::run_default_migrations(&pool).await?;

    dotenvy::from_filename(".env.local").ok();

    // Create a DI container
    let mut c = Container::new();

    let started = Arc::new(Notify::new());

    // Invoke optimizer registration
    let started_clone = started.clone();
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(
                "TestType",
                TestOptimizer {
                    started: started_clone,
                },
            );
            Ok(())
        })
    });

    // Invoke evaluator registration
    let started_clone2 = started.clone();
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<evaluation::Service>>().await?;
            provided
                .register(
                    "TestType",
                    TestOptimizer {
                        started: started_clone2,
                    },
                )
                .await;
            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    fx_durable_ga::register(&mut c);

    // Overwrite the database pool provider
    let provided_pool = pool.clone();
    c.provide(|_| Box::pin(async { Ok(provided_pool) }));

    // Overwrite the job worker count configuration provider
    c.provide(|_| Box::pin(async { Ok(configuration::JobWorkerCount { value: 1 }) }));

    // Invoke all invokables
    c.invoke().await?;

    // Get an app instance from the container
    let app = c.get::<Arc<fx_durable_ga::bootstrap::App>>().await?;

    let request_id = app
        .services()
        .optimization()
        .request_new(
            String::from("TestType"),
            FitnessGoal::maximize(0.95)?,
            Schedule::generational(1, 1),
            Selector::tournament(3),
        )
        .await?;

    println!("Optimization request submitted: {}", request_id);

    started.notified().await;

    // ACT by calling stop
    // The core functionality this test verifies, is that in-progress jobs are properly marked as failed
    // NOTE: app.stop() is currently a no-op (FIXME in bootstrap.rs), so this test may timeout
    app.stop().await?;

    let messages = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            // The job framework updates job state asynchronously, so we loop until the
            // evaluation jobs exist and are marked failed after shutdown.
            let messages = get_all_messages(&pool).await?;
            let eval_messages: Vec<_> = messages
                .iter()
                .filter(|message| message.name == "EvaluateGenotypeMessage")
                .collect();

            let now = Utc::now();
            let mut all_failed = true;
            for message in &eval_messages {
                if !is_failed(&pool, message.id, now).await? {
                    all_failed = false;
                    break;
                }
            }

            if !eval_messages.is_empty() && all_failed {
                return Ok::<_, anyhow::Error>(messages);
            }

            tokio::task::yield_now().await;
        }
    })
    .await??;

    let eval_messages: Vec<_> = messages
        .iter()
        .filter(|message| message.name == "EvaluateGenotypeMessage")
        .collect();
    assert!(
        !eval_messages.is_empty(),
        "expected EvaluateGenotypeMessage jobs to be queued"
    );
    let now = Utc::now();
    for message in &eval_messages {
        let failed = is_failed(&pool, message.id, now).await?;
        assert!(
            failed,
            "expected EvaluateGenotypeMessage jobs to be marked failed"
        );
    }

    Ok(())
}

#[derive(Clone)]
struct TestOptimizer {
    started: Arc<Notify>,
}

impl foreign_service::Optimizer for TestOptimizer {
    type Type = serde_json::Value;

    fn random(&self) -> anyhow::Result<Self::Type> {
        Ok(serde_json::json!({}))
    }

    fn crossover(
        &self,
        _parent1: Self::Type,
        _parent2: Self::Type,
    ) -> anyhow::Result<Self::Type> {
        Ok(serde_json::json!({}))
    }

    fn mutate(
        &self,
        _instance: &mut Self::Type,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl evaluation::Evaluator for TestOptimizer {
    type Type = serde_json::Value;

    fn evaluate<'a>(
        &'a self,
        _instance: &'a Self::Type,
    ) -> futures::future::BoxFuture<
        'a,
        std::result::Result<f64, Box<dyn std::error::Error + Send + Sync>>,
    > {
        let started = self.started.clone();
        Box::pin(async move {
            started.notify_one();
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(0.0)
        })
    }
}
