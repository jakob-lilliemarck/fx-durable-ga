use crate::repositories::genotypes::{Genotype, store_genotypes};
use crate::services::evaluation::Evaluator;
use crate::services::noise_diagnostics::Service;
use crate::services::optimization;
use crate::test_tools::TestConfig;
use futures::future::BoxFuture;
use fx_event_bus::test_tools::get_unacknowledged_events;
use fx_mq_building_blocks::testing_tools::TestQueries;
use fx_mq_jobs::FX_MQ_JOBS_SCHEMA_NAME;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

struct TestEvaluator;

impl Evaluator for TestEvaluator {
    type Type = serde_json::Value;

    fn evaluate<'a>(
        &'a self,
        _instance: &'a Self::Type,
    ) -> BoxFuture<'a, Result<f64, Box<dyn std::error::Error + Send + Sync>>> {
        Box::pin(async move { Ok(0.5) })
    }
}

#[sqlx::test(migrations = false)]
async fn noise_probe_evaluations_do_not_trigger_maintain_population(
    pool: PgPool,
) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;

    let mut c = TestConfig::new(pool.clone())
        .with_listeners()
        .build()
        .await?;
    let evaluation = c.get::<Arc<crate::services::evaluation::Service>>().await?;
    evaluation.register("test", TestEvaluator).await;

    let svc = c.get::<Arc<Service>>().await?;

    let request = optimization::Request::new(
        "test",
        optimization::FitnessGoal::maximize(1.0).unwrap(),
        optimization::Selector::tournament(3),
        optimization::Schedule::generational(10, 2),
    );
    let request_id = request.id;
    optimization::store_request(&pool, request).await?;

    let probe_request_id = Uuid::parse_str("00000000-0000-0000-0000-000000000004")?;
    let genotype = Genotype::new("test", serde_json::json!([1]), request_id, None, None, None)?;
    let stored = store_genotypes(&pool, &[genotype]).await?;
    let genotype = stored.into_iter().next().unwrap();
    let genotype_id = genotype.id();

    let probe_id = svc
        .new_noise_probe(genotype_id, probe_request_id, 1)
        .await?;

    let mq = TestQueries::new(FX_MQ_JOBS_SCHEMA_NAME);
    let mut tx = pool.begin().await?;
    let probe_job = mq.get_all_messages(&mut tx).await?;
    tx.commit().await?;
    let probe_job = probe_job
        .into_iter()
        .find(|j| j.name == "EvaluateNoiseProbeGenotype")
        .unwrap();

    let payload: serde_json::Value = serde_json::from_value(probe_job.payload.clone())?;
    let msg_genotype: Genotype = serde_json::from_value(payload["genotype"].clone())?;

    let events_before = get_unacknowledged_events(&pool).await?;
    svc.evaluate_probe(probe_id, msg_genotype).await?;

    // Poll until the GenotypeEvaluatedEvent is consumed by the handler
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if get_unacknowledged_events(&pool).await? <= events_before {
            break;
        }
        if Instant::now() > deadline {
            anyhow::bail!("timed out waiting for event to be processed");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let mut tx = pool.begin().await?;
    let all_msgs = mq.get_all_messages(&mut tx).await?;
    tx.commit().await?;

    let maintenance_jobs: Vec<_> = all_msgs
        .iter()
        .filter(|j| {
            j.name == "MaintainPopulation"
                || j.name == "ChargeOptimizationBudget"
                || j.name == "BreedGenotypes"
        })
        .collect();

    assert!(
        maintenance_jobs.is_empty(),
        "Expected no optimization-related messages, got: {:?}",
        maintenance_jobs
    );

    Ok(())
}
