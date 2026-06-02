use super::super::models::TimingsSummary;
use super::super::Error as RepositoryError;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Default)]
pub struct GetEvaluationTimingsFilter {
    genotype_id: Option<Uuid>,
    request_id: Option<Uuid>,
    evaluated_by: Option<Uuid>,
}

impl GetEvaluationTimingsFilter {
    pub fn with_genotype_id(mut self, genotype_id: Uuid) -> Self {
        self.genotype_id = Some(genotype_id);
        self
    }

    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_evaluated_by(mut self, evaluated_by: Uuid) -> Self {
        self.evaluated_by = Some(evaluated_by);
        self
    }
}

struct DbTimingsSummary {
    total: i64,
    min_duration_micros: Option<i64>,
    max_duration_micros: Option<i64>,
    avg_duration_micros: Option<i64>,
    percentile_durations: Option<Vec<i64>>,
}

impl From<DbTimingsSummary> for TimingsSummary {
    fn from(value: DbTimingsSummary) -> Self {
        TimingsSummary {
            records: value.total,
            min: value.min_duration_micros
                .map_or(std::time::Duration::ZERO, |m| std::time::Duration::from_micros(m as u64)),
            max: value.max_duration_micros
                .map_or(std::time::Duration::ZERO, |m| std::time::Duration::from_micros(m as u64)),
            avg: value.avg_duration_micros
                .map_or(std::time::Duration::ZERO, |m| std::time::Duration::from_micros(m as u64)),
            percentiles: value.percentile_durations.map_or(vec![], |values| {
                values
                    .iter()
                    .map(|micros| std::time::Duration::from_micros(*micros as u64))
                    .collect()
            }),
        }
    }
}

#[instrument(level = "debug", skip(tx), fields(filter = ?filter))]
pub async fn get_evaluation_timings<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &GetEvaluationTimingsFilter,
    percentiles: &[f64],
) -> Result<TimingsSummary, RepositoryError> {
    let db_timings = sqlx::query_as!(
        DbTimingsSummary,
        r#"
        SELECT
            COUNT(*) AS "total!",
            MIN(duration_micros) AS "min_duration_micros",
            MAX(duration_micros) AS "max_duration_micros",
            AVG(duration_micros)::bigint AS "avg_duration_micros",
            ARRAY(
                SELECT ROUND(val)::bigint
                FROM UNNEST(
                    percentile_cont($4::double PRECISION[])
                    WITHIN GROUP (ORDER BY duration_micros)
                ) AS val
            ) AS "percentile_durations"
        FROM (
            SELECT
                (EXTRACT(EPOCH FROM completed_at - started_at) * 1e6)::bigint AS duration_micros
            FROM evaluation.evaluations
            WHERE ($1::uuid IS NULL OR genotype_id = $1)
              AND ($2::uuid IS NULL OR request_id = $2)
              AND ($3::uuid IS NULL OR evaluated_by = $3)
        ) timed
        "#,
        filter.genotype_id,
        filter.request_id,
        filter.evaluated_by,
        percentiles
    )
    .fetch_one(tx)
    .await?;

    Ok(db_timings.into())
}

#[cfg(test)]
mod tests_get_evaluation_timings {
    use super::{GetEvaluationTimingsFilter, get_evaluation_timings};
    use crate::services::evaluation::repositories::evaluations::Evaluation;
    use crate::services::evaluation::repositories::evaluations::queries::store_evaluations;
    use crate::repositories::genotypes::Genotype;
    use crate::repositories::genotypes::store_genotypes;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use chrono::{Duration as ChronoDuration, TimeZone, Utc};
    use std::time::Duration;
    use sqlx::PgPool;
    use uuid::Uuid;

    async fn seed(pool: &PgPool) -> anyhow::Result<(Uuid, Vec<Uuid>, Uuid, Uuid)> {
        let request = Request::new(
            "timings",
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request_id = request.id;
        store_request(pool, request).await?;

        let genotypes = vec![
            Genotype::new("timings", serde_json::json!([1]), Some(request_id), Some(1), None, None)?,
            Genotype::new("timings", serde_json::json!([2]), Some(request_id), Some(1), None, None)?,
            Genotype::new("timings", serde_json::json!([3]), Some(request_id), Some(1), None, None)?,
        ];
        let ids: Vec<Uuid> = genotypes.iter().map(|g| g.id).collect();
        store_genotypes(pool, &genotypes).await?;

        let host_a = Uuid::now_v7();
        let host_b = Uuid::now_v7();
        let base = Utc
            .with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
            .single()
            .expect("valid timestamp");
        let durations_micros = [100_000i64, 200_000, 300_000];

        for (idx, genotype_id) in ids.iter().enumerate() {
            let started_at = base + ChronoDuration::seconds(idx as i64);
            let completed_at = started_at + ChronoDuration::microseconds(durations_micros[idx]);
            let host = if idx % 2 == 0 { host_a } else { host_b };
            store_evaluations(
                pool,
                &[Evaluation::new(
                    *genotype_id,
                    0.5,
                    Some(started_at),
                    Some(completed_at),
                    Some(host),
                )
                .with_request_id(request_id)],
            )
            .await?;
        }

        Ok((request_id, ids, host_a, host_b))
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_timing_data_with_percentiles(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (request_id, _, _, _) = seed(&pool).await?;

        let summary = get_evaluation_timings(
            &pool,
            &GetEvaluationTimingsFilter::default().with_request_id(request_id),
            &[0.5, 0.9],
        )
        .await?;

        assert_eq!(summary.records(), 3);
        assert_eq!(summary.min(), &Duration::from_micros(100_000));
        assert_eq!(summary.max(), &Duration::from_micros(300_000));
        assert_eq!(summary.avg(), &Duration::from_micros(200_000));
        assert_eq!(summary.percentiles().len(), 2);
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_timing_data_without_percentiles(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (request_id, _, _, _) = seed(&pool).await?;

        let summary = get_evaluation_timings(
            &pool,
            &GetEvaluationTimingsFilter::default().with_request_id(request_id),
            &[],
        )
        .await?;

        assert_eq!(summary.records(), 3);
        assert!(summary.percentiles().is_empty());
        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_filters_timings_by_evaluated_by(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let (request_id, _, host_a, _host_b) = seed(&pool).await?;

        let all = get_evaluation_timings(
            &pool,
            &GetEvaluationTimingsFilter::default().with_request_id(request_id),
            &[],
        )
        .await?;
        assert_eq!(all.records(), 3);

        let filtered = get_evaluation_timings(
            &pool,
            &GetEvaluationTimingsFilter::default()
                .with_request_id(request_id)
                .with_evaluated_by(host_a),
            &[],
        )
        .await?;
        assert_eq!(filtered.records(), 2);
        assert_eq!(filtered.min(), &Duration::from_micros(100_000));
        assert_eq!(filtered.max(), &Duration::from_micros(300_000));
        assert_eq!(filtered.avg(), &Duration::from_micros(200_000));

        Ok(())
    }
}
