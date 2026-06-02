use super::super::Error as RepositoryError;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug)]
pub struct GetAggregatedFitnessFilter {
    cursor: Option<i64>,
    percentiles: Option<Vec<f64>>,
}

impl Default for GetAggregatedFitnessFilter {
    fn default() -> Self {
        Self {
            cursor: None,
            percentiles: None,
        }
    }
}

impl GetAggregatedFitnessFilter {
    pub fn with_cursor(mut self, cursor: i64) -> Self {
        self.cursor = Some(cursor);
        self
    }

    pub fn with_percentiles(mut self, percentiles: Vec<f64>) -> Self {
        self.percentiles = Some(percentiles);
        self
    }
}

#[derive(Debug)]
pub struct AggregatedFitness {
    pub bin_index: i64,
    pub count: i64,
    pub min_fitness: f64,
    pub max_fitness: f64,
    pub avg_fitness: f64,
    pub sample_variance: Option<f64>,
    pub sample_std: Option<f64>,
    pub percentiles: Option<Vec<f64>>,
}

#[instrument(level = "debug", skip(tx), fields(request_id = %request_id, bin_size, limit, filter = ?filter))]
pub async fn get_aggregated_fitness<'tx, E: PgExecutor<'tx>>(
    tx: E,
    request_id: &Uuid,
    bin_size: i64,
    limit: i64,
    filter: &GetAggregatedFitnessFilter,
) -> Result<Vec<AggregatedFitness>, RepositoryError> {
    let stats = sqlx::query_as!(
        AggregatedFitness,
        r#"
        WITH ordered AS (
            SELECT
                id,
                generated_at,
                fitness,
                ROW_NUMBER() OVER (ORDER BY generated_at ASC, id ASC) AS rn
            FROM evaluation.evaluations
            WHERE request_id = $1
        ),
        binned AS (
            SELECT
                ((rn - 1) / $2)::bigint AS bin_index,
                fitness
            FROM ordered
        )
        SELECT
            bin_index "bin_index!",
            COUNT(*) AS "count!",
            MIN(fitness) AS "min_fitness!",
            MAX(fitness) AS "max_fitness!",
            AVG(fitness) AS "avg_fitness!",
            VAR_SAMP(fitness) AS "sample_variance?",
            STDDEV_SAMP(fitness) AS "sample_std?",
            CASE
                WHEN $4::double precision[] IS NULL THEN NULL
                ELSE percentile_cont($4::double precision[])
                    WITHIN GROUP (ORDER BY fitness)
            END AS "percentiles?:Vec<f64>"
        FROM binned
        WHERE ($3::bigint IS NULL OR bin_index > $3::bigint)
        GROUP BY bin_index
        ORDER BY bin_index ASC
        LIMIT $5;
        "#,
        request_id,
        bin_size,
        filter.cursor,
        filter.percentiles.as_deref(),
        limit
    )
    .fetch_all(tx)
    .await?;

    Ok(stats)
}

#[cfg(test)]
mod tests_get_aggregated_fitness {
    use super::{GetAggregatedFitnessFilter, get_aggregated_fitness};
    use crate::repositories::genotypes::Genotype;
    use crate::repositories::genotypes::store_genotypes;
    use crate::services::evaluation::repositories::evaluations::Evaluation;
    use crate::services::evaluation::repositories::evaluations::queries::store_evaluations;
    use crate::services::optimization::store_request;
    use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
    use chrono::{Duration as ChronoDuration, TimeZone, Utc};
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_aggregates_bins_without_percentiles(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let request_id = seed(&pool).await?;

        let stats = get_aggregated_fitness(
            &pool,
            &request_id,
            4,
            10,
            &GetAggregatedFitnessFilter::default(),
        )
        .await?;

        assert!(stats.len() >= 3);
        let first = &stats[0];
        assert_eq!(first.bin_index, 0);
        assert_eq!(first.count, 4);
        assert!((first.min_fitness - 0.10).abs() < 1e-12);
        assert!((first.max_fitness - 0.13).abs() < 1e-12);
        assert!((first.avg_fitness - 0.115).abs() < 1e-12);
        let expected_variance = 0.0005_f64 / 3.0;
        let expected_std = expected_variance.sqrt();
        assert!((first.sample_variance.unwrap() - expected_variance).abs() < 1e-12);
        assert!((first.sample_std.unwrap() - expected_std).abs() < 1e-12);
        assert!(first.percentiles.is_none());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_includes_percentiles_when_requested(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let request_id = seed(&pool).await?;

        let stats = get_aggregated_fitness(
            &pool,
            &request_id,
            4,
            10,
            &GetAggregatedFitnessFilter::default().with_percentiles(vec![0.5, 0.9]),
        )
        .await?;

        let first = &stats[0];
        let percentiles = first.percentiles.as_ref().expect("percentiles");
        assert_eq!(percentiles.len(), 2);
        assert!((percentiles[0] - 0.115).abs() < 1e-12);
        assert!((percentiles[1] - 0.127).abs() < 1e-12);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_partial_last_bin(pool: sqlx::PgPool) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;
        let request_id = seed(&pool).await?;

        let stats = get_aggregated_fitness(
            &pool,
            &request_id,
            5,
            10,
            &GetAggregatedFitnessFilter::default().with_cursor(3),
        )
        .await?;

        assert_eq!(stats.len(), 1);
        let last = &stats[0];
        assert_eq!(last.bin_index, 4);
        assert_eq!(last.count, 2);
        assert!((last.min_fitness - 0.48).abs() < 1e-12);
        assert!((last.max_fitness - 0.49).abs() < 1e-12);
        assert!((last.avg_fitness - 0.485).abs() < 1e-12);
        let expected_variance = 0.00005_f64;
        let expected_std = expected_variance.sqrt();
        assert!((last.sample_variance.unwrap() - expected_variance).abs() < 1e-12);
        assert!((last.sample_std.unwrap() - expected_std).abs() < 1e-12);

        Ok(())
    }

    async fn seed(pool: &sqlx::PgPool) -> anyhow::Result<Uuid> {
        let request = Request::new(
            "test",
            FitnessGoal::maximize(0.9)?,
            Selector::tournament(10),
            Schedule::generational(100, 10),
        );
        let request = store_request(pool, request).await?;

        let base = Utc
            .with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
            .single()
            .expect("valid timestamp");

        let batch_fitness = [
            0.10, 0.11, 0.12, 0.13, 0.20, 0.21, 0.22, 0.23, 0.30, 0.31, 0.32, 0.33,
        ];
        let rolling_fitness = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49];

        let batch_times = [
            base,
            base + ChronoDuration::minutes(1),
            base + ChronoDuration::minutes(2),
        ];

        let mut genotypes = Vec::new();
        for (generation_idx, generated_at) in batch_times.iter().enumerate() {
            for _ in 0..4 {
                let mut genotype = Genotype::new(
                    "test",
                    serde_json::json!([1, 2, 3]),
                    Some(request.id),
                    Some((generation_idx + 1) as i32),
                    None,
                    None,
                )?;
                genotype.generated_at = *generated_at;
                genotypes.push(genotype);
            }
        }

        for (idx, _fitness) in rolling_fitness.iter().enumerate() {
            let generated_at =
                base + ChronoDuration::minutes(3) + ChronoDuration::seconds(idx as i64);
            let mut genotype = Genotype::new(
                "test",
                serde_json::json!([4, 5, 6]),
                Some(request.id),
                Some((batch_times.len() as i32) + 1 + idx as i32),
                None,
                None,
            )?;
            genotype.generated_at = generated_at;
            genotypes.push(genotype);
        }

        store_genotypes(pool, &genotypes).await?;

        let all_fitness: Vec<f64> = batch_fitness
            .iter()
            .chain(rolling_fitness.iter())
            .copied()
            .collect();

        let mut evaluations = Vec::new();
        for (genotype, fitness) in genotypes.iter().zip(all_fitness.iter()) {
            evaluations.push(
                Evaluation::new(
                    genotype.id,
                    *fitness,
                    Some(Utc::now()),
                    Some(Utc::now()),
                    None,
                )
                .with_request_id(request.id)
                .with_generated_at(genotype.generated_at),
            );
        }

        store_evaluations(pool, &evaluations).await?;

        Ok(request.id)
    }
}
