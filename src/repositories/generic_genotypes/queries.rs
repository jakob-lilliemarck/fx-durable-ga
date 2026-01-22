
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        Crossover, Distribution, FitnessGoal, Mutagen, Request, Schedule, Selector,
    };
    use crate::repositories::requests::queries::new_request;
    use serde::{Deserialize, Serialize};
    use std::hash::Hash;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
    struct TestGenome {
        param_a: i32,
        param_b: String,
    }

    /// Creates and inserts a dummy request to satisfy foreign key constraints.
    async fn create_test_request(pool: &sqlx::PgPool) -> Request {
        let request = Request::new(
            "test_request",
            1,
            FitnessGoal::maximize(0.9).unwrap(),
            Selector::tournament(10, 20).expect("is valid"),
            Schedule::generational(100, 10),
            Mutagen::constant(0.5, 0.1).unwrap(),
            Crossover::uniform(0.5).unwrap(),
            Distribution::latin_hypercube(200),
            None::<()>,
        )
        .unwrap();
        new_request(pool, request.clone()).await.unwrap();
        request
    }

    #[sqlx::test(migrations = false)]
    async fn it_writes_and_reads_a_generic_genotype_roundtrip(
        pool: sqlx::PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_default_migrations(&pool).await?;

        // --- ARRANGE ---
        let request = create_test_request(&pool).await;

        let concrete_genome = TestGenome {
            param_a: 42,
            param_b: "hello".to_string(),
        };

        let genotype_to_insert = GenericGenotype::new(
            "TestGenome",
            123,
            &concrete_genome,
            request.id,
            1,
        )?;

        // --- ACT ---
        new_genotypes(&pool, vec![genotype_to_insert.clone()]).await?;
        let retrieved_genotype = get_genotype(&pool, &genotype_to_insert.id).await?;
        let retrieved_id = retrieved_genotype.id;
        let retrieved_hash = retrieved_genotype.genome_hash;
        let deserialized_genome: TestGenome = retrieved_genotype.deserialize()?;

        // --- ASSERT ---
        assert_eq!(concrete_genome, deserialized_genome);
        assert_eq!(genotype_to_insert.id, retrieved_id);
        assert_eq!(genotype_to_insert.genome_hash, retrieved_hash);

        Ok(())
    }
}

use crate::models::GenericGenotype;
use serde_json::Value;
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx), fields(genotype_id = %id))]
pub(crate) async fn get_genotype<'tx, E>(tx: E, id: &Uuid) -> Result<GenericGenotype, super::Error>
where
    E: PgExecutor<'tx>,
{
    let row = sqlx::query!(
        r#"
            SELECT
                id,
                generated_at,
                type_name,
                type_hash,
                genome_data,
                genome_hash,
                request_id,
                generation_id
            FROM fx_durable_ga.generic_genotypes
            WHERE id = $1;
        "#,
        id
    )
    .fetch_one(tx)
    .await?;

    Ok(GenericGenotype {
        id: row.id,
        generated_at: row.generated_at,
        type_name: row.type_name,
        type_hash: row.type_hash,
        genome_data: row.genome_data.unwrap_or(Value::Null),
        genome_hash: row.genome_hash,
        request_id: row.request_id,
        generation_id: row.generation_id,
    })
}

#[instrument(level = "debug", skip(tx, genotypes), fields(genotypes_count = genotypes.len()))]
pub(crate) async fn new_genotypes<'tx, E>(
    tx: E,
    genotypes: Vec<GenericGenotype>,
) -> Result<Vec<GenericGenotype>, super::Error>
where
    E: PgExecutor<'tx>,
{
    if genotypes.is_empty() {
        return Ok(Vec::new());
    }

    let mut ids = Vec::new();
    let mut generated_ats = Vec::new();
    let mut type_names = Vec::new();
    let mut type_hashes = Vec::new();
    let mut genome_datas = Vec::new();
    let mut genome_hashes = Vec::new();
    let mut request_ids = Vec::new();
    let mut generation_ids = Vec::new();

    for g in genotypes {
        ids.push(g.id);
        generated_ats.push(g.generated_at);
        type_names.push(g.type_name);
        type_hashes.push(g.type_hash);
        genome_datas.push(g.genome_data);
        genome_hashes.push(g.genome_hash);
        request_ids.push(g.request_id);
        generation_ids.push(g.generation_id);
    }

    let inserted_rows = sqlx::query!(
        r#"
            INSERT INTO fx_durable_ga.generic_genotypes (
                id,
                generated_at,
                type_name,
                type_hash,
                genome_data,
                genome_hash,
                request_id,
                generation_id
            )
            SELECT * FROM UNNEST(
                $1::uuid[],
                $2::timestamptz[],
                $3::text[],
                $4::int[],
                $5::jsonb[],
                $6::bigint[],
                $7::uuid[],
                $8::int[]
            )
            RETURNING
                id,
                generated_at,
                type_name,
                type_hash,
                genome_data,
                genome_hash,
                request_id,
                generation_id
        "#,
        &ids,
        &generated_ats,
        &type_names,
        &type_hashes,
        &genome_datas,
        &genome_hashes,
        &request_ids,
        &generation_ids
    )
    .fetch_all(tx)
    .await?;

    let result_genotypes = inserted_rows
        .into_iter()
        .map(|row| GenericGenotype {
            id: row.id,
            generated_at: row.generated_at,
            type_name: row.type_name,
            type_hash: row.type_hash,
            genome_data: row.genome_data.unwrap_or(Value::Null),
            genome_hash: row.genome_hash,
            request_id: row.request_id,
            generation_id: row.generation_id,
        })
        .collect();

    Ok(result_genotypes)
}
