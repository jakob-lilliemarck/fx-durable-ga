use chrono::{DateTime, Utc};
use sqlx::{PgExecutor, PgPool, PgTransaction};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Not found")]
    NotFound,
    #[error("Integer overflow at field \"{0}\"")]
    IntegerOverflow(String),
}

pub struct Repository {
    pool: PgPool,
}

pub struct TxRepository<'tx> {
    tx: PgTransaction<'tx>,
}

impl Repository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn new_morphology(&self, morphology: Morphology) -> Result<Morphology, Error> {
        let (db_morphology, db_gene_bounds): (DbMorphology, Vec<DbGeneBounds>) =
            morphology.try_into()?;

        let db_morphology = insert_morphology(&self.pool, db_morphology).await?;

        let db_gene_bounds =
            insert_gene_bounds(&self.pool, db_morphology.id, &db_gene_bounds).await?;

        Ok(Morphology::from((db_morphology, db_gene_bounds)))
    }

    pub async fn get_morphology(&self, id: Uuid) -> Result<Morphology, Error> {
        get_morphology(&self.pool, id).await
    }
}

impl<'tx> TxRepository<'tx> {
    pub fn new(tx: PgTransaction<'tx>) -> Self {
        Self { tx }
    }

    pub async fn new_morphology(&mut self, morphology: Morphology) -> Result<Morphology, Error> {
        let (db_morphology, db_gene_bounds): (DbMorphology, Vec<DbGeneBounds>) =
            morphology.try_into()?;

        let db_morphology = insert_morphology(&mut *self.tx, db_morphology).await?;

        let db_gene_bounds =
            insert_gene_bounds(&mut *self.tx, db_morphology.id, &db_gene_bounds).await?;

        Ok(Morphology::from((db_morphology, db_gene_bounds)))
    }

    pub async fn get_morphology(&mut self, id: Uuid) -> Result<Morphology, Error> {
        get_morphology(&mut *self.tx, id).await
    }
}

#[derive(Debug)]
pub struct GeneBounds {
    lower: i32,
    upper: i32,
    divisor: u32,
}

#[derive(Debug)]
pub struct Morphology {
    id: Uuid,
    revised_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
    gene_bounds: Vec<GeneBounds>,
}

#[derive(Debug, sqlx::FromRow)]
struct DbGeneBounds {
    morphology_id: Uuid,
    position: i32,
    lower: i32,
    upper: i32,
    divisor: i32,
}

#[derive(Debug)]
struct DbMorphology {
    id: Uuid,
    revised_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
}

impl TryFrom<Morphology> for (DbMorphology, Vec<DbGeneBounds>) {
    type Error = Error;

    fn try_from(morphology: Morphology) -> Result<Self, Self::Error> {
        let gene_bounds: Result<Vec<DbGeneBounds>, Error> = morphology
            .gene_bounds
            .iter()
            .enumerate()
            .map(|(i, b)| {
                Ok(DbGeneBounds {
                    morphology_id: morphology.id,
                    position: i32::try_from(i)
                        .map_err(|_| Error::IntegerOverflow("position".to_string()))?,
                    lower: b.lower,
                    upper: b.upper,
                    divisor: i32::try_from(b.divisor)
                        .map_err(|_| Error::IntegerOverflow("divisor".to_string()))?,
                })
            })
            .collect();

        let morphology = DbMorphology {
            id: morphology.id,
            revised_at: morphology.revised_at,
            type_name: morphology.type_name,
            type_hash: morphology.type_hash,
        };

        Ok((morphology, gene_bounds?))
    }
}

impl From<(DbMorphology, Vec<DbGeneBounds>)> for Morphology {
    fn from((db_morphology, db_gene_bounds): (DbMorphology, Vec<DbGeneBounds>)) -> Self {
        Morphology {
            id: db_morphology.id,
            revised_at: db_morphology.revised_at,
            type_name: db_morphology.type_name.clone(),
            type_hash: db_morphology.type_hash,
            gene_bounds: db_gene_bounds
                .iter()
                .map(|b| GeneBounds {
                    lower: b.lower,
                    upper: b.upper,
                    divisor: b.divisor as u32,
                })
                .collect::<Vec<GeneBounds>>(),
        }
    }
}

async fn insert_morphology<'tx, E: PgExecutor<'tx>>(
    tx: E,
    morphology: DbMorphology,
) -> Result<DbMorphology, Error> {
    let db_morphology = sqlx::query_as!(
        DbMorphology,
        r#"
            INSERT INTO fx_durable_ga.morphologies (
                id,
                revised_at,
                type_name,
                type_hash
            )
            VALUES ($1, $2, $3, $4)
            RETURNING
                id,
                revised_at,
                type_name,
                type_hash;
            "#,
        morphology.id,
        morphology.revised_at,
        morphology.type_name,
        morphology.type_hash
    )
    .fetch_one(tx)
    .await?;

    Ok(db_morphology)
}

async fn insert_gene_bounds<'tx, E: PgExecutor<'tx>>(
    tx: E,
    morphology_id: Uuid,
    gene_bounds: &[DbGeneBounds],
) -> Result<Vec<DbGeneBounds>, Error> {
    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO fx_durable_ga.gene_bounds (
            morphology_id, position, lower, upper, divisor
        ) VALUES ",
    );

    let mut first = true;
    // Stream serialization: serialize each event as needed rather than
    // pre-allocating all payloads, reducing peak memory usage
    for (i, b) in gene_bounds.iter().enumerate() {
        if first {
            first = false;
        } else {
            query_builder.push(", ");
        }

        let position =
            i32::try_from(i).map_err(|_| Error::IntegerOverflow("position".to_string()))?;

        query_builder
            .push("(")
            .push_bind(morphology_id)
            .push(", ")
            .push_bind(position)
            .push(", ")
            .push_bind(b.lower)
            .push(", ")
            .push_bind(b.upper)
            .push(", ")
            .push_bind(b.divisor)
            .push(")");
    }

    // Add RETURNING clause
    query_builder.push(" RETURNING morphology_id, position, lower, upper, divisor");

    let bounds = query_builder
        .build_query_as::<DbGeneBounds>()
        .fetch_all(tx)
        .await?;

    Ok(bounds)
}

struct MorphologyRow {
    morphology_id: Uuid,
    revised_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
    position: i32,
    lower: i32,
    upper: i32,
    divisor: i32,
}

impl TryFrom<Vec<MorphologyRow>> for Morphology {
    type Error = Error;

    fn try_from(mut rows: Vec<MorphologyRow>) -> Result<Self, Self::Error> {
        rows.sort_by_key(|b| b.position);

        let first = rows.first().ok_or(Error::NotFound)?;

        let gene_bounds: Result<Vec<GeneBounds>, Error> = rows
            .iter()
            .map(|b| {
                Ok(GeneBounds {
                    lower: b.lower,
                    upper: b.upper,
                    divisor: u32::try_from(b.divisor)
                        .map_err(|_| Error::IntegerOverflow("divisor".to_string()))?,
                })
            })
            .collect();

        Ok(Morphology {
            id: first.morphology_id,
            revised_at: first.revised_at,
            type_name: first.type_name.clone(),
            type_hash: first.type_hash,
            gene_bounds: gene_bounds?,
        })
    }
}

async fn get_morphology<'tx, E: PgExecutor<'tx>>(tx: E, id: Uuid) -> Result<Morphology, Error> {
    let rows = sqlx::query_as!(
        MorphologyRow,
        r#"
            SELECT
                m.id AS morphology_id,
                m.revised_at,
                m.type_name,
                m.type_hash,
                b.position,
                b.lower,
                b.upper,
                b.divisor
            FROM fx_durable_ga.morphologies m
            JOIN fx_durable_ga.gene_bounds b ON m.id = b.morphology_id
            WHERE m.id = $1
            ORDER BY position ASC;
        "#,
        id
    )
    .fetch_all(tx)
    .await?;

    Morphology::try_from(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        todo!()
    }
}
