use chrono::{DateTime, Utc};
use sqlx::{PgExecutor, PgPool, PgTransaction};

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
            insert_gene_bounds(&self.pool, db_morphology.type_hash, &db_gene_bounds).await?;

        Ok(Morphology::from((db_morphology, db_gene_bounds)))
    }

    pub async fn get_morphology(&self, type_hash: i32) -> Result<Morphology, Error> {
        get_morphology(&self.pool, type_hash).await
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
            insert_gene_bounds(&mut *self.tx, db_morphology.type_hash, &db_gene_bounds).await?;

        Ok(Morphology::from((db_morphology, db_gene_bounds)))
    }

    pub async fn get_morphology(&mut self, type_hash: i32) -> Result<Morphology, Error> {
        get_morphology(&mut *self.tx, type_hash).await
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct GeneBounds {
    lower: i32,
    upper: i32,
    divisor: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum GeneBoundError {
    #[error(
        "InvalidBounds: lower bound must be smaller than upper. lower = {lower}, upper={upper}"
    )]
    InvalidBound { lower: i32, upper: i32 },
    #[error("DivisorOverflow: divisor is too large. divisor={divisor}, max={max}")]
    DivisorOverflow { divisor: u32, max: i32 },
}

impl GeneBoundError {
    fn divisor_overflow(divisor: u32) -> Self {
        Self::DivisorOverflow {
            divisor,
            max: i32::MAX,
        }
    }

    fn invalid_bound(lower: i32, upper: i32) -> Self {
        Self::InvalidBound { lower, upper }
    }
}

impl GeneBounds {
    pub fn new(lower: i32, upper: i32, divisor: u32) -> Result<Self, GeneBoundError> {
        if lower > upper {
            return Err(GeneBoundError::invalid_bound(lower, upper));
        };

        let divisor =
            i32::try_from(divisor).map_err(|_| GeneBoundError::divisor_overflow(divisor))?;

        Ok(Self {
            lower,
            upper,
            divisor,
        })
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(Clone, PartialEq))]
pub struct Morphology {
    revised_at: DateTime<Utc>,
    type_name: String,
    type_hash: i32,
    gene_bounds: Vec<GeneBounds>,
}

impl Morphology {
    pub fn new(type_name: &str, type_hash: i32, gene_bounds: Vec<GeneBounds>) -> Self {
        Self {
            revised_at: Utc::now(),
            type_name: type_name.to_string(),
            type_hash,
            gene_bounds,
        }
    }
}

#[derive(Debug, sqlx::FromRow)]
struct DbGeneBounds {
    morphology_id: i32,
    position: i32,
    lower: i32,
    upper: i32,
    divisor: i32,
}

#[derive(Debug)]
struct DbMorphology {
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
                    morphology_id: morphology.type_hash,
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
            revised_at: db_morphology.revised_at,
            type_name: db_morphology.type_name.clone(),
            type_hash: db_morphology.type_hash,
            gene_bounds: db_gene_bounds
                .iter()
                .map(|b| GeneBounds {
                    lower: b.lower,
                    upper: b.upper,
                    divisor: b.divisor,
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
                revised_at,
                type_name,
                type_hash
            )
            VALUES ($1, $2, $3)
            RETURNING
                revised_at,
                type_name,
                type_hash;
            "#,
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
    morphology_id: i32,
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
                    divisor: b.divisor,
                })
            })
            .collect();

        Ok(Morphology {
            revised_at: first.revised_at,
            type_name: first.type_name.clone(),
            type_hash: first.type_hash,
            gene_bounds: gene_bounds?,
        })
    }
}

async fn get_morphology<'tx, E: PgExecutor<'tx>>(
    tx: E,
    type_hash: i32,
) -> Result<Morphology, Error> {
    let rows = sqlx::query_as!(
        MorphologyRow,
        r#"
            SELECT
                m.revised_at,
                m.type_name,
                m.type_hash,
                b.position,
                b.lower,
                b.upper,
                b.divisor
            FROM fx_durable_ga.morphologies m
            JOIN fx_durable_ga.gene_bounds b ON m.type_hash = b.morphology_id
            WHERE m.type_hash = $1
            ORDER BY position ASC;
        "#,
        type_hash
    )
    .fetch_all(tx)
    .await?;

    Morphology::try_from(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::SubsecRound;

    #[test]
    fn it_errors_on_position_overflow() {
        todo!()
    }

    #[test]
    fn it_errors_on_divisor_overflow() {
        todo!()
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_inserts_a_new_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_clone = morphology.clone();

        let inserted = repository.new_morphology(morphology).await?;

        assert_eq!(
            morphology_clone.revised_at.trunc_subsecs(6),
            inserted.revised_at
        );
        assert_eq!(morphology_clone.type_name, inserted.type_name);
        assert_eq!(morphology_clone.type_hash, inserted.type_hash);
        assert_eq!(morphology_clone.gene_bounds, inserted.gene_bounds);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_conflict(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_clone = morphology.clone();

        let _ = repository.new_morphology(morphology).await?;
        let inserted = repository.new_morphology(morphology_clone).await;

        assert!(inserted.is_err());

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_gets_an_existing_morphology(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_type_hash = morphology.type_hash;

        let _ = repository.new_morphology(morphology).await?;
        let selected = repository.get_morphology(morphology_type_hash).await?;

        assert_eq!(morphology_type_hash, selected.type_hash);

        Ok(())
    }

    #[sqlx::test(migrations = "./migrations")]
    async fn it_errors_on_not_found(pool: sqlx::PgPool) -> anyhow::Result<()> {
        let repository = Repository::new(pool);

        let morphology = Morphology::new(
            "test",
            1,
            vec![
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
                GeneBounds::new(1, 10, 10)?,
            ],
        );
        let morphology_type_hash = morphology.type_hash;

        let selected = repository.get_morphology(morphology_type_hash).await;

        assert!(selected.is_err());

        Ok(())
    }
}
