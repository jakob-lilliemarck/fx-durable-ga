use chrono::{DateTime, Utc};
use sqlx::PgExecutor;
use tracing::instrument;
use uuid::Uuid;

#[instrument(level = "debug", skip(tx))]
pub async fn append<'tx, E: PgExecutor<'tx>>(
    tx: E,
    account_id: &Uuid,
    account_type: &str,
    amount: i64,
    reason: &str,
) -> Result<Transaction, super::Error> {
    let now = Utc::now();
    let transaction_id = Uuid::now_v7();
    let transaction = sqlx::query_as!(
        Transaction,
        r#"
        INSERT INTO budgeting.transactions (
            id,
            account_id,
            account_type,
            balance,
            amount,
            timestamp,
            reason
        )
        VALUES (
            $1,
            $2,
            $3,
            COALESCE(
                (
                    SELECT balance
                    FROM budgeting.transactions
                    WHERE account_id = $2
                    ORDER BY timestamp DESC
                    LIMIT 1
                ),
                0
            ) + $4,
            $4,
            $5,
            $6
        )
        RETURNING
            id,
            account_id,
            account_type,
            balance,
            amount,
            timestamp,
            reason;
        "#,
        transaction_id, // $1
        account_id,     // $2
        account_type,   // $3
        amount,         // $4
        now,            // $5
        reason,         // $6
    )
    .fetch_one(tx)
    .await?;
    Ok(transaction)
}

#[cfg(test)]
mod tests_append {
    use sqlx::PgPool;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_appends_to_an_account(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_migrations(&pool).await?;

        let account_id = Uuid::now_v7();

        let transaction = super::append(&pool, &account_id, "test_type", 100, "reason").await?;
        assert_eq!(transaction.balance, 100);
        let balance = super::balance(&pool, &account_id).await?;
        assert_eq!(balance, 100);

        let transaction = super::append(&pool, &account_id, "test_type", 5, "reason").await?;
        assert_eq!(transaction.balance, 105);
        let balance = super::balance(&pool, &account_id).await?;
        assert_eq!(balance, 105);

        let transaction = super::append(&pool, &account_id, "test_type", -10, "reason").await?;
        assert_eq!(transaction.balance, 95);
        let balance = super::balance(&pool, &account_id).await?;
        assert_eq!(balance, 95);

        Ok(())
    }
}

#[instrument(level = "debug", skip(tx))]
pub async fn balance<'tx, E: PgExecutor<'tx>>(
    tx: E,
    account_id: &Uuid,
) -> Result<i64, super::Error> {
    let balance = match sqlx::query_scalar!(
        r#"
        SELECT balance
        FROM budgeting.transactions
        WHERE account_id = $1
        ORDER BY timestamp DESC
        LIMIT 1
        "#,
        account_id
    )
    .fetch_one(tx)
    .await
    {
        Ok(balance) => Ok(balance),
        Err(sqlx::Error::RowNotFound) => Ok(0),
        Err(err) => Err(err),
    }?;

    Ok(balance)
}

#[cfg(test)]
mod tests_balance {
    use sqlx::PgPool;
    use uuid::Uuid;

    #[sqlx::test(migrations = false)]
    async fn it_gets_balance(pool: PgPool) -> anyhow::Result<()> {
        crate::migrations::run_migrations(&pool).await?;

        let account_id = Uuid::now_v7();
        let transaction = super::append(&pool, &account_id, "test_type", 100, "reason").await?;
        assert_eq!(transaction.balance, 100);
        let balance = super::balance(&pool, &account_id).await?;
        assert_eq!(balance, 100);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_gets_balance_of_an_account_without_transactions(
        pool: PgPool,
    ) -> anyhow::Result<()> {
        crate::migrations::run_migrations(&pool).await?;

        let account_id = Uuid::now_v7();

        let balance = super::balance(&pool, &account_id).await?;
        assert_eq!(balance, 0);

        Ok(())
    }
}

/// A single transaction entry for an account.
#[derive(Clone)]
pub struct Transaction {
    pub(crate) id: Uuid,
    pub(crate) amount: i64,
    pub(crate) balance: i64,
    pub(crate) account_id: Uuid,
    pub(crate) account_type: String,
    pub(crate) timestamp: DateTime<Utc>,
    pub(crate) reason: String,
}

impl Transaction {
    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn amount(&self) -> i64 {
        self.amount
    }

    pub fn balance(&self) -> i64 {
        self.balance
    }

    pub fn account_id(&self) -> Uuid {
        self.account_id
    }

    pub fn account_type(&self) -> &str {
        &self.account_type
    }

    pub fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// Filter criteria for querying transaction history.
#[derive(Debug)]
pub struct Filter {
    cursor: Option<(DateTime<Utc>, Uuid)>,
    reason: Option<String>,
    account_id: Option<Uuid>,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            cursor: None,
            reason: None,
            account_id: None,
        }
    }
}

impl Filter {
    pub fn with_account_id(mut self, account_id: Uuid) -> Self {
        self.account_id = Some(account_id);
        self
    }
    pub fn with_cursor(mut self, cursor: (DateTime<Utc>, Uuid)) -> Self {
        self.cursor = Some(cursor);
        self
    }

    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = Some(reason.to_string());
        self
    }
}

#[instrument(level = "debug", skip(tx))]
pub async fn history<'tx, E: PgExecutor<'tx>>(
    tx: E,
    filter: &Filter,
    limit: i64,
) -> Result<Vec<super::Transaction>, super::Error> {
    let transactions = sqlx::query_as!(
        Transaction,
        r#"
        SELECT
            id,
            account_id,
            account_type,
            amount,
            balance,
            timestamp,
            reason
        FROM budgeting.transactions
        WHERE
            ($1::UUID IS NULL OR account_id = $1::UUID)
            AND (
                $2::TIMESTAMPTZ IS NULL
                OR $3::UUID IS NULL
                OR (timestamp, id) < ($2, $3::UUID)
            )
            ORDER BY timestamp DESC, id DESC
        LIMIT $4;
        "#,
        filter.account_id.as_ref(),
        filter.cursor.as_ref().map(|(ts, _)| ts),
        filter.cursor.as_ref().map(|(_, id)| id),
        limit
    )
    .fetch_all(tx)
    .await?;

    Ok(transactions)
}

#[cfg(test)]
mod tests_history {
    use super::{Filter, append, history};
    use sqlx::PgPool;
    use uuid::Uuid;

    struct TestData {
        account_id: Uuid,
    }

    async fn seed(pool: &PgPool) -> anyhow::Result<TestData> {
        crate::migrations::run_migrations(pool).await?;

        let account_id = Uuid::parse_str("00000000-0000-0000-0000-000000000001")?;

        append(pool, &account_id, "test", 100, "first").await?;
        append(pool, &account_id, "test", 50, "second").await?;
        append(pool, &account_id, "test", 25, "third").await?;

        Ok(TestData {
            account_id,
        })
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_history_for_an_account(pool: PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let filter = Filter::default().with_account_id(data.account_id);
        let results = history(&pool, &filter, 10).await?;

        assert_eq!(results.len(), 3);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_limits_results(pool: PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let filter = Filter::default().with_account_id(data.account_id);
        let results = history(&pool, &filter, 2).await?;

        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_returns_empty_for_unknown_account(pool: PgPool) -> anyhow::Result<()> {
        seed(&pool).await?;

        let unknown_id = Uuid::parse_str("00000000-0000-0000-0000-000000000099")?;
        let filter = Filter::default().with_account_id(unknown_id);
        let results = history(&pool, &filter, 10).await?;

        assert!(results.is_empty());

        Ok(())
    }

    #[sqlx::test(migrations = false)]
    async fn it_paginates_with_cursor(pool: PgPool) -> anyhow::Result<()> {
        let data = seed(&pool).await?;

        let filter = Filter::default().with_account_id(data.account_id);
        let first_page = history(&pool, &filter, 2).await?;

        assert_eq!(first_page.len(), 2);

        let last = first_page.last().unwrap();
        let cursor = (last.timestamp, last.id);
        let cursor_filter = Filter::default()
            .with_account_id(data.account_id)
            .with_cursor(cursor);
        let second_page = history(&pool, &cursor_filter, 10).await?;

        assert_eq!(second_page.len(), 1);

        Ok(())
    }
}
