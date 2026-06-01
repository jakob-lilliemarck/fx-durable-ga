use super::test_tools::build_service;
use chrono::Utc;
use fx_event_bus::test_tools::get_unacknowledged_events;
use sqlx::PgPool;
use uuid::Uuid;

#[sqlx::test(migrations = false)]
async fn it_appends_a_transaction(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;

    let svc = build_service(pool).await?;

    let account_id = Uuid::parse_str("00000000-0000-0000-0000-000000000001")?;
    let transaction = svc
        .append(account_id, "test".to_string(), 100, "first".to_string())
        .await?;

    assert_eq!(transaction.account_id, account_id);
    assert_eq!(transaction.amount, 100);
    assert_eq!(transaction.balance, 100);
    assert_eq!(transaction.account_type, "test");
    assert_eq!(transaction.reason, "first");
    assert!(transaction.timestamp <= Utc::now());

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_appends_multiple_and_updates_balance(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;

    let svc = build_service(pool).await?;

    let account_id = Uuid::parse_str("00000000-0000-0000-0000-000000000001")?;

    let t1 = svc
        .append(account_id, "test".to_string(), 100, "first".to_string())
        .await?;
    assert_eq!(t1.balance, 100);

    let t2 = svc
        .append(account_id, "test".to_string(), 50, "second".to_string())
        .await?;
    assert_eq!(t2.balance, 150);

    let t3 = svc
        .append(account_id, "test".to_string(), 25, "third".to_string())
        .await?;
    assert_eq!(t3.balance, 175);

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_handles_charges(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;

    let svc = build_service(pool).await?;

    let account_id = Uuid::parse_str("00000000-0000-0000-0000-000000000001")?;

    let t1 = svc
        .append(account_id, "test".to_string(), 100, "credit".to_string())
        .await?;
    assert_eq!(t1.balance, 100);

    let t2 = svc
        .append(account_id, "test".to_string(), -30, "charge".to_string())
        .await?;
    assert_eq!(t2.balance, 70);

    Ok(())
}

#[sqlx::test(migrations = false)]
async fn it_publishes_a_transaction_created_event(pool: PgPool) -> anyhow::Result<()> {
    crate::migrations::run_default_migrations(&pool).await?;

    let svc = build_service(pool.clone()).await?;

    let account_id = Uuid::parse_str("00000000-0000-0000-0000-000000000001")?;

    let before = get_unacknowledged_events(&pool).await?;

    svc.append(account_id, "test".to_string(), 100, "first".to_string())
        .await?;

    let after = get_unacknowledged_events(&pool).await?;

    assert_eq!(after - before, 1);

    Ok(())
}
