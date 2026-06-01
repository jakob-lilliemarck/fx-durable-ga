use crate::services::budgeting;
use sqlx::PgPool;
use std::sync::Arc;

pub(crate) async fn build_service(pool: PgPool) -> anyhow::Result<Arc<budgeting::Service>> {
    let mut c = crate::test_tools::create_test_container(pool).await?;
    let svc = c.get::<Arc<budgeting::Service>>().await?;
    Ok(svc)
}
