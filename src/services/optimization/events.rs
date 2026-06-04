use super::jobs::MaintainPopulationMessage;
use crate::services::optimization::repositories::requests;
use crate::services::{
    budgeting::{self, TransactionsFilter},
    evaluation,
    optimization::{
        self,
        service::{
            REASON_OPTIMIZATION_BUDGET_ADDED, REASON_OPTIMIZATION_CHARGED,
            REASON_OPTIMIZATION_CREATED,
        },
    },
};
use futures::future::BoxFuture;
use fx_event_bus::Handler;
use fx_mq_jobs::Queries;
use sqlx::PgTransaction;
use std::{ops::Neg, sync::Arc};
use tracing::instrument;

/// Handler that responds to genotype evaluations by scheduling population maintenance.
pub struct GenotypeEvaluatedHandler {
    queries: Arc<Queries>,
}

impl Handler<evaluation::GenotypeEvaluatedEvent> for GenotypeEvaluatedHandler {
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, tx))]
    fn handle<'a>(
        &'a self,
        input: Arc<evaluation::GenotypeEvaluatedEvent>,
        _polled_at: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> futures::future::BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), Self::Error>)> {
        Box::pin(async move {
            // Only process evaluations that belong to an optimization request
            if input.reason != super::service::EVALUATION_REASON {
                return (tx, Ok(()));
            }

            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            let ret = match publisher
                .publish(&MaintainPopulationMessage::new(
                    input.group_id,
                    input.genotype_id,
                    input.fitness,
                ))
                .await
            {
                Err(err) => {
                    tracing::error!(
                        message = "Failed to publish MaintainPopulation",
                        request_id = input.group_id.to_string(),
                    );
                    let err: super::Error = err.into();
                    Err(err)
                }
                _ => Ok(()),
            };

            (publisher.into(), ret)
        })
    }
}

/// Handler that responds to genotype evaluations by scheduling population maintenance.
pub struct TransactionCreatedHandler {
    queries: Arc<Queries>,
    optimizations: Arc<super::Service>,
}

impl Handler<budgeting::TransactionCreatedEvent> for TransactionCreatedHandler {
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, tx))]
    fn handle<'a>(
        &'a self,
        input: Arc<budgeting::TransactionCreatedEvent>,
        _polled_at: chrono::DateTime<chrono::Utc>,
        tx: sqlx::PgTransaction<'a>,
    ) -> BoxFuture<'a, (sqlx::PgTransaction<'a>, Result<(), super::Error>)> {
        Box::pin(async move {
            if ![
                REASON_OPTIMIZATION_CREATED,
                REASON_OPTIMIZATION_BUDGET_ADDED,
                REASON_OPTIMIZATION_CHARGED,
            ]
            .contains(&input.reason.as_str())
            {
                return (tx, Ok(()));
            }

            let filter =
                requests::SearchRequestsFilter::default().with_account_id(input.account_id);

            let requests = match self
                .optimizations
                .requests_ro
                .search_requests(&filter, 1)
                .await
            {
                Err(err) => return (tx, Err(err.into())),
                Ok(request) => request,
            };

            let Some(request) = requests.into_iter().next() else {
                return (tx, Err(super::Error::NoRequestOfAccount(input.account_id)));
            };

            let mut publisher = fx_mq_jobs::Publisher::<PgTransaction<'_>>::new(tx, &self.queries);

            if input.reason == REASON_OPTIMIZATION_CHARGED {
                if let Err(err) = publisher
                    .publish(&super::jobs::BreedGenotypesMessage::new(
                        request.id,
                        input.amount.neg(), // Negated to get the count to breed, as the charge should be negative
                    ))
                    .await
                {
                    tracing::error!(
                        message = "Failed to publish BreedGenotypes",
                        request_id = %request.id,
                    );
                    return (publisher.into(), Err(err.into()));
                };
            }

            if input.reason == REASON_OPTIMIZATION_CREATED
                || input.reason == REASON_OPTIMIZATION_BUDGET_ADDED
            {
                let filter = TransactionsFilter::default()
                    .with_account_id(request.account_id)
                    .with_reason(REASON_OPTIMIZATION_CHARGED);

                let last_charge_id =
                    match self.optimizations.transactions_ro.history(&filter, 1).await {
                        Err(err) => return (publisher.into(), Err(err.into())),
                        Ok(transactions) => transactions.into_iter().next().map(|t| t.id),
                    };

                if let Err(err) = publisher
                    .publish(&super::jobs::ChargeOptimizationBudgetMessage::new(
                        request.account_id,
                        request.schedule.population_size as i64,
                        last_charge_id,
                    ))
                    .await
                {
                    tracing::error!(
                        message = "Failed to publish ChargeOptimizationBudget",
                        request_id = %request.id,
                    );
                    return (publisher.into(), Err(err.into()));
                }
            }

            return (publisher.into(), Ok(()));
        })
    }
}

/// Registers all optimization event handlers with the event bus registry.
#[instrument(level = "debug", skip_all)]
pub(crate) fn register_event_handlers(
    registry: &mut fx_event_bus::EventHandlerRegistry,
    optimizations: &Arc<optimization::Service>,
    queries: &Arc<Queries>,
) {
    registry.with_handler(TransactionCreatedHandler {
        queries: queries.clone(),
        optimizations: optimizations.clone(),
    });

    registry.with_handler(GenotypeEvaluatedHandler {
        queries: queries.clone(),
    });
}
