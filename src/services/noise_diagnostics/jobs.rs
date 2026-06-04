use futures::future::BoxFuture;
use fx_mq_jobs::{Handler, Message};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EvaluateNoiseProbeGenotypeMessage {
    pub probe_id: Uuid,
    pub genotype: crate::repositories::genotypes::Genotype,
}

impl EvaluateNoiseProbeGenotypeMessage {
    pub fn new(probe_id: Uuid, genotype: crate::repositories::genotypes::Genotype) -> Self {
        Self {
            probe_id,
            genotype,
        }
    }
}

impl Message for EvaluateNoiseProbeGenotypeMessage {
    const NAME: &str = "EvaluateNoiseProbeGenotype";
}

pub(super) struct EvaluateNoiseProbeGenotypeHandler {
    service: Arc<super::Service>,
}

impl Handler for EvaluateNoiseProbeGenotypeHandler {
    type Message = EvaluateNoiseProbeGenotypeMessage;
    type Error = super::Error;

    #[instrument(level = "debug", skip(self, _lease_renewer))]
    fn handle<'a>(
        &'a self,
        message: Self::Message,
        _lease_renewer: fx_mq_jobs::LeaseRenewer,
    ) -> BoxFuture<'a, Result<(), Self::Error>> {
        Box::pin(async move {
            self.service
                .evaluate_probe(message.probe_id, message.genotype)
                .await
        })
    }

    fn max_attempts(&self) -> i32 {
        5
    }

    fn try_at(
        &self,
        _: i32,
        attempted_at: chrono::DateTime<chrono::Utc>,
    ) -> chrono::DateTime<chrono::Utc> {
        attempted_at + Duration::from_secs(10)
    }
}

pub(super) fn register_job_handlers(
    builder: fx_mq_jobs::RegistryBuilder,
    service: &Arc<super::Service>,
) -> fx_mq_jobs::RegistryBuilder {
    builder.with_handler(EvaluateNoiseProbeGenotypeHandler {
        service: service.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_has_correct_name() {
        assert_eq!(
            EvaluateNoiseProbeGenotypeMessage::NAME,
            "EvaluateNoiseProbeGenotype"
        );
    }
}
