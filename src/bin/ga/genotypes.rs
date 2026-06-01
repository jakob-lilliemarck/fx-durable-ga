use anyhow::Result;
use clap::{Args, Subcommand};
use uuid::Uuid;

#[derive(Subcommand)]
pub enum GenotypesCommand {
    /// Backfill missing genotype embeddings
    BackfillEmbeddings(BackfillEmbeddingsCommand),
}

impl GenotypesCommand {
    pub async fn execute(self, client: &fx_durable_ga::api_client::Client) -> anyhow::Result<()> {
        match self {
            Self::BackfillEmbeddings(cmd) => cmd.execute(client).await,
        }
    }
}

#[derive(Args)]
pub struct BackfillEmbeddingsCommand {
    /// Optional request ID filter
    #[arg(long)]
    request_id: Option<Uuid>,

    /// Optional generation ID filters (repeatable)
    #[arg(long)]
    generation_id: Vec<i32>,

    /// Optional genotype ID filters (repeatable)
    #[arg(long)]
    genotype_id: Vec<Uuid>,

    /// Optional evaluation filter
    #[arg(long)]
    has_evaluation: Option<bool>,
}

impl BackfillEmbeddingsCommand {
    pub async fn execute(self, client: &fx_durable_ga::api_client::Client) -> Result<()> {
        client
            .genotypes_backfill_embeddings(
                self.request_id,
                self.generation_id,
                self.genotype_id,
                self.has_evaluation,
            )
            .await?;

        println!("Backfill request accepted");
        Ok(())
    }
}
