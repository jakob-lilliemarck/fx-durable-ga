use clap::{Parser, Subcommand};
use fx_durable_ga::api_client::Client;

pub mod completions;
pub mod config;
pub mod genotypes;
pub mod requests;

#[derive(Parser)]
#[command(name = "ga", version, about = "CLI for FX Durable GA API")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Manage optimization requests
    #[command(subcommand)]
    Requests(requests::RequestsCommand),

    /// Manage genotypes
    #[command(subcommand)]
    Genotypes(genotypes::GenotypesCommand),

    /// Shell completion scripts
    #[command(subcommand)]
    Completions(completions::CompletionsCommand),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Handle completions early, without requiring API client
    if let Command::Completions(cmd) = cli.command {
        return cmd.execute().await;
    }

    let api_base_url = <config::ApiBaseUrl as config::Var>::from_env()?;

    let client = Client::new(api_base_url)?;

    match cli.command {
        Command::Requests(cmd) => cmd.execute(&client).await,
        Command::Genotypes(cmd) => cmd.execute(&client).await,
        Command::Completions(_) => unreachable!("Completions already handled"),
    }
}
