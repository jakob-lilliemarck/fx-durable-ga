use anyhow::{Context, Result};
use clap::{ArgGroup, Args, Subcommand};
use fx_durable_ga::services::optimization::{FitnessGoal, Schedule, Selector};
use uuid::Uuid;

#[derive(Subcommand)]
pub enum RequestsCommand {
    /// Create a new optimization request
    New(NewCommand),
    /// Interrupt a running optimization request
    Interrupt(InterruptCommand),
}

impl RequestsCommand {
    pub async fn execute(self, client: &fx_durable_ga::api_client::Client) -> anyhow::Result<()> {
        match self {
            Self::New(cmd) => cmd.execute(client).await,
            Self::Interrupt(cmd) => cmd.execute(client).await,
        }
    }
}

#[derive(Args)]
#[command(
    group(
        ArgGroup::new("goal")
            .required(true)
            .multiple(false)
            .args(["minimize", "maximize"])
    )
)]
pub struct NewCommand {
    /// The optimization type name registered on the server
    #[arg(long)]
    type_name: String,

    /// Total evaluation budget
    #[arg(long)]
    max_evaluations: u32,

    /// Maximum population size
    #[arg(long)]
    population_size: u32,

    /// Selection interval (defaults to population_size for generational behavior)
    #[arg(long)]
    selection_interval: Option<u32>,

    /// Tournament size for tournament selection (omit for roulette selection)
    #[arg(long)]
    tournament_size: Option<usize>,

    /// Minimize fitness until this threshold (mutually exclusive with maximize)
    #[arg(long)]
    minimize: Option<f64>,

    /// Maximize fitness until this threshold (mutually exclusive with minimize)
    #[arg(long)]
    maximize: Option<f64>,

    /// User-defined configuration as JSON string
    #[arg(long)]
    user_defined: Option<String>,

    /// Optional data payload as JSON string
    #[arg(long)]
    data: Option<String>,
}

impl NewCommand {
    pub async fn execute(self, client: &fx_durable_ga::api_client::Client) -> Result<()> {
        let request_id = create_request(
            client,
            self.type_name,
            self.max_evaluations,
            self.population_size,
            self.selection_interval,
            self.tournament_size,
            self.minimize,
            self.maximize,
            self.user_defined,
            self.data,
        )
        .await?;

        println!("Request created: {}", request_id);
        Ok(())
    }
}

async fn create_request(
    client: &fx_durable_ga::api_client::Client,
    type_name: String,
    max_evaluations: u32,
    population_size: u32,
    selection_interval: Option<u32>,
    tournament_size: Option<usize>,
    minimize: Option<f64>,
    maximize: Option<f64>,
    user_defined: Option<String>,
    data: Option<String>,
) -> Result<uuid::Uuid> {
    let schedule = build_schedule(max_evaluations, population_size, selection_interval)?;
    let selector = build_selector(tournament_size)?;
    let goal = build_fitness_goal(minimize, maximize)?;

    let user_defined: Option<serde_json::Value> = user_defined
        .map(|value| serde_json::from_str(&value))
        .transpose()
        .context("Failed to parse --user-defined as JSON")?;

    let data: Option<serde_json::Value> = data
        .map(|value| serde_json::from_str(&value))
        .transpose()
        .context("Failed to parse --data as JSON")?;

    let response = client
        .requests_new(type_name, goal, schedule, selector, user_defined, data)
        .await?;

    Ok(response.request_id)
}

fn build_schedule(
    max_evaluations: u32,
    population_size: u32,
    selection_interval: Option<u32>,
) -> Result<Schedule> {
    if max_evaluations == 0 {
        anyhow::bail!("--max-evaluations must be greater than 0");
    }

    if population_size == 0 {
        anyhow::bail!("--population-size must be greater than 0");
    }

    let selection_interval = selection_interval.unwrap_or(population_size);

    if selection_interval == 0 {
        anyhow::bail!("--selection-interval must be greater than 0");
    }

    if selection_interval > population_size {
        anyhow::bail!("--selection-interval must be <= --population-size");
    }

    Ok(Schedule::new(
        max_evaluations,
        population_size,
        selection_interval,
    ))
}

fn build_selector(tournament_size: Option<usize>) -> Result<Selector> {
    if let Some(size) = tournament_size {
        if size == 0 {
            anyhow::bail!("--tournament-size must be greater than 0");
        }
        Ok(Selector::tournament(size))
    } else {
        Ok(Selector::roulette())
    }
}

fn build_fitness_goal(minimize: Option<f64>, maximize: Option<f64>) -> Result<FitnessGoal> {
    if let Some(threshold) = minimize {
        FitnessGoal::minimize(threshold)
            .map_err(|err| anyhow::anyhow!("Invalid minimize threshold: {}", err))
    } else if let Some(threshold) = maximize {
        FitnessGoal::maximize(threshold)
            .map_err(|err| anyhow::anyhow!("Invalid maximize threshold: {}", err))
    } else {
        anyhow::bail!("Either --minimize or --maximize must be specified")
    }
}

#[derive(Args)]
pub struct InterruptCommand {
    /// The request ID to interrupt
    #[arg(long)]
    id: Uuid,
}

impl InterruptCommand {
    pub async fn execute(self, client: &fx_durable_ga::api_client::Client) -> Result<()> {
        client.requests_interrupt(self.id).await?;
        println!("Request interrupted: {}", self.id);
        Ok(())
    }
}
