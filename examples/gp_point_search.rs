//! # GP Point Search Example
//!
//! This example mirrors `examples/point_search.rs` but represents each coordinate
//! as a tree of simple GP nodes (`Const`, `Add`, `Sub`, `Pow`). The trees are
//! evolved via subtree crossover and targeted mutations that either replace
//! subtrees or nudge integer parameters.

use anyhow::Result;
use const_fnv1a_hash::fnv1a_hash_str_32;
use fx_durable_ga::{
    bootstrap::bootstrap,
    models::{FitnessGoal, GenotypeManager, Schedule, Selector},
    services::optimization::{register_event_handlers, register_job_handlers},
};
use fx_mq_jobs::{FX_MQ_JOBS_SCHEMA_NAME, Queries};
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::postgres::PgPoolOptions;
use std::{env, sync::Arc};
use std::{str::FromStr, time::Duration};
use uuid::Uuid;

const PROGRAM_OUTPUTS: usize = 3;
const MAX_TREE_DEPTH: usize = 6;
const STRUCT_MUTATION_DEPTH: usize = 3;
const CONST_MIN: i32 = -100;
const CONST_MAX: i32 = 100;
const DELTA_MIN: i32 = 1;
const DELTA_MAX: i32 = 100;
const NUDGE_MIN: i32 = 1;
const NUDGE_MAX: i32 = 10;
const WORKERS: usize = 6;
const FITNESS_TARGET: f64 = 0.05;

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ScalarNode {
    Const(i32),
    Add { delta: i32, child: Box<ScalarNode> },
    Sub { delta: i32, child: Box<ScalarNode> },
    Pow(Box<ScalarNode>),
}

impl ScalarNode {
    fn eval(&self) -> f64 {
        match self {
            ScalarNode::Const(v) => *v as f64,
            ScalarNode::Add { delta, child } => child.eval() + *delta as f64,
            ScalarNode::Sub { delta, child } => child.eval() - *delta as f64,
            ScalarNode::Pow(child) => {
                let v = child.eval();
                v * v
            }
        }
    }

    fn random(rng: &mut dyn RngCore, depth: usize, max_depth: usize) -> Self {
        if depth >= max_depth {
            return ScalarNode::Const(rng.random_range(CONST_MIN..=CONST_MAX));
        }

        match rng.random_range(0..4) {
            0 => ScalarNode::Const(rng.random_range(CONST_MIN..=CONST_MAX)),
            1 => ScalarNode::Add {
                delta: rng.random_range(DELTA_MIN..=DELTA_MAX),
                child: Box::new(ScalarNode::random(rng, depth + 1, max_depth)),
            },
            2 => ScalarNode::Sub {
                delta: rng.random_range(DELTA_MIN..=DELTA_MAX),
                child: Box::new(ScalarNode::random(rng, depth + 1, max_depth)),
            },
            _ => ScalarNode::Pow(Box::new(ScalarNode::random(rng, depth + 1, max_depth))),
        }
    }

    fn size(&self) -> usize {
        match self {
            ScalarNode::Const(_) => 1,
            ScalarNode::Add { child, .. }
            | ScalarNode::Sub { child, .. }
            | ScalarNode::Pow(child) => 1 + child.size(),
        }
    }

    fn param_nodes(&self) -> usize {
        match self {
            ScalarNode::Const(_) | ScalarNode::Add { .. } | ScalarNode::Sub { .. } => {
                1 + self.child_param_nodes()
            }
            ScalarNode::Pow(child) => child.param_nodes(),
        }
    }

    fn child_param_nodes(&self) -> usize {
        match self {
            ScalarNode::Add { child, .. }
            | ScalarNode::Sub { child, .. }
            | ScalarNode::Pow(child) => child.param_nodes(),
            ScalarNode::Const(_) => 0,
        }
    }

    fn mutate_param(&mut self, index: usize, rng: &mut dyn RngCore) -> bool {
        let mut current = 0;
        Self::mutate_param_inner(self, index, &mut current, rng)
    }

    fn mutate_param_inner(
        node: &mut ScalarNode,
        target: usize,
        current: &mut usize,
        rng: &mut dyn RngCore,
    ) -> bool {
        let has_param = matches!(
            node,
            ScalarNode::Const(_) | ScalarNode::Add { .. } | ScalarNode::Sub { .. }
        );
        if has_param {
            if *current == target {
                Self::apply_nudge(node, rng);
                return true;
            }
            *current += 1;
        }

        match node {
            ScalarNode::Const(_) => false,
            ScalarNode::Add { child, .. }
            | ScalarNode::Sub { child, .. }
            | ScalarNode::Pow(child) => Self::mutate_param_inner(child, target, current, rng),
        }
    }

    fn apply_nudge(node: &mut ScalarNode, rng: &mut dyn RngCore) {
        let delta_step = rng.random_range(NUDGE_MIN..=NUDGE_MAX);
        let direction = if rng.random_range(0..2) == 0 { -1 } else { 1 };
        let apply_clamp = |value: &mut i32, min, max| {
            *value = (*value + direction * delta_step).clamp(min, max);
        };

        match node {
            ScalarNode::Const(v) => apply_clamp(v, CONST_MIN, CONST_MAX),
            ScalarNode::Add { delta, .. } | ScalarNode::Sub { delta, .. } => {
                apply_clamp(delta, DELTA_MIN, DELTA_MAX)
            }
            ScalarNode::Pow(_) => {}
        }
    }

    fn get_node_at(&self, target: usize) -> Option<&ScalarNode> {
        let mut current = 0;
        Self::get_node(self, target, &mut current)
    }

    fn get_node<'a>(
        node: &'a ScalarNode,
        target: usize,
        current: &mut usize,
    ) -> Option<&'a ScalarNode> {
        if *current == target {
            return Some(node);
        }
        *current += 1;

        match node {
            ScalarNode::Const(_) => None,
            ScalarNode::Add { child, .. }
            | ScalarNode::Sub { child, .. }
            | ScalarNode::Pow(child) => Self::get_node(child, target, current),
        }
    }

    fn replace_at(&mut self, target: usize, replacement: ScalarNode) -> bool {
        let mut current = 0;
        Self::replace(self, target, &mut current, replacement)
    }

    fn replace(
        node: &mut ScalarNode,
        target: usize,
        current: &mut usize,
        replacement: ScalarNode,
    ) -> bool {
        if *current == target {
            *node = replacement;
            return true;
        }
        *current += 1;

        match node {
            ScalarNode::Const(_) => false,
            ScalarNode::Add { child, .. }
            | ScalarNode::Sub { child, .. }
            | ScalarNode::Pow(child) => Self::replace(child, target, current, replacement),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Program {
    outputs: Vec<ScalarNode>,
}

impl Program {
    fn random(rng: &mut dyn RngCore) -> Self {
        let outputs = (0..PROGRAM_OUTPUTS)
            .map(|_| ScalarNode::random(rng, 0, MAX_TREE_DEPTH))
            .collect();
        Self { outputs }
    }

    fn eval(&self) -> [f64; PROGRAM_OUTPUTS] {
        let mut out = [0.0; PROGRAM_OUTPUTS];
        for (idx, node) in self.outputs.iter().enumerate() {
            out[idx] = node.eval();
        }
        out
    }

    fn mutate_structure(&mut self, rng: &mut dyn RngCore) {
        let idx = rng.random_range(0..PROGRAM_OUTPUTS);
        let size = self.outputs[idx].size();
        if size == 0 {
            return;
        }
        let point = rng.random_range(0..size);
        let new_subtree = ScalarNode::random(rng, 0, STRUCT_MUTATION_DEPTH);
        self.outputs[idx].replace_at(point, new_subtree);
    }

    fn mutate_parameter(&mut self, rng: &mut dyn RngCore) {
        let idx = rng.random_range(0..PROGRAM_OUTPUTS);
        let count = self.outputs[idx].param_nodes();
        if count == 0 {
            return;
        }
        let target = rng.random_range(0..count);
        self.outputs[idx].mutate_param(target, rng);
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Point {
    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

struct GPPointManager {
    target: Point,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MutateConfig {
    #[serde(default = "default_mutation_rate")]
    mutation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CrossoverConfig {
    probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct UserConfig {
    #[serde(default)]
    crossover: Option<CrossoverConfig>,
    #[serde(default)]
    mutate: MutateConfig,
}

fn default_mutation_rate() -> f64 {
    0.35
}

impl GPPointManager {
    fn parse_program(&self, genome: &Value) -> anyhow::Result<Program> {
        Ok(serde_json::from_value(genome.clone())?)
    }
}

impl GenotypeManager for GPPointManager {
    fn name(&self) -> &'static str {
        "gp_point"
    }

    fn random(&self, rng: &mut dyn RngCore, _user_defined: &Value) -> anyhow::Result<Value> {
        Ok(serde_json::to_value(Program::random(rng))?)
    }

    fn crossover(
        &self,
        parent1: &Value,
        parent2: &Value,
        rng: &mut dyn RngCore,
        _user_defined: &Value,
    ) -> anyhow::Result<Value> {
        let prog1 = self.parse_program(parent1)?;
        let prog2 = self.parse_program(parent2)?;
        let mut child = prog1.clone();

        let idx1 = rng.random_range(0..PROGRAM_OUTPUTS);
        let idx2 = rng.random_range(0..PROGRAM_OUTPUTS);
        let size1 = prog1.outputs[idx1].size();
        let size2 = prog2.outputs[idx2].size();

        if size1 > 0 && size2 > 0 {
            let point1 = rng.random_range(0..size1);
            let point2 = rng.random_range(0..size2);
            if let Some(subtree) = prog2.outputs[idx2].get_node_at(point2) {
                child.outputs[idx1].replace_at(point1, subtree.clone());
            }
        }

        Ok(serde_json::to_value(child)?)
    }

    fn mutate(
        &self,
        genotype: &mut Value,
        rng: &mut dyn RngCore,
        progress: f64,
        user_defined: &Value,
    ) -> anyhow::Result<()> {
        let mut program = self.parse_program(genotype)?;
        let config = serde_json::from_value::<UserConfig>(user_defined.clone()).unwrap_or_default();
        let mutation_rate = config.mutate.mutation_rate;
        let cooled_rate = mutation_rate * (1.0 - progress).clamp(0.05, 1.0);

        if rng.random_range(0.0..1.0) < cooled_rate {
            program.mutate_structure(rng);
        }
        if rng.random_range(0.0..1.0) < cooled_rate {
            program.mutate_parameter(rng);
        }

        *genotype = serde_json::to_value(program)?;
        Ok(())
    }

    fn evaluate<'a>(
        &'a self,
        genotype: &'a Value,
        _user_defined: &'a Value,
    ) -> futures::future::BoxFuture<'a, anyhow::Result<f64>> {
        let program = match self.parse_program(genotype) {
            Ok(program) => program,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let target = self.target;
        Box::pin(async move {
            let values = program.eval();
            let point = Point {
                x: values[0],
                y: values[1],
                z: values[2],
            };
            Ok(point.distance(&target))
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::from_filename(".env.local").ok();
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let host_id_str = env::var("HOST_ID").expect("HOST_ID must be set");
    let host_id = Uuid::from_str(&host_id_str).expect("HOST_ID could not be parsed to UUID");

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .connect(&database_url)
        .await?;

    fx_event_bus::run_migrations(&pool).await?;
    fx_mq_jobs::run_migrations(&pool, FX_MQ_JOBS_SCHEMA_NAME).await?;

    let manager = GPPointManager {
        target: Point {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    };
    let service = Arc::new(
        bootstrap(host_id, pool.clone())
            .await?
            .with_genotype_manager(manager)
            .build(),
    );

    let mut registry = fx_event_bus::EventHandlerRegistry::new();
    register_event_handlers(
        Arc::new(Queries::new(FX_MQ_JOBS_SCHEMA_NAME)),
        service.clone(),
        &mut registry,
    );
    let mut event_listener = fx_event_bus::Listener::new(pool.clone(), registry);
    tokio::spawn(async move {
        event_listener.listen(None).await?;
        Ok::<(), sqlx::Error>(())
    });

    let host_id =
        Uuid::parse_str("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa").expect("valid static UUID");
    let mut jobs_listener = fx_mq_jobs::Listener::new(
        pool.clone(),
        register_job_handlers(&service, fx_mq_jobs::RegistryBuilder::new()),
        WORKERS,
        host_id,
        Duration::from_secs(900),
    )
    .await?;
    tokio::spawn(async move {
        jobs_listener.listen().await?;
        Ok::<(), anyhow::Error>(())
    });

    let user_config = UserConfig {
        crossover: Some(CrossoverConfig { probability: 0.5 }),
        mutate: MutateConfig { mutation_rate: 0.4 },
    };

    let request_id = service
        .new_optimization_request(
            "gp_point",
            fnv1a_hash_str_32("gp_point") as i32,
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(150, 25),
            Selector::tournament(5),
            user_config,
            None::<()>,
        )
        .await?;

    let poll_interval = Duration::from_secs(2);
    loop {
        if service.is_request_concluded(request_id).await? {
            if let Some((best, fitness)) = service.get_best_genotype(request_id).await? {
                println!(
                    "Best GP program: {} with distance {:.6}",
                    best.id(),
                    fitness
                );
            } else {
                println!("Optimization finished without evaluated genotypes.");
            }
            break;
        }
        tokio::time::sleep(poll_interval).await;
    }

    Ok(())
}
