use anyhow::Result;
use chrono::Utc;
use futures::lock::Mutex;
use fx_durable_ga::repositories::genotypes::TypeName;
use fx_durable_ga::services::optimization::{
    self as foreign_service, FitnessGoal, OptimizerRegistry, Schedule, Selector,
};
use fx_durable_ga::{infrastructure::di::Container, services::evaluation};
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

const PROGRAM_OUTPUTS: usize = 3;
const MAX_TREE_DEPTH: usize = 6;
const STRUCT_MUTATION_DEPTH: usize = 3;
const CONST_MIN: i32 = -100;
const CONST_MAX: i32 = 100;
const DELTA_MIN: i32 = 1;
const DELTA_MAX: i32 = 100;
const NUDGE_MIN: i32 = 1;
const NUDGE_MAX: i32 = 10;
const FITNESS_TARGET: f64 = 0.05;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .pretty()
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    dotenvy::from_filename(".env.local").ok();

    // Create a DI container
    let mut c = Container::new();

    // invoke service registration
    c.invokable(|c| {
        Box::pin(async move {
            let svc = foreign_service::OptimizationService::new(
                "gp_point",
                GPPointManager {
                    target: Point {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                },
            );
            let provided = c.get::<Arc<Mutex<OptimizerRegistry>>>().await?;
            let mut lock = provided.lock().await;
            lock.register(svc.type_name, svc.optimizer);
            Ok(())
        })
    });

    // Invoke evaluator registration
    c.invokable(|c| {
        Box::pin(async move {
            let provided = c.get::<Arc<evaluation::Service>>().await?;
            provided
                .register(
                    "gp_point",
                    GPPointManager {
                        target: Point {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        },
                    },
                )
                .await;
            Ok(())
        })
    });

    // Register fx-durable-ga with the DI container
    //
    // NOTE!
    // Any provider overwrites must happen after registration!
    fx_durable_ga::register(&mut c);

    // Invoke all invokables
    c.invoke().await?;

    // Get an app instance from the container
    let app = c.get::<Arc<fx_durable_ga::bootstrap::App>>().await?;

    // Get a timestamp just before we start the optimization
    let started = Utc::now();

    // Create the optimization request
    let request_id = app
        .services()
        .optimization()
        .request_new(
            String::from("gp_point"),
            FitnessGoal::minimize(FITNESS_TARGET)?,
            Schedule::generational(15, 15),
            Selector::tournament(5),
            Some(serde_json::json!({
                "mutation_rate": 0.3,
                "temperature": 0.7,
            })),
            None::<()>,
        )
        .await?;

    println!("Optimization request submitted: {}", request_id);

    app.services()
        .synchronization()
        .wait_for(&request_id.to_string(), &started)
        .await?;

    let (genotype, fitness) = app
        .services()
        .optimization()
        .get_best_genotype(request_id)
        .await?
        .expect("evaluations should exist");

    if fitness >= FITNESS_TARGET {
        println!("Exhausted the optimization budget without reaching the optimization goal")
    }

    println!(
        "Best GP program: {} with distance {:.6}",
        genotype.id(),
        fitness
    );

    Ok(())
}

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

    fn mutate_param(&mut self, index: usize, rng: &mut dyn RngCore, temperature: f64) -> bool {
        let mut current = 0;
        Self::mutate_param_inner(self, index, &mut current, rng, temperature)
    }

    fn mutate_param_inner(
        node: &mut ScalarNode,
        target: usize,
        current: &mut usize,
        rng: &mut dyn RngCore,
        temperature: f64,
    ) -> bool {
        let has_param = matches!(
            node,
            ScalarNode::Const(_) | ScalarNode::Add { .. } | ScalarNode::Sub { .. }
        );
        if has_param {
            if *current == target {
                Self::apply_nudge(node, rng, temperature);
                return true;
            }
            *current += 1;
        }

        match node {
            ScalarNode::Const(_) => false,
            ScalarNode::Add { child, .. }
            | ScalarNode::Sub { child, .. }
            | ScalarNode::Pow(child) => {
                Self::mutate_param_inner(child, target, current, rng, temperature)
            }
        }
    }

    fn apply_nudge(node: &mut ScalarNode, rng: &mut dyn RngCore, temperature: f64) {
        let base_step = rng.random_range(NUDGE_MIN..=NUDGE_MAX) as f64;
        let delta_step = (base_step * temperature.max(0.1)).round().max(1.0) as i32;
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

    fn mutate_structure(&mut self, rng: &mut dyn RngCore, max_depth: usize) {
        let idx = rng.random_range(0..PROGRAM_OUTPUTS);
        let size = self.outputs[idx].size();
        if size == 0 {
            return;
        }
        let point = rng.random_range(0..size);
        let new_subtree = ScalarNode::random(rng, 0, max_depth);
        self.outputs[idx].replace_at(point, new_subtree);
    }

    fn mutate_parameter(&mut self, rng: &mut dyn RngCore, temperature: f64) {
        let idx = rng.random_range(0..PROGRAM_OUTPUTS);
        let count = self.outputs[idx].param_nodes();
        if count == 0 {
            return;
        }
        let target = rng.random_range(0..count);
        self.outputs[idx].mutate_param(target, rng, temperature);
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

#[derive(Clone)]
struct GPPointManager {
    target: Point,
}

impl TypeName for GPPointManager {
    fn type_name(&self) -> &'static str {
        "gp_point"
    }
}

impl foreign_service::Optimizer for GPPointManager {
    type Type = Program;

    fn random(&self, _user_defined: &Value) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        Ok(Program::random(&mut rng))
    }

    fn crossover(
        &self,
        parent1: Self::Type,
        parent2: Self::Type,
        _user_defined: &Value,
    ) -> anyhow::Result<Self::Type> {
        let mut rng = rand::rng();
        let mut child = parent1.clone();

        let idx1 = rng.random_range(0..PROGRAM_OUTPUTS);
        let idx2 = rng.random_range(0..PROGRAM_OUTPUTS);
        let size1 = parent1.outputs[idx1].size();
        let size2 = parent2.outputs[idx2].size();

        if size1 > 0 && size2 > 0 {
            let point1 = rng.random_range(0..size1);
            let point2 = rng.random_range(0..size2);
            if let Some(subtree) = parent2.outputs[idx2].get_node_at(point2) {
                child.outputs[idx1].replace_at(point1, subtree.clone());
            }
        }

        Ok(child)
    }

    fn mutate(&self, genotype: &mut Self::Type, user_defined: &Value) -> anyhow::Result<()> {
        let mut rng = rand::rng();
        // Reads mutation parameters from flat user_defined payload.
        let mutation_rate = user_defined
            .get("mutation_rate")
            .and_then(Value::as_f64)
            .unwrap_or(0.35);
        let temperature = user_defined
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(0.7);

        if rng.random_range(0.0..1.0) < mutation_rate {
            let depth = ((STRUCT_MUTATION_DEPTH as f64) * temperature.max(0.1))
                .round()
                .clamp(1.0, MAX_TREE_DEPTH as f64) as usize;
            genotype.mutate_structure(&mut rng, depth);
        }
        if rng.random_range(0.0..1.0) < mutation_rate {
            genotype.mutate_parameter(&mut rng, temperature);
        }

        Ok(())
    }
}

impl evaluation::Evaluator for GPPointManager {
    type Type = Program;

    fn evaluate<'a>(
        &'a self,
        genotype: &'a Self::Type,
    ) -> futures::future::BoxFuture<
        'a,
        std::result::Result<f64, Box<dyn std::error::Error + Send + Sync>>,
    > {
        let target = self.target;
        Box::pin(async move {
            let values = genotype.eval();
            let point = Point {
                x: values[0],
                y: values[1],
                z: values[2],
            };
            Ok(point.distance(&target))
        })
    }
}
