#[derive(sqlx::Type, Debug, Clone, PartialEq)]
#[sqlx(type_name = "fx_durable_ga.fitness_goal", rename_all = "lowercase")]
pub enum FitnessGoal {
    Minimize,
    Maximize,
}
