use crate::repositories::genotypes as data;
use crate::services::optimization::Request;
use maud::html;
use serde::Serialize;
use uuid::Uuid;

pub struct PopulationView {
    request: Request,
    population: data::Population,
}

impl PopulationView {
    pub fn new(population: data::Population, request: Request) -> Self {
        Self {
            population,
            request,
        }
    }
}

impl maud::Render for PopulationView {
    fn render(&self) -> maud::Markup {
        html! {
            dl class="fx-flex fx-stats" {
                div {
                    dt { "Evaluation budget" }
                    dd { (self.request.schedule.max_evaluations) }
                }
                div {
                    dt { "Population size" }
                    dd { (self.request.schedule.population_size) }
                }
                div {
                    dt { "Evaluated" }
                    dd { (self.population.evaluated_genotypes) }
                }
                div {
                    dt { "Current Generation" }
                    dd { (self.population.current_generation) }
                }
            }
        }
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct Population {
    pub request_id: Uuid,
    pub(crate) evaluated_genotypes: i64,
    pub(crate) live_genotypes: i64,
    pub(crate) current_generation: i32,
    pub(crate) min_fitness: Option<f64>,
    pub(crate) max_fitness: Option<f64>,
}

impl From<data::Population> for Population {
    fn from(p: data::Population) -> Self {
        Self {
            request_id: p.request_id,
            evaluated_genotypes: p.evaluated_genotypes,
            live_genotypes: p.live_genotypes,
            current_generation: p.current_generation,
            min_fitness: p.min_fitness,
            max_fitness: p.max_fitness,
        }
    }
}

impl serde::Serialize for PopulationView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.population.serialize(serializer)
    }
}
