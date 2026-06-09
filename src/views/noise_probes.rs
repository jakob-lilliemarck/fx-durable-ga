use crate::services::noise_diagnostics::ProbeNoise;
use maud::{Markup, html};

pub struct NoiseProbesView {
    probes: Vec<ProbeNoise>,
}

impl NoiseProbesView {
    pub fn new() -> Self {
        Self { probes: Vec::new() }
    }

    pub fn with_probes(mut self, probes: Vec<ProbeNoise>) -> Self {
        self.probes = probes;
        self
    }

    fn chart_config(&self) -> Option<String> {
        let has_data = self
            .probes
            .iter()
            .any(|p| p.mean_fitness.is_some() && p.stddev_fitness.is_some());

        if !has_data {
            return None;
        }

        let data: Vec<serde_json::Value> = self
            .probes
            .iter()
            .filter_map(|p| {
                Some(serde_json::json!({
                    "x": p.mean_fitness?,
                    "y": p.stddev_fitness?,
                }))
            })
            .collect();

        let config = serde_json::json!({
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": "Noise probes",
                    "data": data,
                    "backgroundColor": "rgba(31, 119, 180, 0.6)",
                }]
            },
            "options": {
                "responsive": true,
                "maintainAspectRatio": false,
                "animation": false,
                "scales": {
                    "x": {
                        "title": { "display": true, "text": "Mean fitness" }
                    },
                    "y": {
                        "title": { "display": true, "text": "Noise floor (std dev)" },
                        "min": 0
                    }
                },
                "plugins": {
                    "legend": { "display": false }
                }
            }
        });

        serde_json::to_string(&config).ok()
    }
}

impl maud::Render for NoiseProbesView {
    fn render(&self) -> Markup {
        let converged_count = self.probes.iter().filter(|p| p.converged).count();
        let evaluated_count = self.probes.iter().filter(|p| p.sample_count > 0).count();
        let total_count = self.probes.len();

        html! {
            div id="noise-probes" class="fx-flex" {
                div class="heading" { "Noise probes" }

                @if self.probes.is_empty() {
                    div { "Run a noise probe from the Genotypes section." }
                } @else {
                    div {
                        (total_count) " probe" (if total_count == 1 { "" } else { "s" })
                        ", " (evaluated_count) " with data"
                        ", " (converged_count) " converged"
                    }

                    @if let Some(config) = self.chart_config() {
                        section class="fx-flex chart" {
                            canvas data-line-chart data-line-config=(config) {}
                        }
                    } @else {
                        div { "Waiting for evaluations..." }
                    }
                }
            }
        }
    }
}

impl serde::Serialize for NoiseProbesView {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.probes.serialize(serializer)
    }
}

impl schemars::JsonSchema for NoiseProbesView {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        Vec::<ProbeNoise>::schema_name()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<ProbeNoise>::json_schema(generator)
    }
}
