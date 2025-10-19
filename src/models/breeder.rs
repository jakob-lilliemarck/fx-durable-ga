use crate::models::{Genotype, Morphology, Request};

pub(crate) struct Breeder<'a> {
    request: &'a Request,
    morphology: &'a Morphology,
}

impl<'a> Breeder<'a> {
    pub(crate) fn new(request: &'a Request, morphology: &'a Morphology) -> Self {
        Self {
            request,
            morphology,
        }
    }

    fn breed_child(
        &self,
        parent1: &Genotype,
        parent2: &Genotype,
        next_generation_id: i32,
        progress: f64,
        rng: &mut impl rand::Rng,
    ) -> Genotype {
        let genome = self.request.crossover.apply(rng, parent1, parent2);
        let mut child = Genotype::new(
            &self.request.type_name,
            self.request.type_hash,
            genome,
            self.request.id,
            next_generation_id,
        );

        // progress passed in by service or future ProgressCalculator
        self.request
            .mutagen
            .mutate(rng, &mut child, self.morphology, progress);

        child
    }

    pub(crate) fn breed_batch(
        &self,
        parent_pairs: &[(usize, usize)],
        candidates: &[(Genotype, Option<f64>)],
        next_generation_id: i32,
        progress: f64,
        rng: &mut impl rand::Rng,
    ) -> Vec<Genotype> {
        parent_pairs
            .iter()
            .map(|&(i, j)| {
                self.breed_child(
                    &candidates[i].0,
                    &candidates[j].0,
                    next_generation_id,
                    progress,
                    rng,
                )
            })
            .collect()
    }
}
