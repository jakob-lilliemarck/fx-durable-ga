use crate::repositories::genotypes::Genotype;
use crate::services::optimization::OptimizerErased;
use crate::services::optimization::Request;
use tracing::instrument;

pub(crate) struct Breeder;

impl Breeder {
    /// Breeds a single child from two parents.
    #[instrument(level = "debug", skip(
        request,
        optimizer,
        parent_a,
        parent_b,
    ),
    fields(
        parent_a_id = %parent_a.id(),
        parent_b_id = %parent_b.id(),
        generation_id = next_generation_id,
    ))]
    fn breed_child(
        request: &Request,
        optimizer: &dyn OptimizerErased,
        parent_a: &Genotype,
        parent_b: &Genotype,
        next_generation_id: i32,
    ) -> anyhow::Result<Genotype> {
        let genome_a = parent_a.genome().clone();
        let genome_b = parent_b.genome().clone();

        // Crossover
        let genome = optimizer.crossover(genome_a, genome_b)?;

        // Mutation
        let genome = optimizer.mutate(genome)?;

        let child = Genotype::new(
            &request.type_name,
            genome,
            Some(request.id),
            Some(next_generation_id),
            Some(&parent_a.id),
            Some(&parent_b.id),
        )?;

        Ok(child)
    }

    /// Breeds one child per parent pair.
    #[instrument(level = "debug", skip(
        request,
        optimizer,
        pairs
    ),
    fields(
        num_offspring = pairs.len(),
        generation_id = next_generation_id,
    ))]
    pub(crate) fn breed_batch<'a>(
        request: &Request,
        optimizer: &dyn OptimizerErased,
        pairs: &[(&'a Genotype, &'a Genotype)],
        next_generation_id: i32,
    ) -> anyhow::Result<Vec<Genotype>> {
        let mut batch: Vec<Genotype> = Vec::with_capacity(pairs.len());

        for (parent_a, parent_b) in pairs.iter() {
            let child =
                Self::breed_child(request, optimizer, parent_a, parent_b, next_generation_id)?;
            batch.push(child);
        }

        Ok(batch)
    }
}
