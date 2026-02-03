use crate::models::evolution::GenotypeManager;
use crate::models::{Genotype, Request};
use std::collections::HashSet;
use tracing::instrument;

/// Handles the breeding process by combining crossover and mutation operations.
pub(crate) struct Breeder;

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Deduplicated {
    pub(crate) genotype: Genotype,
    pub(crate) existing: bool,
    pub(crate) _idx: usize,
}

impl Deduplicated {
    fn new(genotype: Genotype, existing: bool, idx: usize) -> Deduplicated {
        Deduplicated {
            genotype,
            _idx: idx,
            existing,
        }
    }
}

impl Breeder {
    /// Creates a single child from two parents using crossover and mutation.
    #[instrument(level = "debug", skip(request, manager, parent_a, parent_b, rng), fields(parent_a_id = %parent_a.id(), parent_b_id = %parent_b.id(), generation_id = next_generation_id, type_hash = request.type_hash))]
    fn breed_child(
        request: &Request,
        manager: &dyn GenotypeManager,
        parent_a: &Genotype,
        parent_b: &Genotype,
        next_generation_id: i32,
        rng: &mut dyn rand::RngCore,
    ) -> anyhow::Result<Genotype> {
        let genome_a = parent_a.genome().clone();
        let genome_b = parent_b.genome().clone();

        // Crossover
        let mut genome = manager.crossover(&genome_a, &genome_b, rng, &request.user_defined)?;

        // Mutation
        manager.mutate(&mut genome, rng, &request.user_defined)?;

        let child = Genotype::new(
            &request.type_name,
            request.type_hash,
            genome,
            request.id,
            next_generation_id,
            Some(&parent_a.id),
            Some(&parent_b.id),
        );

        Ok(child)
    }

    /// Creates N children for each parent pair
    #[instrument(level = "debug", skip(request, manager, pairs), fields(num_pairs = pairs.len(), generation_id = next_generation_id, type_hash = request.type_hash))]
    pub(crate) fn breed_batch<'a>(
        request: &Request,
        manager: &dyn GenotypeManager,
        pairs: &[(&'a Genotype, &'a Genotype)],
        next_generation_id: i32,
        num_siblings: usize,
    ) -> anyhow::Result<(Vec<Vec<Genotype>>, Vec<i64>)> {
        let mut rng = rand::rng();
        let mut batch: Vec<Vec<Genotype>> = Vec::with_capacity(pairs.len() * num_siblings);
        let mut hashes: HashSet<i64> = HashSet::new();

        for (parent_a, parent_b) in pairs.iter() {
            let mut children: Vec<Genotype> = Vec::with_capacity(pairs.len() * num_siblings);
            for _ in 0..num_siblings {
                let child = Self::breed_child(
                    request,
                    manager,
                    parent_a,
                    parent_b,
                    next_generation_id,
                    &mut rng,
                )?;

                hashes.insert(child.genome_hash());
                children.push(child);
            }

            batch.push(children)
        }

        Ok((batch, hashes.drain().collect()))
    }

    /// Creates N children for each parent pair
    #[instrument(level = "debug")]
    pub(crate) fn deduplicate_batch(
        mut existing: HashSet<i64>,
        candidates: Vec<Vec<Genotype>>,
    ) -> Vec<Deduplicated> {
        let mut results = Vec::with_capacity(candidates.len());
        let mut taken: HashSet<i64> = HashSet::new();

        for (idx, siblings) in candidates.into_iter().enumerate() {
            let mut unique: Option<Genotype> = None;
            let mut unique_in_batch: Option<Genotype> = None;
            let mut fallback: Option<Genotype> = None;

            for (i, child) in siblings.into_iter().enumerate() {
                let in_batch = taken.contains(&child.genome_hash);
                let in_existing = existing.contains(&child.genome_hash);

                if !in_batch && !in_existing {
                    unique = Some(child);
                    break;
                }

                if in_batch || in_existing {
                    tracing::debug!(
                        message = "Encountered genome hash collision",
                        hash = child.genome_hash,
                        attempt = i
                    );
                }

                if !in_batch && unique_in_batch.is_none() {
                    unique_in_batch = Some(child);
                    continue;
                }

                if unique.is_none() && fallback.is_none() {
                    fallback = Some(child);
                }
            }

            let chosen = if let Some(g) = unique {
                existing.insert(g.genome_hash);
                taken.insert(g.genome_hash);
                Deduplicated::new(g, false, idx)
            } else if let Some(g) = unique_in_batch {
                tracing::warn!(
                    message = "Deduplication attempts exhausted. Emitting existing duplicate",
                    hash = g.genome_hash
                );
                taken.insert(g.genome_hash);
                Deduplicated::new(g, true, idx)
            } else if let Some(g) = fallback {
                tracing::warn!(
                    message = "Deduplication attempts exhausted. Emitting in-batch duplicate",
                    hash = g.genome_hash
                );
                let collided = existing.contains(&g.genome_hash);
                taken.insert(g.genome_hash);
                Deduplicated::new(g, collided, idx)
            } else {
                // No candidates available; this indicates upstream logic provided an empty slot.
                panic!("deduplicate received an empty candidate list");
            };

            results.push(chosen);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use uuid::Uuid;

    fn create_test_genotype(id: &str, genome: Value) -> Genotype {
        Genotype::new(
            "test",
            123,
            genome,
            Uuid::parse_str(id).unwrap(),
            1,
            None,
            None,
        )
    }

    /// Prefers new hashes over ones already in `existing`.
    #[test]
    fn deduplicate_prefers_new_hashes() {
        let g1 = create_test_genotype(
            "00000000-0000-0000-0000-000000000001",
            serde_json::json!([1, 2, 3]),
        );

        let g2 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([2, 3, 4]),
        );

        let g3 = create_test_genotype(
            "00000000-0000-0000-0000-000000000003",
            serde_json::json!([3, 4, 5]),
        );

        let g4 = create_test_genotype(
            "00000000-0000-0000-0000-000000000004",
            serde_json::json!([4, 5, 6]),
        );

        let mut existing = HashSet::<i64>::new();
        existing.insert(g1.genome_hash);
        existing.insert(g3.genome_hash);
        existing.insert(g4.genome_hash);

        let candidates = vec![vec![g1.clone(), g2.clone()], vec![g3.clone(), g4.clone()]];
        let results = Breeder::deduplicate_batch(existing, candidates);

        assert_eq!(results[0].existing, false);
        assert_eq!(results[0]._idx, 0);
        assert_eq!(results[0].genotype, g2);

        assert_eq!(results[1].existing, true);
        assert_eq!(results[1]._idx, 1);
        assert_eq!(results[1].genotype, g3);

        assert_eq!(results.len(), 2);
    }

    /// Falls back to picking duplicates within the batch when every option collides.
    /// Ensures such picks are marked as coming from `existing`.
    #[test]
    fn deduplicate_handles_batch_dupes() {
        let g1 = create_test_genotype(
            "00000000-0000-0000-0000-000000000001",
            serde_json::json!([1, 2, 3]),
        );

        let g2 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([1, 2, 3]),
        );

        let g3 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([1, 2, 3]),
        );

        let g4 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([1, 2, 3]),
        );

        let mut existing = HashSet::<i64>::new();
        existing.insert(g1.genome_hash);

        let candidates = vec![vec![g1.clone(), g2.clone()], vec![g3.clone(), g4.clone()]];
        let results = Breeder::deduplicate_batch(existing, candidates);

        assert_eq!(results[0].existing, true);
        assert_eq!(results[0]._idx, 0);
        assert_eq!(results[0].genotype, g1);

        assert_eq!(results[1].existing, true);
        assert_eq!(results[1]._idx, 1);
        assert_eq!(results[1].genotype, g3);

        assert_eq!(results.len(), 2);
    }

    /// Keeps the earliest candidate when it collides less than later siblings.
    /// Confirms ordering preference matches the selection rules.
    #[test]
    fn deduplicate_keeps_earlier_winner() {
        let g1 = create_test_genotype(
            "00000000-0000-0000-0000-000000000001",
            serde_json::json!([1, 2, 3]),
        );

        let g2 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([2, 3, 4]),
        );

        let g3 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([1, 2, 3]),
        );

        let g4 = create_test_genotype(
            "00000000-0000-0000-0000-000000000002",
            serde_json::json!([1, 2, 3]),
        );

        let mut existing = HashSet::<i64>::new();
        existing.insert(g2.genome_hash);

        let candidates = vec![vec![g1.clone(), g2.clone()], vec![g3.clone(), g4.clone()]];
        let results = Breeder::deduplicate_batch(existing, candidates);

        assert_eq!(results[0].existing, false);
        assert_eq!(results[0]._idx, 0);
        assert_eq!(results[0].genotype, g1);

        assert_eq!(results[1].existing, true);
        assert_eq!(results[1]._idx, 1);
        assert_eq!(results[1].genotype, g3);

        assert_eq!(results.len(), 2);
    }

    /// Returns the first batch-unique child even if it still collides in `existing`.
    /// Ensures the result is marked as coming from the DB.
    #[test]
    fn deduplicate_marks_batch_unique_existing() {
        let g1 = create_test_genotype(
            "00000000-0000-0000-0000-000000000010",
            serde_json::json!([1, 2, 3]),
        );

        let g2 = create_test_genotype(
            "00000000-0000-0000-0000-000000000011",
            serde_json::json!([4, 5, 6]),
        );

        let mut existing = HashSet::<i64>::new();
        existing.insert(g1.genome_hash);
        existing.insert(g2.genome_hash);

        let candidates = vec![vec![g1.clone(), g2.clone()]];
        let results = Breeder::deduplicate_batch(existing, candidates);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0]._idx, 0);
        assert_eq!(results[0].genotype, g1);
        assert!(results[0].existing);
    }

    /// Panics when a slot arrives empty, surfacing upstream contract violations.
    #[test]
    #[should_panic(expected = "deduplicate received an empty candidate list")]
    fn deduplicate_panics_on_empty_slot() {
        let existing = HashSet::<i64>::new();
        let candidates: Vec<Vec<Genotype>> = vec![vec![]];
        Breeder::deduplicate_batch(existing, candidates);
    }
}
