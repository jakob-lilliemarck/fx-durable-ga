[x] Restructure genotypes - should they really know their fitness and generation id? Perhaps track in populations instead?
[x] Better strategy handling - make decisions in models, orchestrate in service
[x] Consider how float optimization would work - is it currently possible to optimize a number in tange 0.400-0.700 in 100 steps?

[ ] Better initial distributions - Latin Hypercube Sampling, Sobol sequences, or grid-based initialization to improve search space exploration
[ ] Deduplicate genomes - CRITICAL: Analysis shows 50-70% of evaluations are duplicates! Never evaluate IDENTICAL genomes twice within one request. Genomes should be hashable, lookup method using EXISTS to check if already evaluated. Only eliminate identical genomes, keep similar ones (fitness cliffs).

[ ] Early termination strategies - Stop when no improvement for N generations, population convergence detection, fitness plateau detection
[ ] Elite preservation - Keep top N performers without re-evaluation to reduce redundant computation

[ ] Request builder method - all of the configuration in a single type-safe place
[ ] Add database indexes

[ ] Fill out tests
[ ] Add termination/completion record
[ ] Add instrumentation
[ ] Add documentation
[ ] Add a README

////
// EFFICIENCY ANALYSIS - Key metric: reduce number of evaluations
//
// Analysis of duplicate genomes shows massive optimization potential:
// - {0,0,0} appears 16 times (optimal solution)
// - {99,0,0} appears 19 times
// - {230,0,0} appears 14 times
// - Many other high-duplication patterns
//
// PRIORITY OPTIMIZATIONS (by impact on evaluation count):
// 1. Genome deduplication (50-70% savings) - CRITICAL
// 2. Better initial distribution (10-30% savings)
// 3. Early termination (20-40% savings)
// 4. Elite preservation (10-20% savings)
//
// Combined these could reduce evaluations by 60-80%!
//
// IMPORTANT: Only eliminate IDENTICAL genomes, never similar ones.
// Similar genomes may have vastly different fitness (enum optimization, fitness cliffs).

////
I am considering how tor refactor:
- maintain_rolling
- maintain_generational

ideally, I would like the service to be quite ignorant about the specifics of different startegies - that logic shall be contained in models, much like for src/models/mutagen.rs for instance.

In assessing this I have some observations:
1. current_gen is queries in both methods. Perhaps we shall maintain a generation counter on the Request entity instead? Something like an INTEGER[] that we append to when we increment the generation? Queries always select the latest number, never the full array?

2. all the arguments apart from the Request, come from Strategy - which is also contained on the request, so the request itself at src/models/request.rs  can probably be used to make these decisions? Some logic might be placed in src/models/strategy.rs

3. Both queries get population count - that we will probably still have to query for, but we should not need to call the query in two separate methods.

4. both methods interact with breed_new_genotypes, but pass different numbers. breeding, or "selection" should also not happen in the body of service methods, but in a model like src/models/mutagen.rs or src/models/crossover.rs. Services shall orchestrate repository calls and logic, but the actual decision-making should happen in the models. - We may want to introduce something like a "Breeder"

Please assess the code-base and help me debate how to refactor this in a readable, nice and efficient way.
