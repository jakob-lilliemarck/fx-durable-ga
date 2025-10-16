- Consider how float optimization would work - is it currently possible to optimize a number in tange 0.400-0.700 in 100 steps?
- Better strategy handling - make decisions in models, orchestrate in service
- Better initial distributions
- Restructure genotypes - should they really know their fitness and generation id? Perhaps track in populations instead?
- Fill out tests
- Add termination/completion record
- Add database indices
- Add instrumentation
- Add documentation
- Add a README

////
I am considering how tor refactor:
- maintain_rolling
- maintain_generational

ideally, I would like the service to be quite ignorant about the specifics of different startegies - that logic shall be contained in models, much like for src/models/mutagen.rs for instance.

In assessing this I have some observations:
1. current_gen is queries in both methods. Perhaps we shall maintain a generation counter on the Request entity instead? Something like an INTEGER[] that we append to when we increment the generation? Queries always select the latest number, never the full array?

2. all the arguments apart from the Request, come from Strategy - which is also contained on the request, so the request itself at src/models/request.rs  can probably be used to make these decisions? Some logic might be placed in src/models/strategy.rs

3. Both queries get population count - that we will probably still have to query for, but we should not need to call the query in two separate methods.

4. both methods interact with breed_new_individuals, but pass different numbers. breeding, or "selection" should also not happen in the body of service methods, but in a model like src/models/mutagen.rs or src/models/crossover.rs. Services shall orchestrate repository calls and logic, but the actual decision-making should happen in the models. - We may want to introduce something like a "Breeder"

Please assess the code-base and help me debate how to refactor this in a readable, nice and efficient way.
