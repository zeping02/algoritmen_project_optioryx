# Decision Log

# Decision 1

## Context
The initial greedy implementation was correct but too slow on the full competition data because it repeatedly rescanned all active batch pairs.

## Options
- Keep the full rescan for simplicity.
- Use a lazy priority queue of positive savings candidates.
- Replace greedy with a stochastic method immediately.

## Chosen Solution
Use a lazy priority queue for savings candidates.

## Reason
It preserves the same Clarke-Wright-style greedy idea while avoiding repeated full rescans.

## Consequences
The algorithm remains deterministic and explainable. It still evaluates many candidate pairs, but it became practical on the full dataset.

## Link With Paper
This supports the implementation section: same heuristic principle, better data structure.

# Decision 2

## Context
Optioryx feedback emphasized overlap, warm starts, and richer local search.

## Options
- Use only the savings result.
- Add stochastic simulated annealing after greedy.
- Add deterministic local search with swaps and relocations.

## Chosen Solution
Add bounded pairwise local search after savings.

## Reason
It gives a large score improvement and is easy to explain: neighboring batches along the snake route are likely to exchange compatible orders.

## Consequences
Runtime increases, but solution quality improves strongly. The method is deterministic for a given input/config.

## Link With Paper
This is a clear quality-versus-runtime trade-off.

# Decision 3

## Context
The improved solution still contained more routes than theoretically necessary.

## Options
- Ignore route count and keep pairwise local search only.
- Allow arbitrary large reoptimization.
- Add greedy route elimination for small batches.

## Chosen Solution
Add route elimination: try to dissolve small batches into existing batches with spare capacity when incremental insertion cost is lower than the removed route cost.

## Reason
It directly targets remaining route overhead and is connected to the competition objective.

## Consequences
Score improved from about `5.10029e+06` to `5.08056e+06`, and later post-elimination polish improved it further to about `5.07088e+06`.

## Link With Paper
This can be described as a post-processing heuristic that trades extra runtime for lower route count and better score.

# Decision 4

## Context
SA and GA are required by the assignment, but did not beat the final greedy variant in short competition runs.

## Options
- Make SA/GA the final submission method.
- Keep them for comparison and analysis.
- Remove them.

## Chosen Solution
Keep SA and GA as implemented comparison methods, but use greedy eliminate as the final submission generator.

## Reason
The final greedy variant gives the best measured local score and is more predictable.

## Consequences
The paper can still compare all required methods, while the competition submission uses the empirically strongest approach.

## Link With Paper
This supports the "why this final approach" section.

# Decision 5

## Context
Route elimination improved score, but the first version chose the first feasible route to remove and greedily assigned its orders.

## Options
- Keep first-improvement elimination.
- Select the feasible elimination with the largest immediate saving.
- Use beam search for assigning removed-route orders to destination routes.
- Use exact min-cost flow.

## Chosen Solution
Use best-improvement source selection and a bounded beam search for the assignment of orders from the removed route.

## Reason
Best-improvement avoids wasting capacity on a weak early elimination. Beam search explores several plausible destination assignments without the complexity of an exact solver.

## Consequences
The best measured score improved to about `5.06601e+06` and route count dropped to 517. Runtime and implementation complexity increased.

## Link With Paper
This is a useful example of a controlled trade-off: limited extra search gives measurable improvement while staying explainable.

# Decision 6

## Context
After best-improvement beam route elimination, the solution reached 517 routes, which equals the theoretical minimum obtained by summing `ceil(wave_orders / 20)` over all waves. Further score improvements therefore had to come from better route composition, not from fewer routes.

## Link with bedrijfsfeedback
This addresses the feedback about overlap and ASIS/snake spread: orders should be grouped so their union of pick locations forms a short snake-route span with useful overlap.

## Options
- Keep only swap/relocate local search.
- Add exact two-route repartitioning.
- Add deterministic two-route rebuild with strict acceptance.
- Add a large stochastic ruin-and-recreate phase.

## Chosen Solution
Add deterministic two-route rebuild as an optional greedy post-processing step.

## Reason
It can escape local optima that require moving multiple orders at once, but remains simpler and easier to explain than exact partitioning or a broad stochastic destroy/repair method. Every accepted rebuild is locally score-safe because it must strictly lower the combined cost of the two routes.

## Consequences
The best measured score improved from about `5.06601e+06` to about `5.01802e+06`. Runtime increased substantially, making this a quality-oriented offline configuration rather than a production-speed setting.

## Link With Paper
Use this decision in the discussion of trade-offs: the project first minimized route count, then improved fixed-route composition with a larger deterministic neighbourhood.

# Decision 7

## Context
Multi-ordering beam route elimination and deeper final polishing were tested as potential extensions.

## Link with bedrijfsfeedback
Both tests support the requested runtime-versus-quality analysis and show that alternatives were evaluated instead of assumed.

## Options
- Use multi-ordering beam assignment as final.
- Use deeper final local search.
- Keep them documented but not selected.

## Chosen Solution
Keep multi-ordering and deep final polish configurable, but do not select them as the final configuration.

## Reason
Multi-ordering produced a valid but slightly worse score (`5.06623e+06`). Increasing final polish from four to eight passes did not improve the best rebuild score (`5.01802e+06`).

## Consequences
The final config stays focused on the measured best combination: beam route elimination, six pair-rebuild passes with window 40, and four final local-search passes.

## Link With Paper
These are negative results for the alternatives section and strengthen the justification for the final choice.

# Decision 8

## Context
The two-route rebuild improved route composition, but its seed-based construction inserted remaining orders in a fixed span-descending order. That can place easy orders before constrained orders and close capacity in a way that blocks better assignments.

## Link with bedrijfsfeedback
The feedback emphasized overlap and route spread. Regret ordering is directly linked to this: orders that strongly prefer one rebuilt route because of lower incremental snake-route cost are placed first.

## Options
- Keep only fixed span-descending insertion.
- Try randomized rebuild orderings.
- Add deterministic regret-greedy insertion.

## Chosen Solution
Add deterministic regret-greedy insertion as an extra candidate inside pair rebuild.

## Reason
It stays explainable and reproducible, adds a larger effective neighbourhood, and every accepted rebuild must still strictly improve the combined route cost.

## Consequences
The final measured score improved from about `5.01802e+06` to about `5.00225e+06`. Eight regret-aware rebuild passes were best; a ninth pass was slightly worse, so it was not selected.

## Link With Paper
Use this decision to explain why the final improvement is not just parameter tuning: it changes the order-construction heuristic inside the rebuild neighbourhood.

# Decision 9

## Context
After regret-greedy pair rebuild, the solution was still locally constrained by moves that only repartition two routes at a time. Some improvements require a coordinated exchange among three routes.

## Link with bedrijfsfeedback
The feedback asks for richer neighbourhoods and better use of overlap/snake spread. Three-route rebuild is a richer deterministic neighbourhood that can redistribute orders across a small local route group while preserving wave and capacity constraints.

## Options
- Stop at pair rebuild because it is simpler.
- Add stochastic ruin-and-recreate over many routes.
- Add bounded deterministic three-route rebuild.

## Chosen Solution
Add bounded three-route rebuild with sorted split, seeded greedy assignment, and seeded regret assignment.

## Reason
It is larger than pair rebuild but still explainable, deterministic, and controlled by strict improvement acceptance. The search is bounded by a small snake-neighbourhood window.

## Consequences
The best measured score improved from about `5.00225e+06` to about `4.99696e+06`. A fourth triple pass matched the three-pass score but took longer, so three passes were selected.

## Link With Paper
This decision supports the final discussion of neighbourhood size: pair rebuild solved many local optima, while three-route rebuild handled remaining 3-route interactions with diminishing returns after three passes.

# Decision 10

## Context
Three-route rebuild changed route composition in larger steps. After such a move, smaller two-route rebuilds can again become useful because nearby route centers and contents have changed.

## Link with bedrijfsfeedback
This remains tied to richer neighbourhoods and overlap/snake spread: the algorithm alternates broad repartitioning with local refinement while preserving all hard submission constraints.

## Options
- Stop after triple rebuild and final swap/relocate local search.
- Add another triple pass.
- Rerun bounded pair rebuild after triple rebuild.

## Chosen Solution
Add optional post-triple pair rebuild and select two passes with window 40.

## Reason
It directly tests whether triple moves open new two-route improvements. The measured results improved from about `4.99696e+06` to about `4.99627e+06`; a third pass was worse, so two passes were selected.

## Consequences
The final score improved slightly at the cost of extra runtime. The method remains deterministic and strictly improvement-based.

## Link With Paper
Use this decision to discuss complementarity of neighbourhoods and diminishing returns.

# Decision 11

## Context
Post-triple pair rebuild improved the solution, suggesting that alternating neighbourhood sizes can expose improvements that a single block of passes misses.

## Link with bedrijfsfeedback
This continues the richer-neighbourhood requirement while staying deterministic, feasible, and traceable for the paper.

## Options
- Stop after post-triple pair rebuild.
- Add another pair pass.
- Add a second bounded triple phase after the pair polish.

## Chosen Solution
Add an optional second triple phase and select two passes with window 4.

## Reason
The first pass improved the score to about `4.99572e+06`; the second improved it further to about `4.99368e+06`. A third pass did not improve the score, so two passes are selected.

## Consequences
The final configuration is more runtime-heavy but gives the best measured local score. The method remains deterministic and strictly improvement-based.

## Link With Paper
Use this decision to explain the final interleaved local-search schedule: pair rebuild, triple rebuild, pair polish, then second triple polish.
