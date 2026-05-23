# Iteration Log

## Baseline

- Approach: original savings greedy plus existing SA/GA/seed-selection infrastructure.
- Issue: full competition greedy did not finish within the first 120 second debug-oriented run.
- Feedback gap: no strong local submission workflow and limited paper evidence.

## Iteration A: Lazy Savings Greedy

- Change: replaced repeated full pair rescans with a lazy priority queue.
- Result: full data runs became practical.
- Paper note: same greedy choice, better data structure.

## Iteration B: Faster Cost Evaluation

- Change: precomputed order item masks and used compact route-cost calculations in greedy.
- Result: Release greedy fast around `5.51108e+06`.
- Paper note: route cost depends on unique locations ordered by snake rank.

## Iteration C: Bounded Local Search

- Change: added swaps and relocations between batches with nearby snake-span centers.
- Result: strong score improvement, eventually to about `5.10029e+06` with wider configs.
- Paper note: deterministic alternative to long stochastic search.

## Iteration D: Route Elimination

- Change: dissolved small batches into other batches with spare capacity if incremental insertion cost was lower than removed route cost.
- Result: score improved to about `5.08056e+06`, routes reduced from 534 to 520.
- Paper note: quality improved at the cost of extra runtime.

## Iteration E: Wider Elimination And Post-Elimination Polish

- Change: allowed route elimination to consider larger source batches and reran pairwise local search after elimination.
- Result: score improved to about `5.07088e+06`, routes reduced to 518.
- Paper note: route elimination and local search are complementary; further window widening showed diminishing returns.

## Iteration F: Best-Improvement And Beam Elimination

- Change: route elimination now evaluates feasible source routes and chooses the one with the largest immediate saving. A bounded beam search can assign removed-route orders to destination routes.
- Result: score improved to about `5.06601e+06`, routes reduced to 517.
- Paper note: broader search inside route elimination helps, but larger beam settings showed diminishing returns and sometimes worse final local optima.

## Iteration G: Paper Support

- Change: added `docs/algorithm_comparison.md`, `docs/decision_log.md`, and `docs/paper_outline.md`.
- Result: important decisions and measurements are now traceable for the final paper.

## Iteration H: Multi-Ordering Beam Check

- Change: route-elimination beam assignment can test several deterministic source-orderings: many items first, few items first, widest snake span first, and regret first.
- Result: the measured score was about `5.06623e+06`, slightly worse than the single-ordering beam result of `5.06601e+06`.
- Paper note: this is a useful negative result. A locally better assignment order inside one eliminated route can steer later route-elimination choices into a worse local optimum.

## Iteration I: Two-Route Rebuild

- Change: added a deterministic pair-rebuild heuristic after route elimination. It combines two nearby routes, rebuilds two feasible routes using sorted split and greedy seed strategies, polishes the pair with swap/relocate descent, and accepts only strict improvements.
- Result: one rebuild pass with window 40 improved the score to about `5.04224e+06`; five passes plus final local search improved it to about `5.01849e+06`; six passes plus final local search reached about `5.01802e+06`.
- Paper note: after the theoretical minimum route count of 517 was reached, the main bottleneck became route composition. The rebuild heuristic improves composition without violating wave or capacity constraints.

## Iteration J: Final Polish Saturation

- Change: added a final local-search polish after rebuild passes and tested deeper final polish.
- Result: four final polish passes improved the rebuilt solution, but increasing to eight final passes did not improve beyond `5.01802e+06`.
- Paper note: this gives a natural stopping point for the current deterministic heuristic family and documents diminishing returns.
