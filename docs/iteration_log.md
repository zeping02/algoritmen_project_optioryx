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

## Iteration K: Regret-Greedy Pair Rebuild

- Change: pair rebuild now also tries a regret-greedy assignment order. Instead of inserting orders in fixed span order only, it repeatedly chooses the order for which the difference between the two route insertion costs is largest.
- Result: the one-pass window-40 rebuild improved from about `5.04224e+06` to `5.03799e+06`. The final configuration improved from `5.01802e+06` to `5.00225e+06`.
- Paper note: this is a stronger but still deterministic rebuild neighbourhood. It supports the argument that remaining improvement came from route composition rather than route count.

## Iteration L: Extra Rebuild Pass Saturation

- Change: tested seven, eight, and nine regret-aware rebuild passes.
- Result: seven passes reached about `5.00343e+06`, eight passes reached the best score of about `5.00225e+06`, and nine passes was slightly worse at about `5.00228e+06`.
- Paper note: eight passes is selected because it is the best measured setting; the ninth pass is a documented negative result.

## Iteration M: Three-Route Rebuild

- Change: added a bounded three-route rebuild after pair rebuild. It combines three snake-nearby routes, tries sorted split, seeded greedy assignment, and seeded regret assignment, then accepts only if the three-route total cost strictly decreases.
- Result: a probe with one pair-rebuild pass plus one triple pass scored about `5.01327e+06`. On the final setting, one triple pass reached `4.99975e+06`, two passes reached `4.99804e+06`, and three passes reached `4.99696e+06`.
- Paper note: this shows that after pair rebuild saturation, remaining improvements can still require coordinated movement across three routes.

## Iteration N: Triple-Pass Saturation

- Change: tested a fourth three-route rebuild pass.
- Result: four passes matched the three-pass score (`4.99696e+06`) but took longer, so three passes were selected.
- Paper note: the fourth pass is a clean negative result and justifies the final parameter choice.

## Iteration O: Post-Triple Pair Polish

- Change: reran pair rebuild after triple rebuild, because three-route moves can create new two-route improvement opportunities.
- Result: one post-triple pair pass with window 16 reached about `4.99656e+06`; window 40 reached about `4.99653e+06`; two passes with window 40 reached the best score of about `4.99627e+06`.
- Paper note: this demonstrates an interleaved-neighbourhood effect: larger triple moves and smaller pair moves are complementary.

## Iteration P: Post-Triple Pair Saturation

- Change: tested a third post-triple pair rebuild pass.
- Result: the third pass was worse at about `4.99642e+06`, so two post-triple pair passes were selected.
- Paper note: this is a documented negative result that supports the final parameter choice.

## Iteration Q: Second Triple Phase

- Change: after post-triple pair polish, ran another bounded triple-rebuild phase to test whether the pair polish reopened useful three-route improvements.
- Result: one second-triple pass reached about `4.99572e+06`; two passes reached about `4.99368e+06`.
- Paper note: the improvement confirms that pair and triple neighbourhoods are complementary when interleaved.

## Iteration R: Second Triple Saturation

- Change: tested a third pass in the second triple phase.
- Result: the third pass matched the two-pass score (`4.99368e+06`) but took longer, so two passes were selected.
- Paper note: another negative result that explains the final parameter choice.

## Iteration S: Combinatorial Triple Neighbourhood

- Change: extended triple rebuild so it can test all local three-route combinations inside a center-sorted window, in addition to the previous arithmetic-gap triples.
- Result: with combination window 4 the score improved from `4.99368e+06` to about `4.99104e+06` at roughly the same runtime scale.
- Paper note: the older triple scan missed route triples such as positions `(i, i+1, i+4)` or `(i, i+2, i+3)`. The new option expands the deterministic neighbourhood while still accepting only strictly lower three-route cost.

## Iteration T: Combination-Window Tuning

- Change: benchmarked combination windows 5, 6, 7, 8, 9, 10, and 11 on the full competition data.
- Result: the best measured score is now `4.94495e+06` with 517 routes and valid submission. Runtime rose from about 704 seconds for the old second-triple config to about 2383 seconds for window 11.
- Paper note: this is a clear quality/runtime trade-off. Larger windows improve route composition, but the algorithm becomes an offline-quality solver rather than a fast operational solver.

## Iteration U: Tail Pair Rebuild

- Change: after the second combinatorial triple phase, added an optional pair-rebuild phase to catch two-route improvements opened by the final large three-route moves.
- Result: one pass improved the score to about `4.94229e+06`; two passes improved it to about `4.94162e+06`; three passes reached about `4.94153e+06`.
- Paper note: this confirms the same complementarity pattern as earlier: a broad triple neighbourhood can reopen smaller pair-neighbourhood improvements.

## Iteration V: Tail Pair Saturation

- Change: tested a fourth tail pair-rebuild pass.
- Result: the fourth pass matched the three-pass score (`4.94153e+06`) but took longer, so three tail pair passes were selected.
- Paper note: this is a clean negative result and justifies the chosen stopping point for this added phase.
