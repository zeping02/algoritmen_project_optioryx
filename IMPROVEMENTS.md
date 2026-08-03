# Improvements

## Processed Optioryx Feedback

- Local submission validation is part of the CLI and now has regression tests for common Kaggle errors:
  duplicate or missing `ORDER_ID`, `LIST_ID` crossing waves, and `LIST_ID` above `nmax`.
- The greedy warm-start is stronger: after Clarke-Wright-style savings merges, a bounded local search swaps or relocates orders between nearby batches.
- Simulated annealing has an extra `merge_split` neighbourhood, and `mixed` now samples relocate, swap, and merge-split moves.
- Competition configs distinguish fast and stronger greedy runs.

## Algorithm Changes

- Greedy no longer rescans all active batch pairs from scratch after every merge.
  It uses a lazy priority queue of positive savings candidates.
- Greedy candidate evaluation uses compact item masks in the hot loop.
  The general `BatchEvaluator` remains the final source of truth for validation and scoring.
- A targeted local search improves greedy batches:
  batches are ordered by snake-route span center, and only nearby batches are compared.
  This keeps the search bounded while exploiting the overlap/snake-pattern structure from the feedback.
- The evaluator precomputes order item masks and order items sorted by snake rank.
- Local search now fully descends each visited pair of nearby batches before moving on,
  instead of applying only one swap or relocation per pair.
- A route-elimination post-processing step can dissolve small routes into existing routes with spare capacity.
- Route elimination now uses best-improvement source selection and can use beam search for assigning the removed route's orders to destination routes.
- Route elimination can optionally test several deterministic source-orderings during beam assignment. This was kept as an experimental switch because it was valid but did not beat the best single-ordering run.
- A two-route rebuild heuristic was added after route elimination. It combines two snake-nearby routes, rebuilds two feasible routes using sorted split and greedy seed strategies, polishes the pair with the existing swap/relocate descent, and only accepts strictly lower total cost.
- The two-route rebuild now also tests a regret-greedy assignment variant. At each step it assigns the order whose cost difference between the two rebuilt routes is largest, so hard-to-place orders are handled before easy ones.
- A bounded three-route rebuild step was added after pair rebuild. It rebuilds three snake-nearby routes with sorted split, seeded greedy assignment, and seeded regret assignment, then keeps the change only if the three-route total cost strictly decreases.
- Pair rebuild can optionally run again after triple rebuild. This catches two-route improvements opened by larger three-route moves.
- Triple rebuild can optionally run again after post-triple pair polish. This creates a bounded pair/triple cycle and tests whether small refinements reopen useful three-route moves.
- Triple rebuild can optionally test all local three-route combinations inside a small center-sorted window, in addition to the older arithmetic-gap triples. This expands the deterministic neighbourhood while keeping strict improvement acceptance.
- Pair rebuild can optionally run after the final second-triple phase. This catches two-route rebuild improvements opened by the last large three-route repartitioning step.
- A final local-search polish can run after route rebuilds to catch improvements opened by the larger pair reconstruction moves.

## Useful Competition Commands

Build in Release mode:

```text
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release
```

Generate a fast baseline:

```text
build_release/optioryx_cli solve --solver greedy --input data --output build_release/greedy_fast.csv --config configs/greedy_competition_fast.cfg
```

Generate the stronger local-search submission:

```text
build_release/optioryx_cli solve --solver greedy --input data --output build_release/greedy_strong.csv --config configs/greedy_competition_strong.cfg
build_release/optioryx_cli validate-submission --input data --submission build_release/greedy_strong.csv
```

## Latest Local Measurements

Measured on the included competition data in a Release build:

- `greedy_competition_fast`: score around `5.51108e+06`, about 20 seconds.
- `greedy_competition_local`: score around `5.36616e+06`, about 22 seconds.
- `greedy_competition_strong`: score around `5.26104e+06`, about 26 seconds.
- `greedy_competition_ultra`: score around `5.18844e+06`, about 43 seconds before pair descent.
- `greedy_competition_deeper`: score around `5.10877e+06`, about 88 seconds.
- `greedy_competition_best`: score around `5.10241e+06`, about 82 seconds.
- `greedy_competition_best_plus`: score around `5.10029e+06`, about 81 seconds.
- `greedy_competition_eliminate`: score around `5.08056e+06`, about 95 seconds, 520 routes.
- `greedy_competition_eliminate_wide`: score around `5.07925e+06`, about 78 seconds, 518 routes.
- `greedy_competition_eliminate_polished`: score around `5.07205e+06`, about 106 seconds, 518 routes.
- `greedy_competition_eliminate_polished_plus`: score around `5.07088e+06`, about 121 seconds, 518 routes.
- Earlier `greedy_competition_final`: score around `5.07088e+06`, no extra gain versus polished plus before beam/rebuild improvements.
- `greedy_competition_beam_eliminate`: score around `5.06601e+06`, about 135 seconds, 517 routes.
- `greedy_competition_beam_eliminate_polished`: score around `5.06601e+06`, no extra score gain versus beam eliminate in the measured run.
- `greedy_competition_beam_multi_order`: score around `5.06623e+06`, valid but not chosen because multi-ordering did not improve score.
- `greedy_competition_pair_rebuild_fast`: score around `5.05438e+06`, about 194 seconds, 517 routes.
- `greedy_competition_pair_rebuild_wide`: score around `5.04631e+06`, about 201 seconds, 517 routes.
- `greedy_competition_pair_rebuild_xwide`: score around `5.04224e+06`, about 310 seconds, 517 routes.
- `greedy_competition_pair_rebuild_five_pass_polished`: score around `5.01849e+06`, about 724 seconds, 517 routes.
- `greedy_competition_pair_rebuild_six_pass_polished`: score around `5.01802e+06`, about 853 seconds, 517 routes.
- `greedy_competition_pair_rebuild_six_pass_deep_polish`: score around `5.01802e+06`, no extra score gain versus four final polish passes.
- `greedy_competition_final` after regret-greedy rebuild assignment: score around `5.00225e+06`, about 487 seconds, 517 routes.
- `greedy_competition_final_plus3`: score around `5.00228e+06`, valid but not chosen because the ninth rebuild pass was slightly worse than eight passes.
- `greedy_competition_triple_rebuild_probe`: score around `5.01327e+06`, about 169 seconds, showing that triple rebuild improves a shallow pair-rebuild solution.
- `greedy_competition_final_triple`: score around `4.99975e+06`, about 520 seconds.
- `greedy_competition_final_triple2`: score around `4.99804e+06`, about 544 seconds.
- `greedy_competition_final_triple3`: score around `4.99696e+06`, about 565 seconds.
- `greedy_competition_final_triple4`: score around `4.99696e+06`, no gain over triple3 and therefore not chosen.
- `greedy_competition_final_triple_pair_polish`: score around `4.99656e+06`, about 609 seconds.
- `greedy_competition_final_triple_pair_polish_wide`: score around `4.99653e+06`, about 629 seconds.
- `greedy_competition_final_triple_pair_polish_wide2`: score around `4.99627e+06`, about 671 seconds.
- `greedy_competition_final_triple_pair_polish_wide3`: score around `4.99642e+06`, valid but worse than two post-triple pair passes.
- `greedy_competition_final_second_triple`: score around `4.99572e+06`, about 692 seconds.
- `greedy_competition_final_second_triple2`: score around `4.99368e+06`, about 704 seconds.
- `greedy_competition_final_second_triple3`: score around `4.99368e+06`, no score gain over two second-triple passes.
- `greedy_competition_final_combo_triple`: score around `4.99104e+06`, about 708 seconds, using local triple-combination window 4.
- `greedy_competition_final_combo_triple_wide`: score around `4.97641e+06`, about 938 seconds, using local triple-combination window 5.
- `greedy_competition_final_combo_triple_wider`: score around `4.97376e+06`, about 1095 seconds, using local triple-combination window 6.
- `greedy_competition_final_combo_triple_ultra`: score around `4.97093e+06`, about 1256 seconds, using local triple-combination window 7.
- `greedy_competition_final_combo_triple_ultra8`: score around `4.96238e+06`, about 1541 seconds, using local triple-combination window 8.
- `greedy_competition_final_combo_triple_ultra9`: score around `4.95201e+06`, about 1803 seconds, using local triple-combination window 9.
- `greedy_competition_final_combo_triple_ultra10`: score around `4.94891e+06`, about 2045 seconds, using local triple-combination window 10.
- `greedy_competition_final_combo_triple_ultra11`: score around `4.94495e+06`, about 2383 seconds.
- `greedy_competition_final_tail_pair`: score around `4.94229e+06`, about 2454 seconds, one tail pair-rebuild pass after the second triple phase.
- `greedy_competition_final_tail_pair2`: score around `4.94162e+06`, about 2489 seconds, two tail pair-rebuild passes.
- `greedy_competition_final_tail_pair3`: score around `4.94153e+06`, about 2523 seconds, best measured quality-oriented score so far.
- `greedy_competition_final_tail_pair4`: score around `4.94153e+06`, no gain over three tail pair passes.

The exact runtime depends on machine load, but the strong config is clearly better than the Pulse Pick reference score mentioned in `MetaAnalyse.pdf`.

## Remaining Limitations

- The exact initial savings phase still considers many candidate pairs, so runtime is not yet at the 5 second production reference.
- The best-scoring final config is intentionally quality-oriented and takes about 42 minutes locally. For a faster paper/competition comparison, the one-pass, five-pass rebuild, or second-triple configs are useful trade-off points.
- Larger triple-combination windows than 11 may still improve score, but they were not yet measured because each extra window step is a long compute experiment.
- Simulated annealing and the genetic algorithm currently benefit mainly from the improved greedy warm-start; their random search needs more efficient delta evaluation before very long runs become attractive.
- `src/main.cpp` still combines CLI parsing, solver construction, and benchmarking. Splitting it into smaller CLI/benchmark/solver-factory modules would improve maintainability.
