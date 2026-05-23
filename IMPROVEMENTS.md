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
- `greedy_competition_ultra`: score around `5.18844e+06`, about 43 seconds.

The exact runtime depends on machine load, but the strong config is clearly better than the Pulse Pick reference score mentioned in `MetaAnalyse.pdf`.

## Remaining Limitations

- The exact initial savings phase still considers many candidate pairs, so runtime is not yet at the 5 second production reference.
- Simulated annealing and the genetic algorithm currently benefit mainly from the improved greedy warm-start; their random search needs more efficient delta evaluation before very long runs become attractive.
- `src/main.cpp` still combines CLI parsing, solver construction, and benchmarking. Splitting it into smaller CLI/benchmark/solver-factory modules would improve maintainability.
