# Optioryx Order Batching Project

This repository contains a C++20 implementation of the Order Batching assignment:

- Greedy savings heuristic
- Bounded greedy local search using snake-span-nearby swaps and relocations
- Simulated annealing with configurable start strategies, neighborhoods, and cooling schedules
- Genetic algorithm with permutation decoding and elitism
- Paper-inspired seed-and-selection heuristics
- Python scripts for aggregation and plotting

## Project Layout

- `include/optioryx`: public headers
- `src`: solver implementations and CLI
- `fixtures`: runnable synthetic instances
- `scripts`: benchmark runners, aggregators, and plot generators
- `tests`: C++ regression tests and a smoke test for the benchmark pipeline

## Instance Format

The canonical input format is a plain-text file with these sections:

```text
NAME small_fixture
NMAX 2
ITEMS DEPOT A B C
DEPOT DEPOT
SNAKE DEPOT A B C
DISTANCES
0 2 3 4
2 0 1 2
3 1 0 1
4 2 1 0
END_DISTANCES
ORDERS
o1 wave_1 A B
o2 wave_1 C
END_ORDERS
```

Rules:

- `ITEMS` defines all item identifiers used in the instance.
- `DEPOT` must reference one of the declared items.
- `SNAKE` must contain each item exactly once.
- `DISTANCES` must be a square matrix with one row per item.
- Each order line is `order_id wave_id item1 item2 ...`.

## CLI

```text
optioryx_cli solve --solver <name> --input <path> --output <path> [--config <path>] [--seed <n>]
optioryx_cli benchmark --input <path> --out-dir <path> [--suite <name>]
optioryx_cli dump-convergence --solver <name> --input <path> --output <path> [--config <path>] [--seed <n>]
```

Supported solver names:

- `greedy`
- `simulated_annealing` or `sa`
- `genetic`
- `seed_selection` or `seed`

## Example Commands

```text
optioryx_cli solve --solver greedy --input fixtures/small.instance --output build/greedy.json
optioryx_cli solve --solver genetic --input fixtures/small.instance --output build/ga.json --seed 11
optioryx_cli benchmark --input fixtures/medium.instance --out-dir build/benchmark --suite assignment
python scripts/aggregate_results.py --input-dir build/benchmark --output-dir build/benchmark/analysis
python scripts/plot_results.py --input-dir build/benchmark --output-dir build/benchmark/plots
python scripts/analyze_experiment_suite.py --input-dir build/benchmark --output-dir build/benchmark/factorial_analysis --suite-type sa
```

## Reproducible Benchmark Suites

The assignment experiments are split into independent suites. Stochastic configurations use the same ten seeds (`0` through `9`); deterministic configurations run once. Every numeric parameter is explicit in `suite_manifest.csv`, so a label cannot silently inherit a changed default.

| Suite | Configurations | Full-instance runs | Purpose |
|---|---:|---:|---|
| `sa_factorial` | 8 | 80 | Complete 2 cooling x 2 neighbourhood x 2 start design |
| `ga_sensitivity` | 9 | 90 | One-factor-at-a-time generations, population, mutation and elitism study |
| `ga_sensitivity_fast` | 9 | 90 | Practical OFAT study around the competition-fast GA budget |
| `seed_factorial` | 8 | 8 | Full 2 x 2 paper design, bounded lookahead under both seeds, and matched immediate-cost ablations |
| `comparison` | 4 | 22 | One preregistered representative of greedy, SA, GA and seed-selection |
| `comparison_fast` | 4 | 22 | Practical full-data comparison using the fast solver budgets |
| `comparison_fast_matched` | 4 | 22 | Fast comparison with the same LS1/w8 greedy construction for Greedy, SA and GA |
| `assignment` | 26 | 179 | Deduplicated union of the four academically complete suites |

The two SA cooling schedules both start at `60` and are calibrated to reach `0.05` after 400 updates. Its greedy-start factor is explicitly the strong greedy construction with one local-search pass and neighbour window 8; identifiers therefore contain `start-greedy-ls-1-win-8`. The randomized start does not use these two greedy-only controls. The GA baseline is population 40, 120 generations, tournament size 3, mutation 0.20, insertion share 0.50 and elitism 0.05. GA's greedy population seed also records its historical LS1/w8 construction explicitly in its configuration, identifier and solution metadata. Its eight variants change exactly one of generations (`60`, `240`), population (`20`, `80`), mutation (`0.05`, `0.40`) or elitism (`0.01`, `0.15`).

The `_fast` suites are additional practical workloads; they do not replace the academically complete suites and are not folded into `assignment`. `ga_sensitivity_fast` uses baseline population 12, 20 generations, tournament size 3, mutation 0.20, insertion share 0.50 and elitism 0.08. Its OFAT variants use generations `10`/`40`, population `6`/`24`, mutation `0.05`/`0.40`, or elitism `0`/`0.20`, always with ten seeds. The legacy `comparison_fast` keeps savings-greedy and SA at LS0/w0 while its GA population seed retains the historical LS1/w8 start. Existing results from that suite therefore remain reproducible but are not a fully matched warm-start comparison. `comparison_fast_matched` keeps the same fast SA/GA budgets and seed-selection comparator while using LS1/w8 for Greedy and both stochastic warm starts. The SA schedule explicitly uses `T0=60`, `Tf=0.05`, `alpha=0.9426276099699734` and the recorded Lundy--Mees counterpart `beta=0.1665277777777778`; SA and GA use ten seeds while deterministic methods run once.

### Seed-and-selection rule definitions

The paper's two-dimensional covering-area rules are represented on the fixed one-dimensional snake order without changing their direction:

- `smallest_snake_span` and `largest_snake_span` select the order with respectively the smallest or largest interval `max_snake_rank - min_snake_rank`. They adapt *smallest/greatest covering area*.
- `maximum_location_overlap` selects the order sharing the most unique pick locations with the current batch. It adapts *greatest identical/overlap locations*; ties use the smallest immediate route-cost increment.
- `minimum_span_expansion` minimizes `span(batch + order) - span(batch)`. It adapts *smallest additional covering area*; ties again use immediate route cost.

The custom `bounded_two_step_lookahead` first shortlists the `K` candidates with the smallest immediate route-cost increment. For every shortlisted first candidate it evaluates every feasible second candidate and chooses the first candidate whose best two-addition batch cost is smallest. This costs `O(Kn)` evaluations per choice and uses `lookahead_width=8` in the benchmark suite. `minimum_incremental_cost` is run under the same two seed rules as the immediate-cost ablation. The older `highest_standalone_cost` and `hybrid_score` values and aliases remain accepted for existing configs, but they are not paper factors.

A standalone seed-selection config can set the width explicitly:

```text
config_id=seed_largest_lookahead_k8
seed_rule=largest_snake_span
selection_rule=bounded_two_step_lookahead
lookahead_width=8
```

Inspect a workload without running a solver:

```text
build_release/optioryx_cli benchmark --input data --out-dir results/assignment_plan --suite assignment --dry-run
```

Run suites separately so every intermediate dataset remains available for the report:

```text
build_release/optioryx_cli benchmark --input data --out-dir results/sa_factorial --suite sa_factorial
build_release/optioryx_cli benchmark --input data --out-dir results/ga_sensitivity --suite ga_sensitivity
build_release/optioryx_cli benchmark --input data --out-dir results/ga_sensitivity_fast --suite ga_sensitivity_fast
build_release/optioryx_cli benchmark --input data --out-dir results/seed_factorial --suite seed_factorial
build_release/optioryx_cli benchmark --input data --out-dir results/comparison --suite comparison
build_release/optioryx_cli benchmark --input data --out-dir results/comparison_fast --suite comparison_fast
build_release/optioryx_cli benchmark --input data --out-dir results/comparison_fast_matched --suite comparison_fast_matched
```

On the competition instance with 12 waves, the combined academic suite plans 2,148 wave-solves. The fast GA sensitivity suite plans 1,080 wave-solves; either fast comparison suite plans 264. Use `--suite smoke` for a pipeline check.

Each completed benchmark writes:

- `suite_manifest.csv`: exact configs, seeds, expected workload and convergence-field semantics;
- `instance_runs.csv` and `summary.txt`: full-instance quality/runtime, including standard deviations across seeds;
- `runs.csv` and `wave_summary.csv`: raw and summarized wave-level results;
- `convergence.csv`: SA temperature or GA diversity in `control_value`;
- `ga_diversity.csv`: the same GA trace with the normalized-Hamming diversity column named explicitly;
- `solutions/`: the validated solution and metadata for every configuration/seed.

All floating-point fields in JSON and CSV use `max_digits10` precision. Objectives, runtimes, convergence values and continuous configuration metadata therefore round-trip to the identical binary `double`; this matters when configurations differ by only a few objective units. The plot pipeline writes both raster previews and vector PDFs, including an absolute `convergence` plot and a `convergence_relative` plot normalized against each seed's own start.

Use `scripts/analyze_experiment_suite.py` after a completed factor or sensitivity suite. For SA it computes paired-seed main effects and interactions; for GA it computes one-factor-at-a-time paired-seed deltas and final population diversity; for seed selection it keeps the paper-inspired `2 x 2` experiment separate from the matched custom-lookahead ablation. Every analysis writes CSV tables, PNG/PDF figures and an `analysis_protocol.json` with input hashes and statistical conventions:

```text
python scripts/analyze_experiment_suite.py --input-dir results/sa_factorial --output-dir results/sa_factorial/factorial_analysis --suite-type sa
python scripts/analyze_experiment_suite.py --input-dir results/ga_sensitivity_fast --output-dir results/ga_sensitivity_fast/factorial_analysis --suite-type ga
python scripts/analyze_experiment_suite.py --input-dir results/seed_factorial --output-dir results/seed_factorial/factorial_analysis --suite-type seed
```

For runtime growth, use controlled nested subsets of one fixed wave rather than treating differently structured waves as interchangeable input sizes. The runner gives every solver and seed the identical subsets, shuffles execution order, records hashes/hardware/protocol details, and marks its power-law fit as empirical rather than a complexity proof:

```text
python scripts/run_scaling_experiments.py --wave W000003 --experiment greedy_matched greedy configs/greedy_comparison_matched.cfg --experiment sa_matched simulated_annealing configs/sa_comparison_matched.cfg --experiment ga_matched genetic configs/ga_comparison_matched.cfg --n-list 50,100,200,400,800 --repeats 3 --seeds 0,1,2 --subset-seed 20260803 --schedule-seed 20260804 --timeout-seconds 120 --out-dir results/scaling_controlled_matched --require-release --fail-fast
```

These three scaling configurations all use the same LS1/window-8 greedy
construction. Use a new output directory (as above) so the legacy unmatched
study remains auditable rather than being overwritten.

This produces `runs.csv`, `summary.csv`, `fits.csv`, `subset_manifest.csv`, `protocol.json` and `runtime_vs_n.png/.pdf`.

## Competition Data

If the downloaded Kaggle files live in `data/`, you can point `--input` at the directory directly. The loader will read `pick_data.csv` and `dist_matrix.csv`, use `ASIS_SEQUENCE` as the fixed route order, and enforce `nmax = 20`.

Generate a submission:

```text
optioryx_cli solve --solver seed_selection --input data --output build/competition_submission.csv
```

For the current strongest competition-oriented run, build in Release mode and use:

```text
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release
build_release/optioryx_cli solve --solver greedy --input data --output build_release/greedy_final.csv --config configs/greedy_competition_final.cfg
build_release/optioryx_cli validate-submission --input data --submission build_release/greedy_final.csv
```

Latest local Release measurement for this final config: valid submission, score around `4.94153e+06`, 517 routes.

Summarize a submission for the paper:

```text
python scripts/summarize_submission.py --cli build_release/optioryx_cli --input data --submission build_release/greedy_final.csv --output build_release/greedy_final_summary.csv
```

Validate and score it locally:

```text
optioryx_cli validate-submission --input data --submission build/competition_submission.csv
```

Score the provided example submission:

```text
optioryx_cli validate-submission --input data --submission data/example_submission.csv
```

## Config Files

Optional solver config files use `key=value` lines. Example for simulated annealing:

```text
config_id=sa_lundy_swap_random
start_strategy=randomized
neighborhood=swap
cooling=lundy_mees
iterations=600
initial_temperature=80
final_temperature=0.05
lundy_beta=0.01
```

For a greedy SA start, the construction strength is explicit rather than inherited from `GreedyConfig` defaults:

```text
start_strategy=greedy
greedy_start_local_search_passes=1
greedy_start_local_search_neighbor_window=8
```

Greedy also accepts local-search controls:

```text
config_id=greedy_competition_strong
local_search_passes=2
local_search_neighbor_window=12
```

Use `configs/greedy_competition_fast.cfg` for the original fast savings baseline without local search, `configs/greedy_competition_local.cfg` for a balanced run, `configs/greedy_competition_strong.cfg`/`ultra`/`deep` for progressively stronger runs, `configs/greedy_competition_beam_eliminate.cfg` for the best pre-rebuild score, and `configs/greedy_competition_final.cfg` for the best measured quality-oriented competition score so far.
The reproducible matched scaling study uses `configs/greedy_comparison_matched.cfg`,
`configs/sa_comparison_matched.cfg`, and `configs/ga_comparison_matched.cfg`.

The final quality-oriented greedy pipeline reaches the wave-wise route-count
lower bound

```text
sum_w ceil(order_count(w) / nmax) = 517
```

The sum must be taken per wave because routes may not mix waves; applying one
ceiling to all 10,246 orders would therefore be invalid. Route-elimination uses
a bounded beam to move every order of one eligible source route into free
capacity elsewhere. Pair/triple rebuilds then repartition two or three routes
that are close after sorting on snake-span center. Every phase accepts only a
strict decrease in the exact route objective. The final config records
`route_elimination_ordering_trials=1` explicitly; it does not use a hidden
multi-ordering search.

## Verification

The repository ships with:

- synthetic fixtures in `fixtures`
- C++ regression tests in `tests/test_main.cpp`
- a smoke benchmark pipeline in `tests/smoke_test.py`

The Python analysis stack expects `numpy`, `pandas`, and `matplotlib`.

See `IMPROVEMENTS.md` for the feedback-driven changes, local measurements, and remaining limitations. Paper-oriented notes are in `docs/algorithm_comparison.md`, `docs/decision_log.md`, `docs/iteration_log.md`, and `docs/paper_outline.md`.
