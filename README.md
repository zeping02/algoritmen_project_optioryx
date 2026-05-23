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
```

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

Latest local Release measurement for this final config: valid submission, score around `4.99368e+06`, 517 routes.

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

Greedy also accepts local-search controls:

```text
config_id=greedy_competition_strong
local_search_passes=2
local_search_neighbor_window=12
```

Use `configs/greedy_competition_fast.cfg` for the original fast savings baseline without local search, `configs/greedy_competition_local.cfg` for a balanced run, `configs/greedy_competition_strong.cfg`/`ultra`/`deep` for progressively stronger runs, `configs/greedy_competition_beam_eliminate.cfg` for the best pre-rebuild score, and `configs/greedy_competition_final.cfg` for the best measured competition score so far.

## Verification

The repository ships with:

- synthetic fixtures in `fixtures`
- C++ regression tests in `tests/test_main.cpp`
- a smoke benchmark pipeline in `tests/smoke_test.py`

The Python analysis stack expects `numpy`, `pandas`, and `matplotlib`.

See `IMPROVEMENTS.md` for the feedback-driven changes, local measurements, and remaining limitations. Paper-oriented notes are in `docs/algorithm_comparison.md`, `docs/decision_log.md`, `docs/iteration_log.md`, and `docs/paper_outline.md`.
