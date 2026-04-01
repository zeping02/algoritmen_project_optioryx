# Optioryx Order Batching Project

This repository contains a C++20 implementation of the Order Batching assignment:

- Greedy savings heuristic
- Simulated annealing with configurable start strategies, neighborhoods, and cooling schedules
- Genetic algorithm with permutation decoding and elitism
- Paper-inspired seed-and-selection heuristics
- Python scripts for aggregation and plotting

## Project Layout

- `include/optioryx`: public headers
- `src`: solver implementations and CLI
- `data/fixtures`: runnable synthetic instances
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
optioryx_cli solve --solver greedy --input data/fixtures/small.instance --output build/greedy.json
optioryx_cli solve --solver genetic --input data/fixtures/small.instance --output build/ga.json --seed 11
optioryx_cli benchmark --input data/fixtures/medium.instance --out-dir build/benchmark --suite assignment
python scripts/aggregate_results.py --input-dir build/benchmark --output-dir build/benchmark/analysis
python scripts/plot_results.py --input-dir build/benchmark --output-dir build/benchmark/plots
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

## Verification

The repository ships with:

- synthetic fixtures in `data/fixtures`
- C++ regression tests in `tests/test_main.cpp`
- a smoke benchmark pipeline in `tests/smoke_test.py`

The Python analysis stack expects `numpy`, `pandas`, and `matplotlib`.
