import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_parameters(value: str) -> dict[str, str]:
    return dict(parameter.split("=", 1) for parameter in value.split(";") if parameter)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    workdir = Path(args.workdir)
    benchmark_dir = workdir / "benchmark"

    if workdir.exists():
        shutil.rmtree(workdir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            args.cli,
            "benchmark",
            "--input",
            args.instance,
            "--out-dir",
            str(benchmark_dir),
            "--suite",
            "smoke",
        ],
        cwd=repo_root,
    )

    run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_results.py"),
            "--input-dir",
            str(benchmark_dir),
            "--output-dir",
            str(benchmark_dir / "analysis"),
        ],
        cwd=repo_root,
    )

    run(
        [
            sys.executable,
            str(repo_root / "scripts" / "plot_results.py"),
            "--input-dir",
            str(benchmark_dir),
            "--output-dir",
            str(benchmark_dir / "plots"),
        ],
        cwd=repo_root,
    )

    expected_files = [
        benchmark_dir / "runs.csv",
        benchmark_dir / "instance_runs.csv",
        benchmark_dir / "convergence.csv",
        benchmark_dir / "ga_diversity.csv",
        benchmark_dir / "suite_manifest.csv",
        benchmark_dir / "wave_summary.csv",
        benchmark_dir / "summary.txt",
        benchmark_dir / "analysis" / "summary.csv",
        benchmark_dir / "plots" / "runtime.png",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        print("Missing smoke-test outputs:")
        for path in missing:
            print(path)
        return 1

    smoke_sa_solutions = list((benchmark_dir / "solutions").glob("simulated_annealing_*.json"))
    if not smoke_sa_solutions:
        print("Smoke benchmark did not write an SA solution")
        return 1
    with smoke_sa_solutions[0].open(encoding="utf-8") as handle:
        smoke_sa_solution = json.load(handle)
    if any(
        wave["metadata"].get("greedy_start_local_search_passes") != "1"
        or wave["metadata"].get("greedy_start_local_search_neighbor_window") != "8"
        for wave in smoke_sa_solution["waves"]
    ):
        print("SA parser/metadata did not preserve the explicit strong greedy-start config")
        return 1

    smoke_ga_solutions = list((benchmark_dir / "solutions").glob("genetic_*.json"))
    if not smoke_ga_solutions:
        print("Smoke benchmark did not write a GA solution")
        return 1
    with smoke_ga_solutions[0].open(encoding="utf-8") as handle:
        smoke_ga_solution = json.load(handle)
    if "start-greedy-ls-1-win-8" not in smoke_ga_solution["config_id"] or any(
        wave["metadata"].get("greedy_start_local_search_passes") != "1"
        or wave["metadata"].get("greedy_start_local_search_neighbor_window") != "8"
        for wave in smoke_ga_solution["waves"]
    ):
        print("GA parser/metadata did not preserve the explicit LS1/w8 greedy-start config")
        return 1

    ga_override_config = workdir / "ga_start_override.cfg"
    ga_override_config.write_text(
        "\n".join(
            [
                "config_id=ga_smoke_start-greedy-ls-0-win-0",
                "greedy_start_local_search_passes=0",
                "greedy_start_local_search_neighbor_window=0",
                "population_size=2",
                "generations=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ga_override_solution_path = workdir / "ga_start_override.json"
    run(
        [
            args.cli,
            "solve",
            "--solver",
            "genetic",
            "--input",
            args.instance,
            "--output",
            str(ga_override_solution_path),
            "--config",
            str(ga_override_config),
            "--seed",
            "0",
        ],
        cwd=repo_root,
    )
    with ga_override_solution_path.open(encoding="utf-8") as handle:
        ga_override_solution = json.load(handle)
    if any(
        wave["metadata"].get("greedy_start_local_search_passes") != "0"
        or wave["metadata"].get("greedy_start_local_search_neighbor_window") != "0"
        for wave in ga_override_solution["waves"]
    ):
        print("GA CLI parser did not apply the explicit LS0/w0 warm-start override")
        return 1

    # Dry-run manifests audit the complete experiment design without executing
    # the expensive assignment workload on every CI run.
    expected_plans = {
        "sa_factorial": (8, 80),
        "ga_sensitivity": (9, 90),
        "ga_sensitivity_fast": (9, 90),
        "seed_factorial": (8, 8),
        "comparison": (4, 22),
        "comparison_fast": (4, 22),
        "comparison_fast_matched": (4, 22),
        "assignment": (26, 179),
    }
    manifests: dict[str, list[dict[str, str]]] = {}
    for suite, (expected_configs, expected_runs) in expected_plans.items():
        plan_dir = workdir / "plans" / suite
        run(
            [
                args.cli,
                "benchmark",
                "--input",
                args.instance,
                "--out-dir",
                str(plan_dir),
                "--suite",
                suite,
                "--dry-run",
            ],
            cwd=repo_root,
        )
        rows = read_manifest(plan_dir / "suite_manifest.csv")
        manifests[suite] = rows
        if len(rows) != expected_configs:
            print(f"{suite}: expected {expected_configs} configs, got {len(rows)}")
            return 1
        planned_runs = sum(int(row["instance_runs"]) for row in rows)
        if planned_runs != expected_runs:
            print(f"{suite}: expected {expected_runs} runs, got {planned_runs}")
            return 1
        for row in rows:
            parameters = parse_parameters(row["parameters"])
            if parameters.get("config_id") != row["config_id"]:
                print(f"{suite}: manifest/config label mismatch")
                return 1

    sa_rows = manifests["sa_factorial"]
    sa_factors = {
        (
            parse_parameters(row["parameters"])["cooling"],
            parse_parameters(row["parameters"])["neighborhood"],
            parse_parameters(row["parameters"])["start_strategy"],
        )
        for row in sa_rows
    }
    if len(sa_factors) != 8 or any(int(row["seed_count"]) != 10 for row in sa_rows):
        print("sa_factorial is not a complete 2x2x2 design with ten seeds")
        return 1
    for row in sa_rows:
        parameters = parse_parameters(row["parameters"])
        if (
            parameters["greedy_start_local_search_passes"] != "1"
            or parameters["greedy_start_local_search_neighbor_window"] != "8"
        ):
            print("sa_factorial does not keep the explicit strong LS1/w8 greedy-start config")
            return 1
        if parameters["start_strategy"] == "greedy" and "start-greedy-ls-1-win-8" not in row["config_id"]:
            print("sa_factorial greedy-start config_id does not expose LS1/w8")
            return 1

    ga_rows = manifests["ga_sensitivity"]
    if any(row["convergence_control"] != "population_diversity_mean_normalized_hamming" for row in ga_rows):
        print("ga_sensitivity does not label its diversity measure")
        return 1
    if any(
        parse_parameters(row["parameters"])["greedy_start_local_search_passes"] != "1"
        or parse_parameters(row["parameters"])["greedy_start_local_search_neighbor_window"] != "8"
        or "start-greedy-ls-1-win-8" not in row["config_id"]
        for row in ga_rows
    ):
        print("ga_sensitivity does not expose its LS1/w8 greedy warm start")
        return 1

    ga_fast_rows = manifests["ga_sensitivity_fast"]
    ga_fast_parameters = [parse_parameters(row["parameters"]) for row in ga_fast_rows]
    actual_ga_fast_design = {
        (
            parameters["population_size"],
            parameters["generations"],
            parameters["tournament_size"],
            parameters["mutation_rate"],
            parameters["insertion_rate"],
            parameters["elitism_fraction"],
        )
        for parameters in ga_fast_parameters
    }
    expected_ga_fast_design = {
        ("12", "20", "3", "0.20", "0.50", "0.08"),
        ("12", "10", "3", "0.20", "0.50", "0.08"),
        ("12", "40", "3", "0.20", "0.50", "0.08"),
        ("6", "20", "3", "0.20", "0.50", "0.08"),
        ("24", "20", "3", "0.20", "0.50", "0.08"),
        ("12", "20", "3", "0.05", "0.50", "0.08"),
        ("12", "20", "3", "0.40", "0.50", "0.08"),
        ("12", "20", "3", "0.20", "0.50", "0"),
        ("12", "20", "3", "0.20", "0.50", "0.20"),
    }
    if actual_ga_fast_design != expected_ga_fast_design or any(
        int(row["seed_count"]) != 10
        or row["convergence_control"] != "population_diversity_mean_normalized_hamming"
        for row in ga_fast_rows
    ):
        print("ga_sensitivity_fast is not the exact nine-config OFAT design with ten seeds")
        return 1
    if any(
        parameters["greedy_start_local_search_passes"] != "1"
        or parameters["greedy_start_local_search_neighbor_window"] != "8"
        or "start-greedy-ls-1-win-8" not in row["config_id"]
        for row, parameters in zip(ga_fast_rows, ga_fast_parameters)
    ):
        print("ga_sensitivity_fast does not expose its LS1/w8 greedy warm start")
        return 1

    seed_rows = manifests["seed_factorial"]
    seed_parameters = [parse_parameters(row["parameters"]) for row in seed_rows]
    expected_seed_rules = {"smallest_snake_span", "largest_snake_span"}
    expected_paper_selections = {"maximum_location_overlap", "minimum_span_expansion"}
    paper_combinations = {
        (parameters["seed_rule"], parameters["selection_rule"])
        for parameters in seed_parameters
        if parameters["selection_rule"] in expected_paper_selections
    }
    expected_paper_combinations = {
        (seed_rule, selection_rule)
        for seed_rule in expected_seed_rules
        for selection_rule in expected_paper_selections
    }
    custom = [
        parameters
        for parameters in seed_parameters
        if parameters["selection_rule"] == "bounded_two_step_lookahead"
    ]
    ablation = [
        parameters
        for parameters in seed_parameters
        if parameters["selection_rule"] == "minimum_incremental_cost"
    ]
    if paper_combinations != expected_paper_combinations:
        print("seed_factorial does not contain the full 2x2 paper design")
        return 1
    if {parameters["seed_rule"] for parameters in custom} != expected_seed_rules or any(
        parameters.get("lookahead_width") != "8" for parameters in custom
    ):
        print("seed_factorial does not run bounded lookahead with explicit K under both seeds")
        return 1
    if {parameters["seed_rule"] for parameters in ablation} != expected_seed_rules:
        print("seed_factorial does not run the immediate-cost ablation under both seeds")
        return 1

    comparison_fast_rows = manifests["comparison_fast"]
    comparison_fast = {
        row["solver"]: (row, parse_parameters(row["parameters"]))
        for row in comparison_fast_rows
    }
    greedy_row, greedy_parameters = comparison_fast["greedy"]
    sa_row, sa_parameters = comparison_fast["simulated_annealing"]
    ga_row, ga_parameters = comparison_fast["genetic"]
    seed_row, seed_parameters = comparison_fast["seed_selection"]
    if int(greedy_row["seed_count"]) != 1 or any(
        greedy_parameters[key] != "0"
        for key in (
            "local_search_passes",
            "local_search_neighbor_window",
            "route_elimination_passes",
            "pair_rebuild_passes",
            "triple_rebuild_passes",
            "final_local_search_passes",
        )
    ):
        print("comparison_fast does not use the explicit deterministic greedy-fast baseline")
        return 1
    expected_sa_fast = {
        "cooling": "geometric",
        "neighborhood": "relocate",
        "start_strategy": "greedy",
        "greedy_start_local_search_passes": "0",
        "greedy_start_local_search_neighbor_window": "0",
        "iterations": "120",
        "initial_temperature": "60",
        "final_temperature": "0.05",
        "geometric_alpha": "0.9426276099699734",
        "lundy_beta": "0.1665277777777778",
    }
    if int(sa_row["seed_count"]) != 10 or any(
        sa_parameters[key] != value for key, value in expected_sa_fast.items()
    ):
        print("comparison_fast SA parameters or seed count are incorrect")
        return 1
    if "start-greedy-ls-0-win-0" not in sa_row["config_id"]:
        print("comparison_fast SA config_id does not expose the matched LS0/w0 start")
        return 1
    expected_ga_fast_baseline = {
        "greedy_start_local_search_passes": "1",
        "greedy_start_local_search_neighbor_window": "8",
        "population_size": "12",
        "generations": "20",
        "tournament_size": "3",
        "mutation_rate": "0.20",
        "insertion_rate": "0.50",
        "elitism_fraction": "0.08",
    }
    if int(ga_row["seed_count"]) != 10 or any(
        ga_parameters[key] != value for key, value in expected_ga_fast_baseline.items()
    ):
        print("comparison_fast GA baseline or seed count is incorrect")
        return 1
    if "start-greedy-ls-1-win-8" not in ga_row["config_id"]:
        print("comparison_fast GA config_id does not expose its LS1/w8 warm start")
        return 1
    if (
        int(seed_row["seed_count"]) != 1
        or seed_parameters["seed_rule"] != "largest_snake_span"
        or seed_parameters["selection_rule"] != "bounded_two_step_lookahead"
        or seed_parameters["lookahead_width"] != "8"
    ):
        print("comparison_fast does not use the deterministic custom lookahead configuration")
        return 1

    comparison_fast_matched_rows = manifests["comparison_fast_matched"]
    comparison_fast_matched = {
        row["solver"]: (row, parse_parameters(row["parameters"]))
        for row in comparison_fast_matched_rows
    }
    matched_greedy_row, matched_greedy_parameters = comparison_fast_matched["greedy"]
    matched_sa_row, matched_sa_parameters = comparison_fast_matched["simulated_annealing"]
    matched_ga_row, matched_ga_parameters = comparison_fast_matched["genetic"]
    matched_seed_row, matched_seed_parameters = comparison_fast_matched["seed_selection"]
    if (
        int(matched_greedy_row["seed_count"]) != 1
        or matched_greedy_parameters["local_search_passes"] != "1"
        or matched_greedy_parameters["local_search_neighbor_window"] != "8"
    ):
        print("comparison_fast_matched greedy baseline is not LS1/w8")
        return 1
    if (
        int(matched_sa_row["seed_count"]) != 10
        or matched_sa_parameters["greedy_start_local_search_passes"] != "1"
        or matched_sa_parameters["greedy_start_local_search_neighbor_window"] != "8"
        or "start-greedy-ls-1-win-8" not in matched_sa_row["config_id"]
    ):
        print("comparison_fast_matched SA warm start is not LS1/w8")
        return 1
    if (
        int(matched_ga_row["seed_count"]) != 10
        or matched_ga_parameters["greedy_start_local_search_passes"] != "1"
        or matched_ga_parameters["greedy_start_local_search_neighbor_window"] != "8"
        or "start-greedy-ls-1-win-8" not in matched_ga_row["config_id"]
    ):
        print("comparison_fast_matched GA warm start is not LS1/w8")
        return 1
    if any(
        matched_sa_parameters[key] != value
        for key, value in expected_sa_fast.items()
        if key not in {
            "greedy_start_local_search_passes",
            "greedy_start_local_search_neighbor_window",
        }
    ) or any(
        matched_ga_parameters[key] != value
        for key, value in expected_ga_fast_baseline.items()
        if key not in {
            "greedy_start_local_search_passes",
            "greedy_start_local_search_neighbor_window",
        }
    ):
        print("comparison_fast_matched changed a fast SA/GA budget")
        return 1
    if (
        int(matched_seed_row["seed_count"]) != 1
        or matched_seed_parameters["seed_rule"] != "largest_snake_span"
        or matched_seed_parameters["selection_rule"] != "bounded_two_step_lookahead"
        or matched_seed_parameters["lookahead_width"] != "8"
    ):
        print("comparison_fast_matched changed the deterministic seed-selection comparator")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
