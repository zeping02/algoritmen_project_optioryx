"""Auditable paired inference for the archived GA one-factor-at-a-time study.

The unit of analysis is a complete-instance result for one configuration and
one seed.  Wave objectives and internal solver runtimes are summed before any
paired contrast is formed.  The script never invokes a solver and refuses to
overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "ga_sensitivity_fast"
DEFAULT_OUTPUT = ROOT / "results" / "retrospective_ga_ofat_inference_final_20260807"

BASELINE_ID = "ga_pop-12_gen-20_tour-3_mut-0p20_insert-0p50_elite-0p08"
FACTOR_COLUMNS = [
    "population_size",
    "generations",
    "mutation_rate",
    "elitism_fraction",
]
CONSTANT_MANIFEST_COLUMNS = ["tournament_size", "insertion_rate"]
EXPECTED_BASELINE = {
    "population_size": "12",
    "generations": "20",
    "tournament_size": "3",
    "mutation_rate": "0.2",
    "insertion_rate": "0.5",
    "elitism_fraction": "0.08",
}
EXPECTED_LEVELS = {
    "population_size": {"6", "12", "24"},
    "generations": {"10", "20", "40"},
    "mutation_rate": {"0.05", "0.2", "0.4"},
    "elitism_fraction": {"0", "0.08", "0.2"},
}


@dataclass(frozen=True)
class EvidencePaths:
    manifest: Path
    runs: Path
    instance_runs: Path
    diversity: Path
    solutions: Path
    ga_config: Path
    main_source: Path
    solver_source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"", "-0"} else normalized


def parse_parameter_string(value: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"invalid manifest parameter token: {token!r}")
        key, raw = token.split("=", 1)
        if key in parsed:
            raise ValueError(f"duplicate manifest parameter: {key}")
        parsed[key] = canonical(raw)
    return parsed


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks columns: {', '.join(missing)}")


def read_key_value_config(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid config line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        parsed[key.strip()] = canonical(value)
    return parsed


def load_manifest(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    frame = pd.read_csv(path, dtype=str)
    require_columns(
        frame,
        [
            "suite",
            "solver",
            "config_id",
            "seed_count",
            "instance_runs",
            "wave_runs",
            "seeds",
            "parameters",
        ],
        path.name,
    )
    if len(frame) != 9 or frame["config_id"].nunique() != 9:
        raise ValueError("the GA OFAT manifest must contain exactly nine unique configs")
    if set(frame["suite"]) != {"ga_sensitivity_fast"}:
        raise ValueError("unexpected suite name in manifest")
    if set(frame["solver"]) != {"genetic"}:
        raise ValueError("manifest contains a non-GA solver")
    if set(frame["seed_count"].astype(int)) != {10}:
        raise ValueError("every config must declare ten seeds")
    if set(frame["instance_runs"].astype(int)) != {10}:
        raise ValueError("every config must declare ten instance runs")
    if set(frame["wave_runs"].astype(int)) != {120}:
        raise ValueError("every config must declare 120 wave runs")
    expected_seed_text = ";".join(str(seed) for seed in range(10))
    if set(frame["seeds"]) != {expected_seed_text}:
        raise ValueError("manifest seed sets are not exactly 0 through 9")

    parameters = {
        str(row.config_id): parse_parameter_string(row.parameters)
        for row in frame.itertuples(index=False)
    }
    if set(parameters) != set(frame["config_id"]):
        raise ValueError("manifest config IDs are not unique")
    if BASELINE_ID not in parameters:
        raise ValueError("expected GA baseline is absent")
    for config_id, values in parameters.items():
        if values.get("config_id") != config_id:
            raise ValueError(f"embedded config_id mismatch for {config_id}")
        for key in EXPECTED_BASELINE:
            if key not in values:
                raise ValueError(f"manifest config {config_id} lacks {key}")

    baseline = parameters[BASELINE_ID]
    for key, expected in EXPECTED_BASELINE.items():
        if canonical(baseline[key]) != expected:
            raise ValueError(f"baseline {key} is {baseline[key]}, expected {expected}")
    for constant in CONSTANT_MANIFEST_COLUMNS:
        observed = {values[constant] for values in parameters.values()}
        if observed != {EXPECTED_BASELINE[constant]}:
            raise ValueError(f"OFAT constant {constant} unexpectedly varies: {observed}")
    for factor, expected_levels in EXPECTED_LEVELS.items():
        observed = {values[factor] for values in parameters.values()}
        if observed != expected_levels:
            raise ValueError(f"unexpected levels for {factor}: {observed}")

    changed_factor: dict[str, str] = {}
    factor_counts = {factor: 0 for factor in FACTOR_COLUMNS}
    for config_id, values in parameters.items():
        if config_id == BASELINE_ID:
            continue
        changed = [
            factor
            for factor in FACTOR_COLUMNS
            if values[factor] != baseline[factor]
        ]
        if len(changed) != 1:
            raise ValueError(f"config {config_id} changes {len(changed)} target factors")
        changed_factor[config_id] = changed[0]
        factor_counts[changed[0]] += 1
    if set(factor_counts.values()) != {2}:
        raise ValueError(f"OFAT design is not two variants per factor: {factor_counts}")
    return frame, parameters


def load_complete_instance_data(
    paths: EvidencePaths,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs = pd.read_csv(paths.runs)
    require_columns(
        runs,
        ["solver", "config_id", "wave_id", "seed", "objective", "runtime_ms"],
        paths.runs.name,
    )
    if set(runs["solver"]) != {"genetic"}:
        raise ValueError("runs.csv contains non-GA rows")
    if runs.duplicated(["config_id", "seed", "wave_id"]).any():
        raise ValueError("runs.csv contains duplicate config/seed/wave rows")
    for column in ["objective", "runtime_ms"]:
        runs[column] = pd.to_numeric(runs[column], errors="raise")
        if not np.isfinite(runs[column]).all():
            raise ValueError(f"runs.csv has non-finite {column}")
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)
    if (runs["runtime_ms"] < 0).any():
        raise ValueError("runs.csv has negative runtimes")

    expected_configs = set(manifest["config_id"])
    if set(runs["config_id"]) != expected_configs:
        raise ValueError("runs.csv config set differs from manifest")
    counts = runs.groupby(["config_id", "seed"])["wave_id"].nunique()
    if not (counts == 12).all() or len(counts) != 90:
        raise ValueError("each config/seed must contain exactly twelve waves")
    if set(runs["seed"]) != set(range(10)):
        raise ValueError("runs.csv seed set is not exactly 0 through 9")

    totals = (
        runs.groupby(["config_id", "seed"], as_index=False)
        .agg(
            objective=("objective", "sum"),
            runtime_ms=("runtime_ms", "sum"),
            wave_count=("wave_id", "nunique"),
        )
        .sort_values(["config_id", "seed"], kind="stable")
    )

    instance = pd.read_csv(paths.instance_runs)
    require_columns(
        instance,
        ["solver", "config_id", "seed", "objective", "runtime_ms", "feasible"],
        paths.instance_runs.name,
    )
    if len(instance) != 90 or instance.duplicated(["config_id", "seed"]).any():
        raise ValueError("instance_runs.csv must contain 90 unique config/seed rows")
    if set(instance["solver"]) != {"genetic"}:
        raise ValueError("instance_runs.csv contains non-GA rows")
    feasible = instance["feasible"].astype(str).str.lower()
    if set(feasible) != {"true"}:
        raise ValueError("not all archived instance runs are feasible")
    instance["seed"] = pd.to_numeric(instance["seed"], errors="raise").astype(int)
    instance["objective"] = pd.to_numeric(instance["objective"], errors="raise")
    instance["runtime_ms"] = pd.to_numeric(instance["runtime_ms"], errors="raise")
    check = totals.merge(
        instance[["config_id", "seed", "objective", "runtime_ms"]],
        on=["config_id", "seed"],
        suffixes=("_waves", "_instance"),
        validate="one_to_one",
    )
    objective_error = np.abs(check["objective_waves"] - check["objective_instance"])
    if float(objective_error.max()) > 1e-6:
        raise ValueError("summed wave objectives disagree with instance objectives")
    overhead = check["runtime_ms_instance"] - check["runtime_ms_waves"]
    validation = {
        "wave_rows": int(len(runs)),
        "complete_instance_rows": int(len(totals)),
        "all_instance_runs_feasible": True,
        "max_abs_objective_sum_error": float(objective_error.max()),
        "instance_minus_summed_wave_runtime_ms": {
            "minimum": float(overhead.min()),
            "maximum": float(overhead.max()),
            "mean": float(overhead.mean()),
        },
        "runtime_used_for_inference": "sum_of_internal_wave_solver_runtime_ms",
    }
    return totals, validation


def load_diversity(
    path: Path,
    parameters: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        [
            "solver",
            "config_id",
            "wave_id",
            "seed",
            "generation",
            "population_diversity_mean_normalized_hamming",
        ],
        path.name,
    )
    if set(frame["solver"]) != {"genetic"}:
        raise ValueError("ga_diversity.csv contains non-GA rows")
    if frame.duplicated(["config_id", "seed", "wave_id", "generation"]).any():
        raise ValueError("ga_diversity.csv contains duplicate convergence rows")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["generation"] = pd.to_numeric(frame["generation"], errors="raise").astype(int)
    value_column = "population_diversity_mean_normalized_hamming"
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    if not frame[value_column].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("population diversity is outside [0, 1]")
    if set(frame["config_id"]) != set(parameters):
        raise ValueError("diversity config set differs from manifest")

    per_seed_records: list[dict[str, Any]] = []
    for config_id, config_frame in frame.groupby("config_id", sort=True):
        final_generation = int(parameters[config_id]["generations"])
        for (seed, wave_id), wave_frame in config_frame.groupby(
            ["seed", "wave_id"], sort=True
        ):
            observed = set(wave_frame["generation"])
            expected = set(range(final_generation + 1))
            if observed != expected or len(wave_frame) != final_generation + 1:
                raise ValueError(
                    f"incomplete diversity trajectory for {config_id}, seed {seed}, wave {wave_id}"
                )
        initial = config_frame[config_frame["generation"] == 0]
        final = config_frame[config_frame["generation"] == final_generation]
        initial_counts = initial.groupby("seed")["wave_id"].nunique()
        final_counts = final.groupby("seed")["wave_id"].nunique()
        if not (initial_counts == 12).all() or not (final_counts == 12).all():
            raise ValueError(f"initial/final diversity waves are incomplete for {config_id}")
        initial_means = initial.groupby("seed")[value_column].mean()
        final_means = final.groupby("seed")[value_column].mean()
        if set(initial_means.index) != set(range(10)) or set(final_means.index) != set(range(10)):
            raise ValueError(f"diversity seed set is incomplete for {config_id}")
        for seed in range(10):
            initial_value = float(initial_means.loc[seed])
            final_value = float(final_means.loc[seed])
            per_seed_records.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "final_generation": final_generation,
                    "initial_diversity": initial_value,
                    "final_diversity": final_value,
                    "diversity_change": final_value - initial_value,
                    "diversity_retained_percent": (
                        100.0 * final_value / initial_value
                        if initial_value != 0.0
                        else math.nan
                    ),
                }
            )
    per_seed = pd.DataFrame(per_seed_records).sort_values(
        ["config_id", "seed"], kind="stable"
    )
    validation = {
        "trajectory_rows": int(len(frame)),
        "config_seed_summaries": int(len(per_seed)),
        "metric": "mean normalized positional Hamming distance over chromosome pairs",
        "aggregation": "mean_across_12_waves_within_seed_at_generation_0_and_final",
        "range_check": "all values in [0, 1]",
    }
    return per_seed, validation


def validate_solution_metadata(
    solution_dir: Path,
    parameters: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, Any]]:
    files = sorted(solution_dir.glob("*.json"))
    if len(files) != 90:
        raise ValueError(f"expected 90 archived GA solution files, found {len(files)}")
    hashes: dict[str, str] = {}
    metadata_records: list[dict[str, str]] = []
    root_keys: set[tuple[str, int]] = set()
    for path in files:
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        config_id = str(payload.get("config_id", ""))
        seed = int(payload.get("seed", -1))
        if config_id not in parameters or seed not in range(10):
            raise ValueError(f"unexpected config/seed in {path.name}")
        key = (config_id, seed)
        if key in root_keys:
            raise ValueError(f"duplicate archived solution for {config_id}, seed {seed}")
        root_keys.add(key)
        if not bool(payload.get("feasible")) or payload.get("validation_errors"):
            raise ValueError(f"archived solution is not valid: {path.name}")
        waves = payload.get("waves")
        if not isinstance(waves, list) or len(waves) != 12:
            raise ValueError(f"archived solution lacks twelve waves: {path.name}")
        for wave in waves:
            metadata = {str(k): canonical(v) for k, v in wave.get("metadata", {}).items()}
            required = {
                "config_id",
                "crossover",
                "elitism_count",
                "elitism_fraction",
                "fitness",
                "generations",
                "insertion_rate",
                "mutation_rate",
                "population_size",
                "representation",
                "tournament_size",
            }
            if not required.issubset(metadata):
                raise ValueError(f"incomplete GA metadata in {path.name}")
            expected = parameters[config_id]
            checks = {
                "config_id": config_id,
                "crossover": "order_crossover",
                "elitism_fraction": expected["elitism_fraction"],
                "fitness": "decoded_total_route_cost",
                "generations": expected["generations"],
                "insertion_rate": expected["insertion_rate"],
                "mutation_rate": expected["mutation_rate"],
                "population_size": expected["population_size"],
                "representation": "order_permutation",
                "tournament_size": expected["tournament_size"],
            }
            numeric_fields = {
                "elitism_fraction",
                "generations",
                "insertion_rate",
                "mutation_rate",
                "population_size",
                "tournament_size",
            }
            for field, expected_value in checks.items():
                if field in numeric_fields:
                    matches = math.isclose(
                        float(metadata[field]),
                        float(expected_value),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                else:
                    matches = metadata[field] == canonical(expected_value)
                if not matches:
                    raise ValueError(
                        f"metadata {field} mismatch in {path.name}: "
                        f"{metadata[field]} != {expected_value}"
                    )
            for field in numeric_fields:
                metadata[field] = canonical(checks[field])
            metadata_records.append(
                {
                    "config_id": config_id,
                    "seed": str(seed),
                    "wave_id": str(wave.get("wave_id", "")),
                    **metadata,
                }
            )
    expected_keys = {(config_id, seed) for config_id in parameters for seed in range(10)}
    if root_keys != expected_keys:
        raise ValueError("archived solution set is incomplete")
    canonical_metadata = json.dumps(
        sorted(metadata_records, key=lambda record: (record["config_id"], record["seed"], record["wave_id"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    evidence = {
        "solution_files": len(files),
        "wave_metadata_records": len(metadata_records),
        "canonical_metadata_sha256": hashlib.sha256(canonical_metadata).hexdigest(),
        "crossover_values": sorted({record["crossover"] for record in metadata_records}),
        "tournament_size_values": sorted({record["tournament_size"] for record in metadata_records}),
        "insertion_rate_values": sorted({record["insertion_rate"] for record in metadata_records}),
        "representation_values": sorted({record["representation"] for record in metadata_records}),
        "fitness_values": sorted({record["fitness"] for record in metadata_records}),
        "elitism_fraction_values": sorted({record["elitism_fraction"] for record in metadata_records}),
        "elitism_count_values_by_config": {
            config_id: sorted(
                {
                    record["elitism_count"]
                    for record in metadata_records
                    if record["config_id"] == config_id
                }
            )
            for config_id in sorted(parameters)
        },
    }
    return hashes, evidence


def validate_source_and_config(paths: EvidencePaths) -> dict[str, Any]:
    fast_config = read_key_value_config(paths.ga_config)
    for key, expected in EXPECTED_BASELINE.items():
        if fast_config.get(key) != expected:
            raise ValueError(
                f"{paths.ga_config.name} baseline {key} is {fast_config.get(key)}, expected {expected}"
            )
    main_text = paths.main_source.read_text(encoding="utf-8")
    solver_text = paths.solver_source.read_text(encoding="utf-8")
    main_tokens = [
        'specs.push_back({"genetic", make_ga_config(12, 20, "0.20", "0.08"), seeds});',
        'specs.push_back({"genetic", make_ga_config(12, 10, "0.20", "0.08"), seeds});',
        'specs.push_back({"genetic", make_ga_config(12, 40, "0.20", "0.08"), seeds});',
        'specs.push_back({"genetic", make_ga_config(6, 20, "0.20", "0.08"), seeds});',
        'specs.push_back({"genetic", make_ga_config(24, 20, "0.20", "0.08"), seeds});',
        'specs.push_back({"genetic", make_ga_config(12, 20, "0.05", "0.08"), seeds});',
        'specs.push_back({"genetic", make_ga_config(12, 20, "0.40", "0.08"), seeds});',
        'specs.push_back({"genetic", make_ga_config(12, 20, "0.20", "0"), seeds});',
        'specs.push_back({"genetic", make_ga_config(12, 20, "0.20", "0.20"), seeds});',
        '{"tournament_size", "3"}',
        '{"insertion_rate", "0.50"}',
    ]
    missing_main = [token for token in main_tokens if token not in main_text]
    if missing_main:
        raise ValueError(f"current main source lacks expected GA suite tokens: {missing_main}")
    solver_tokens = [
        "std::vector<int> order_crossover(",
        'result.metadata["crossover"] = "order_crossover";',
        "if (probability(rng) < insertion_rate)",
        "std::swap(permutation[first], permutation[second]);",
        "tournament_pick(population, std::max(2, config.tournament_size), rng)",
        "static_cast<double>(population_size) * config.elitism_fraction",
    ]
    missing_solver = [token for token in solver_tokens if token not in solver_text]
    if missing_solver:
        raise ValueError(f"current GA source lacks expected implementation tokens: {missing_solver}")
    return {
        "archived_manifest_is_authoritative_for_run_parameters": True,
        "matching_baseline_config_file": paths.ga_config.relative_to(ROOT).as_posix(),
        "source_confirms": {
            "crossover": "order crossover (OX)",
            "selection": "tournament selection, size 3 in this suite",
            "mutation_operator_mix": (
                "conditional on mutation, insertion with probability 0.50 and swap otherwise"
            ),
            "elitism": "round(population_size * elitism_fraction), clamped to [0, population_size]",
        },
    }


def sample_stats(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if clean.size < 2 or not np.isfinite(clean).all():
        raise ValueError("Student-t inference requires at least two finite values")
    n = int(clean.size)
    mean = float(np.mean(clean))
    sd = float(np.std(clean, ddof=1))
    se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, n - 1))
    half_width = critical * se
    if sd == 0.0:
        if mean == 0.0:
            t_statistic, p_value = 0.0, 1.0
        else:
            t_statistic, p_value = math.copysign(math.inf, mean), 0.0
    else:
        test = stats.ttest_1samp(clean, popmean=0.0)
        t_statistic = float(test.statistic)
        p_value = float(test.pvalue)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "t_statistic": t_statistic,
        "p_raw": p_value,
    }


def descriptive_stats(values: pd.Series, prefix: str) -> dict[str, float | int]:
    stats_record = sample_stats(values)
    return {
        f"{prefix}_n": stats_record["n"],
        f"{prefix}_mean": stats_record["mean"],
        f"{prefix}_sd": stats_record["sd"],
        f"{prefix}_ci95_low": stats_record["ci95_low"],
        f"{prefix}_ci95_high": stats_record["ci95_high"],
    }


def add_holm(frame: pd.DataFrame, raw_column: str, output_prefix: str) -> pd.DataFrame:
    if len(frame) != 8:
        raise ValueError("Holm correction must cover exactly eight variant contrasts")
    p_values = pd.to_numeric(frame[raw_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(p_values).all() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError(f"invalid p-values in {raw_column}")
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    result = frame.copy()
    result[f"{output_prefix}_p_holm_8"] = adjusted
    result[f"{output_prefix}_holm_reject_0p05"] = adjusted < 0.05
    return result


def build_paired_data(
    totals: pd.DataFrame,
    diversity: pd.DataFrame,
    parameters: dict[str, dict[str, str]],
) -> pd.DataFrame:
    baseline = totals[totals["config_id"] == BASELINE_ID].rename(
        columns={
            "objective": "baseline_objective",
            "runtime_ms": "baseline_runtime_ms",
        }
    )[["seed", "baseline_objective", "baseline_runtime_ms"]]
    baseline_diversity = diversity[diversity["config_id"] == BASELINE_ID].rename(
        columns={
            "initial_diversity": "baseline_initial_diversity",
            "final_diversity": "baseline_final_diversity",
            "diversity_change": "baseline_diversity_change",
            "diversity_retained_percent": "baseline_diversity_retained_percent",
        }
    )[
        [
            "seed",
            "baseline_initial_diversity",
            "baseline_final_diversity",
            "baseline_diversity_change",
            "baseline_diversity_retained_percent",
        ]
    ]
    baseline_parameters = parameters[BASELINE_ID]
    records: list[dict[str, Any]] = []
    for config_id in sorted(set(totals["config_id"]) - {BASELINE_ID}):
        changed = [
            factor
            for factor in FACTOR_COLUMNS
            if parameters[config_id][factor] != baseline_parameters[factor]
        ]
        if len(changed) != 1:
            raise ValueError(f"invalid variant {config_id}")
        factor = changed[0]
        variant = totals[totals["config_id"] == config_id][
            ["seed", "objective", "runtime_ms"]
        ].merge(
            diversity[diversity["config_id"] == config_id][
                [
                    "seed",
                    "initial_diversity",
                    "final_diversity",
                    "diversity_change",
                    "diversity_retained_percent",
                ]
            ],
            on="seed",
            validate="one_to_one",
        )
        paired = (
            variant.merge(baseline, on="seed", validate="one_to_one")
            .merge(baseline_diversity, on="seed", validate="one_to_one")
            .sort_values("seed", kind="stable")
        )
        if set(paired["seed"]) != set(range(10)) or len(paired) != 10:
            raise ValueError(f"variant {config_id} is not paired on all ten seeds")
        for row in paired.itertuples(index=False):
            objective_delta = row.objective - row.baseline_objective
            runtime_delta = row.runtime_ms - row.baseline_runtime_ms
            records.append(
                {
                    "config_id": config_id,
                    "baseline_config_id": BASELINE_ID,
                    "changed_factor": factor,
                    "baseline_value": baseline_parameters[factor],
                    "variant_value": parameters[config_id][factor],
                    "seed": int(row.seed),
                    "variant_objective": float(row.objective),
                    "baseline_objective": float(row.baseline_objective),
                    "objective_delta_variant_minus_baseline": float(objective_delta),
                    "objective_delta_percent": float(
                        100.0 * objective_delta / row.baseline_objective
                    ),
                    "variant_runtime_ms": float(row.runtime_ms),
                    "baseline_runtime_ms": float(row.baseline_runtime_ms),
                    "runtime_delta_ms_variant_minus_baseline": float(runtime_delta),
                    "runtime_delta_percent": float(
                        100.0 * runtime_delta / row.baseline_runtime_ms
                    ),
                    "variant_initial_diversity": float(row.initial_diversity),
                    "baseline_initial_diversity": float(row.baseline_initial_diversity),
                    "variant_final_diversity": float(row.final_diversity),
                    "baseline_final_diversity": float(row.baseline_final_diversity),
                    "final_diversity_delta_variant_minus_baseline": float(
                        row.final_diversity - row.baseline_final_diversity
                    ),
                    "variant_diversity_change_final_minus_initial": float(
                        row.diversity_change
                    ),
                    "baseline_diversity_change_final_minus_initial": float(
                        row.baseline_diversity_change
                    ),
                    "variant_diversity_retained_percent": float(
                        row.diversity_retained_percent
                    ),
                    "baseline_diversity_retained_percent": float(
                        row.baseline_diversity_retained_percent
                    ),
                }
            )
    paired_frame = pd.DataFrame(records).sort_values(
        ["changed_factor", "variant_value", "seed"], kind="stable"
    )
    if len(paired_frame) != 80:
        raise ValueError("expected exactly 80 paired variant/seed rows")
    return paired_frame


def build_contrast_summary(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_id, frame in paired.groupby("config_id", sort=False):
        first = frame.iloc[0]
        objective = sample_stats(frame["objective_delta_variant_minus_baseline"])
        runtime = sample_stats(frame["runtime_delta_ms_variant_minus_baseline"])
        runtime_percent = sample_stats(frame["runtime_delta_percent"])
        diversity = sample_stats(
            frame["final_diversity_delta_variant_minus_baseline"]
        )
        records.append(
            {
                "config_id": config_id,
                "baseline_config_id": BASELINE_ID,
                "changed_factor": first["changed_factor"],
                "baseline_value": first["baseline_value"],
                "variant_value": first["variant_value"],
                "paired_seed_count": int(frame["seed"].nunique()),
                "variant_objective_mean": float(frame["variant_objective"].mean()),
                "baseline_objective_mean": float(frame["baseline_objective"].mean()),
                **{f"objective_delta_{key}": value for key, value in objective.items()},
                "variant_runtime_ms_mean": float(frame["variant_runtime_ms"].mean()),
                "baseline_runtime_ms_mean": float(frame["baseline_runtime_ms"].mean()),
                **{f"runtime_delta_ms_{key}": value for key, value in runtime.items()},
                **{
                    f"runtime_delta_percent_{key}": value
                    for key, value in runtime_percent.items()
                    if key not in {"t_statistic", "p_raw"}
                },
                "variant_final_diversity_mean": float(
                    frame["variant_final_diversity"].mean()
                ),
                "baseline_final_diversity_mean": float(
                    frame["baseline_final_diversity"].mean()
                ),
                **{f"final_diversity_delta_{key}": value for key, value in diversity.items()},
            }
        )
    summary = pd.DataFrame(records).sort_values(
        ["changed_factor", "variant_value"], kind="stable"
    )
    summary = add_holm(summary, "objective_delta_p_raw", "objective")
    summary = add_holm(summary, "runtime_delta_ms_p_raw", "runtime")
    summary = add_holm(summary, "final_diversity_delta_p_raw", "final_diversity")
    return summary


def build_diversity_summary(
    diversity: pd.DataFrame,
    parameters: dict[str, dict[str, str]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_id, frame in diversity.groupby("config_id", sort=True):
        record: dict[str, Any] = {
            "config_id": config_id,
            "is_baseline": config_id == BASELINE_ID,
            "population_size": parameters[config_id]["population_size"],
            "generations": parameters[config_id]["generations"],
            "mutation_rate": parameters[config_id]["mutation_rate"],
            "elitism_fraction": parameters[config_id]["elitism_fraction"],
            "tournament_size": parameters[config_id]["tournament_size"],
            "insertion_rate": parameters[config_id]["insertion_rate"],
        }
        record.update(descriptive_stats(frame["initial_diversity"], "initial_diversity"))
        record.update(descriptive_stats(frame["final_diversity"], "final_diversity"))
        record.update(descriptive_stats(frame["diversity_change"], "diversity_change"))
        record.update(
            descriptive_stats(
                frame["diversity_retained_percent"],
                "diversity_retained_percent",
            )
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["is_baseline", "config_id"], ascending=[False, True], kind="stable"
    )


def fixed_conditions_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition": "representation_and_decoder",
                "archived_value": "order permutation; DP-decoded total route cost",
                "status": "fixed across all nine configurations",
                "paper_scope": "all OFAT conclusions are conditional on this representation and decoder",
            },
            {
                "condition": "crossover",
                "archived_value": "order crossover (OX)",
                "status": "fixed across all nine configurations",
                "paper_scope": "no comparison with another crossover operator was performed",
            },
            {
                "condition": "tournament_size",
                "archived_value": "3",
                "status": "fixed across all nine configurations",
                "paper_scope": "effects are conditional on tournament selection with size 3",
            },
            {
                "condition": "mutation_operator_mix",
                "archived_value": "50% insertion and 50% swap, conditional on mutation",
                "status": "fixed through insertion_rate=0.50",
                "paper_scope": "mutation-rate effects do not compare operator mixtures",
            },
            {
                "condition": "elitism_fraction",
                "archived_value": "baseline 0.08; variants 0 and 0.20",
                "status": "fixed at 0.08 outside the two elitism contrasts",
                "paper_scope": "generation and mutation-rate effects are conditional on elitism 0.08",
            },
            {
                "condition": "effective_elite_count",
                "archived_value": "population 6: 0; population 12 at 0.08: 1; population 24: 2",
                "status": "derived by rounding population_size times elitism_fraction",
                "paper_scope": "the population-size contrast also changes the absolute number of elites",
            },
            {
                "condition": "paired_seeds",
                "archived_value": "0 through 9",
                "status": "same seed labels for every configuration",
                "paper_scope": "contrasts are paired by seed on the complete instance",
            },
        ]
    )


def fnum(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}".replace(",", "\u202f")


def make_report(summary: pd.DataFrame, diversity: pd.DataFrame) -> str:
    factor_labels = {
        "population_size": "populatie",
        "generations": "generaties",
        "mutation_rate": "mutatiekans",
        "elitism_fraction": "elitisme",
    }
    lines = [
        "# Gekoppelde inferentie voor de GA-OFAT-studie",
        "",
        "De archiefdata bevat negen configuraties, tien gedeelde seeds per configuratie en twaalf waves per volledige run. "
        "Elk contrast is berekend als variant min basis. Voor de doelfunctie betekent een negatieve waarde dus een verbetering. "
        "De statistische eenheid is een volledige instance na sommering over alle waves.",
        "",
        "## Kwaliteit en uitvoeringstijd",
        "",
        "| factor | variant (basis) | $\\Delta D$ gemiddeld [95%-BI] | ruwe p | Holm p | $\\Delta t$ in ms [95%-BI] en relatief | Holm p tijd |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        label = factor_labels.get(row.changed_factor, row.changed_factor)
        objective_ci = (
            f"{fnum(row.objective_delta_mean, 1)} "
            f"[{fnum(row.objective_delta_ci95_low, 1)}, {fnum(row.objective_delta_ci95_high, 1)}]"
        )
        runtime_ci = (
            f"{fnum(row.runtime_delta_ms_mean, 1)} "
            f"[{fnum(row.runtime_delta_ms_ci95_low, 1)}, {fnum(row.runtime_delta_ms_ci95_high, 1)}] "
            f"({row.runtime_delta_percent_mean:+.2f}%)"
        )
        lines.append(
            f"| {label} | {row.variant_value} ({row.baseline_value}) | {objective_ci} | "
            f"{row.objective_delta_p_raw:.6g} | {row.objective_p_holm_8:.6g} | "
            f"{runtime_ci} | {row.runtime_p_holm_8:.6g} |"
        )
    significant_quality = summary[summary["objective_holm_reject_0p05"]]
    if significant_quality.empty:
        quality_statement = (
            "Na Holm-correctie over de acht kwaliteitscontrasten verschilt geen variant significant van de basis bij 5%."
        )
    else:
        quality_statement = (
            "Na Holm-correctie blijven de volgende kwaliteitscontrasten significant: "
            + ", ".join(
                f"{factor_labels.get(row.changed_factor, row.changed_factor)}={row.variant_value}"
                for row in significant_quality.itertuples(index=False)
            )
            + "."
        )
    lines.extend(["", quality_statement])

    baseline_diversity = diversity[diversity["is_baseline"]].iloc[0]
    significant_diversity = summary[summary["final_diversity_holm_reject_0p05"]]
    if significant_diversity.empty:
        diversity_statement = (
            "Na Holm-correctie verschilt geen einddiversiteitscontrast significant van de basis."
        )
    else:
        diversity_statement = (
            "Na Holm-correctie verschillen de volgende einddiversiteiten significant van de basis: "
            + ", ".join(
                f"{factor_labels.get(row.changed_factor, row.changed_factor)}={row.variant_value} "
                f"($\\Delta H={row.final_diversity_delta_mean:+.3f}$)"
                for row in significant_diversity.itertuples(index=False)
            )
            + "."
        )
    lines.extend(
        [
            "",
            "## Diversiteit",
            "",
            f"Bij de basisconfiguratie daalt de gemiddelde genormaliseerde positionele Hammingdiversiteit "
            f"van {baseline_diversity.initial_diversity_mean:.3f} bij generatie 0 naar "
            f"{baseline_diversity.final_diversity_mean:.3f} bij generatie 20. Daarmee blijft gemiddeld "
            f"{baseline_diversity.diversity_retained_percent_mean:.1f}% van de initiële diversiteit over.",
            "",
            "De einddiversiteit is per seed eerst over de twaalf waves gemiddeld. Voor elk variant-basiscontrast staat "
            "de volledige gekoppelde inferentie in `ga_ofat_contrasts.csv`. De tabel `ga_ofat_diversity_summary.csv` "
            "bevat daarnaast de beginwaarde, eindwaarde, verandering en behouden fractie voor alle negen configuraties.",
            "",
            diversity_statement,
            "",
            "## Afbakening",
            "",
            "Deze OFAT-resultaten zijn conditioneel op order crossover, tournamentselectie met grootte 3 en een "
            "50/50-mix van insertion- en swapmutatie. Voor populatie-, generatie- en mutatiekanscontrasten bleef "
            "het elitisme 0,08. Alleen de twee elitismecontrasten wijzigden die fractie naar 0 of 0,20. "
            "Door afronding veranderde bij de populatievarianten wel het absolute aantal elites van 1 bij populatie 12 "
            "naar 0 bij populatie 6 en 2 bij populatie 24. Het populatie-effect kan daarom niet los van dat afgeleide aantal worden gelezen. "
            "Interacties tussen factoren zijn niet identificeerbaar uit een OFAT-opzet. De runtimevolgorde werd in "
            "de oorspronkelijke suite niet gerandomiseerd, waardoor de tijdscontrasten ook systeem- en volgorde-effecten kunnen bevatten.",
            "",
            "De ruwe p-waarden zijn tweezijdige gekoppelde Student-t-toetsen. Holm-correctie is afzonderlijk "
            "toegepast op drie families van telkens acht vooraf bepaalde contrasten: doelfunctie, interne solverruntime "
            "en einddiversiteit.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = EvidencePaths(
        manifest=input_dir / "suite_manifest.csv",
        runs=input_dir / "runs.csv",
        instance_runs=input_dir / "instance_runs.csv",
        diversity=input_dir / "ga_diversity.csv",
        solutions=input_dir / "solutions",
        ga_config=ROOT / "configs" / "ga_competition_fast.cfg",
        main_source=ROOT / "src" / "main.cpp",
        solver_source=ROOT / "src" / "algorithms" / "genetic_solver.cpp",
    )
    required_paths = [
        paths.manifest,
        paths.runs,
        paths.instance_runs,
        paths.diversity,
        paths.ga_config,
        paths.main_source,
        paths.solver_source,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing or not paths.solutions.is_dir():
        raise FileNotFoundError(f"missing required evidence: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    manifest, parameters = load_manifest(paths.manifest)
    totals, result_validation = load_complete_instance_data(paths, manifest)
    diversity_per_seed, diversity_validation = load_diversity(paths.diversity, parameters)
    solution_hashes, solution_evidence = validate_solution_metadata(
        paths.solutions, parameters
    )
    source_evidence = validate_source_and_config(paths)

    paired = build_paired_data(totals, diversity_per_seed, parameters)
    contrast_summary = build_contrast_summary(paired)
    diversity_summary = build_diversity_summary(diversity_per_seed, parameters)
    fixed_conditions = fixed_conditions_table()

    output_dir.mkdir(parents=True, exist_ok=False)
    paired_path = output_dir / "ga_ofat_paired_seed_differences.csv"
    contrast_path = output_dir / "ga_ofat_contrasts.csv"
    diversity_path = output_dir / "ga_ofat_diversity_summary.csv"
    conditions_path = output_dir / "ga_ofat_fixed_conditions.csv"
    report_path = output_dir / "ga_ofat_report.md"
    paired.to_csv(paired_path, index=False)
    contrast_summary.to_csv(contrast_path, index=False)
    diversity_summary.to_csv(diversity_path, index=False)
    fixed_conditions.to_csv(conditions_path, index=False)
    report_path.write_text(make_report(contrast_summary, diversity_summary), encoding="utf-8")

    primary_inputs = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in required_paths
    }
    script_path = Path(__file__).resolve()
    primary_inputs[script_path.relative_to(ROOT).as_posix()] = sha256_file(script_path)
    artifacts = [paired_path, contrast_path, diversity_path, conditions_path, report_path]
    protocol = {
        "analysis": "retrospective_ga_ofat_paired_inference",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": input_dir.relative_to(ROOT).as_posix(),
        "output_directory": output_dir.relative_to(ROOT).as_posix(),
        "statistical_unit": "complete_instance_per_config_seed",
        "pairing_key": "seed",
        "paired_seeds": list(range(10)),
        "wave_aggregation": "sum objective and internal solver runtime over 12 waves before inference",
        "contrast_sign": "variant_minus_baseline; negative objective delta is better",
        "confidence_interval": "two-sided 95% Student-t interval for paired mean, df=9",
        "test": "two-sided paired Student-t test, implemented as one-sample t test of paired differences",
        "multiplicity": {
            "method": "Holm step-down family-wise error correction",
            "alpha": 0.05,
            "families": {
                "objective": 8,
                "internal_solver_runtime_ms": 8,
                "final_diversity": 8,
            },
        },
        "baseline_config_id": BASELINE_ID,
        "target_factors": FACTOR_COLUMNS,
        "design": {
            "configurations": 9,
            "variants": 8,
            "variants_per_factor": 2,
            "manifest_parameters": parameters,
        },
        "validation": {
            "result_data": result_validation,
            "diversity_data": diversity_validation,
            "solution_metadata": solution_evidence,
            "source_and_config": source_evidence,
        },
        "limitations": [
            "OFAT estimates are conditional and do not identify factor interactions.",
            "At elitism fraction 0.08, changing population size also changes the rounded elite count (0, 1, or 2), so that derived mechanism is part of the population contrast.",
            "The archived suite was not randomized across configurations, so runtime contrasts may include order and system-load effects.",
            "Seed labels are paired, but changing a hyperparameter changes subsequent random-number consumption.",
            "The ten-seed Student-t analysis assumes the paired differences are adequate replicates for the reported mean contrasts.",
        ],
        "sha256": {
            "primary_inputs_and_evidence": primary_inputs,
            "archived_solution_files": solution_hashes,
            "output_artifacts_excluding_protocol": {
                path.name: sha256_file(path) for path in artifacts
            },
        },
        "artifacts": [path.name for path in artifacts] + ["analysis_protocol.json"],
    }
    protocol_path = output_dir / "analysis_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(artifacts) + 1} artifacts to {output_dir}")
    print(contrast_summary[
        [
            "changed_factor",
            "variant_value",
            "objective_delta_mean",
            "objective_delta_ci95_low",
            "objective_delta_ci95_high",
            "objective_delta_p_raw",
            "objective_p_holm_8",
            "runtime_delta_percent_mean",
            "runtime_p_holm_8",
            "final_diversity_delta_mean",
            "final_diversity_p_holm_8",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
