"""Auditable paired inference for a completed 2x2x2 SA factorial suite.

The statistical unit is one complete instance for one configuration and seed.
Wave results are summed before the seven factorial contrasts are formed.  This
script never invokes a solver and refuses to overwrite an existing directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


FACTORS = ("cooling", "neighborhood", "start_strategy")
LEVELS = {
    "cooling": ("geometric", "lundy_mees"),
    "neighborhood": ("relocate", "swap"),
    "start_strategy": ("greedy", "randomized"),
}
REFERENCES = {factor: levels[0] for factor, levels in LEVELS.items()}
ALTERNATIVES = {factor: levels[1] for factor, levels in LEVELS.items()}
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_WAVES = 12
EXPECTED_CONFIGS = 8

MANIFEST_COLUMNS = {
    "suite",
    "solver",
    "config_id",
    "seed_count",
    "instance_runs",
    "wave_runs",
    "seeds",
    "parameters",
}
RUN_COLUMNS = {
    "solver",
    "config_id",
    "wave_id",
    "seed",
    "order_count",
    "objective",
    "runtime_ms",
}
INSTANCE_COLUMNS = {
    "solver",
    "config_id",
    "seed",
    "order_count",
    "objective",
    "runtime_ms",
    "feasible",
}
FACTORIAL_FILES = {
    "sa_config_summary.csv",
    "sa_main_effects.csv",
    "sa_interactions.csv",
    "analysis_protocol.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Completed suite directory containing runs.csv and factorial_analysis/.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="New directory for the paired inference artifacts.",
    )
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required input: {path}")


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks columns: {', '.join(missing)}")


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


def parse_parameters(value: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_token in str(value).split(";"):
        token = raw_token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"invalid manifest parameter token: {token!r}")
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key or key in parsed:
            raise ValueError(f"duplicate or empty manifest parameter: {key!r}")
        parsed[key] = canonical(raw_value)
    return parsed


def parse_seed_set(value: Any, context: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(value).split(";") if token.strip()]
    try:
        seeds = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f"invalid seed set for {context}: {value!r}") from exc
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds for {context}")
    return seeds


def finite_numeric(
    frame: pd.DataFrame,
    column: str,
    source: str,
    *,
    nonnegative: bool = False,
    integer: bool = False,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite {column}")
    if nonnegative and (values < 0).any():
        raise ValueError(f"{source} contains negative {column}")
    if integer and not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{source} contains non-integer {column}")
    return values.astype(int) if integer else values.astype(float)


def load_manifest(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]], str]:
    frame = pd.read_csv(path, dtype=str)
    require_columns(frame, MANIFEST_COLUMNS, path.name)
    require(
        len(frame) == EXPECTED_CONFIGS
        and frame["config_id"].nunique() == EXPECTED_CONFIGS,
        "suite_manifest.csv must contain exactly eight unique configurations",
    )
    suites = set(frame["suite"].dropna().astype(str))
    require(len(suites) == 1 and "" not in suites, "manifest must declare one suite name")
    suite_name = next(iter(suites))
    require(
        set(frame["solver"].astype(str)) == {"simulated_annealing"},
        "manifest contains a non-SA solver",
    )

    seed_counts = finite_numeric(frame, "seed_count", path.name, integer=True)
    instance_counts = finite_numeric(frame, "instance_runs", path.name, integer=True)
    wave_counts = finite_numeric(frame, "wave_runs", path.name, integer=True)
    require(set(seed_counts) == {10}, "every factorial cell must declare ten seeds")
    require(set(instance_counts) == {10}, "every factorial cell must declare ten instance runs")
    require(
        set(wave_counts) == {10 * EXPECTED_WAVES},
        f"every factorial cell must declare {10 * EXPECTED_WAVES} wave runs",
    )
    for row in frame.itertuples(index=False):
        require(
            parse_seed_set(row.seeds, str(row.config_id)) == EXPECTED_SEEDS,
            f"manifest seeds for {row.config_id} are not exactly 0 through 9 in order",
        )

    parameters: dict[str, dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        config_id = str(row.config_id)
        parsed = parse_parameters(row.parameters)
        require(
            parsed.get("config_id") == config_id,
            f"embedded config_id differs for {config_id}",
        )
        for factor in FACTORS:
            require(factor in parsed, f"manifest config {config_id} lacks factor {factor}")
        parameters[config_id] = parsed

    parameter_key_sets = {frozenset(values) for values in parameters.values()}
    require(len(parameter_key_sets) == 1, "factorial cells do not expose the same parameters")
    varying_allowed = {"config_id", *FACTORS}
    for key in next(iter(parameter_key_sets)) - varying_allowed:
        observed = {values[key] for values in parameters.values()}
        require(
            len(observed) == 1,
            f"non-factor parameter {key!r} unexpectedly varies across cells: {sorted(observed)}",
        )

    observed_cells: set[tuple[str, str, str]] = set()
    for config_id, values in parameters.items():
        cell = tuple(values[factor] for factor in FACTORS)
        for factor, level in zip(FACTORS, cell):
            require(
                level in LEVELS[factor],
                f"unexpected {factor} level {level!r} in {config_id}",
            )
        require(cell not in observed_cells, f"duplicate factorial cell: {cell}")
        observed_cells.add(cell)
    expected_cells = set(itertools.product(*(LEVELS[factor] for factor in FACTORS)))
    require(observed_cells == expected_cells, "manifest is not the complete expected 2x2x2 design")

    expanded = frame.copy()
    for factor in FACTORS:
        expanded[factor] = expanded["config_id"].map(
            {config_id: values[factor] for config_id, values in parameters.items()}
        )
    return expanded, parameters, suite_name


def load_runs(path: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    runs = pd.read_csv(path, dtype={"config_id": str, "wave_id": str})
    require_columns(runs, RUN_COLUMNS, path.name)
    require(
        len(runs) == EXPECTED_CONFIGS * len(EXPECTED_SEEDS) * EXPECTED_WAVES,
        "runs.csv must contain exactly 960 config/seed/wave rows",
    )
    require(
        set(runs["solver"].astype(str)) == {"simulated_annealing"},
        "runs.csv contains a non-SA solver",
    )
    expected_configs = set(manifest["config_id"].astype(str))
    require(set(runs["config_id"].astype(str)) == expected_configs, "runs/manifest config sets differ")
    require(
        not runs.duplicated(["config_id", "seed", "wave_id"]).any(),
        "runs.csv contains duplicate config/seed/wave rows",
    )
    runs["seed"] = finite_numeric(runs, "seed", path.name, integer=True)
    runs["order_count"] = finite_numeric(
        runs, "order_count", path.name, nonnegative=True, integer=True
    )
    runs["objective"] = finite_numeric(runs, "objective", path.name, nonnegative=True)
    runs["runtime_ms"] = finite_numeric(runs, "runtime_ms", path.name, nonnegative=True)
    require((runs["order_count"] > 0).all(), "runs.csv contains an empty wave")

    expected_pairs = {
        (config_id, seed) for config_id in expected_configs for seed in EXPECTED_SEEDS
    }
    observed_pairs = set(zip(runs["config_id"].astype(str), runs["seed"].astype(int)))
    require(observed_pairs == expected_pairs, "runs.csv lacks part of the 8x10 config/seed grid")

    wave_sets = runs.groupby(["config_id", "seed"], sort=False)["wave_id"].agg(
        lambda values: tuple(sorted(set(values.astype(str))))
    )
    unique_wave_sets = set(wave_sets)
    require(len(unique_wave_sets) == 1, "config/seed runs do not share one complete wave set")
    wave_ids = list(next(iter(unique_wave_sets)))
    require(len(wave_ids) == EXPECTED_WAVES, "each complete instance must contain twelve waves")
    require(
        (runs.groupby("wave_id")["order_count"].nunique() == 1).all(),
        "wave order counts differ across configurations or seeds",
    )

    totals = (
        runs.groupby(["config_id", "seed"], as_index=False)
        .agg(
            order_count=("order_count", "sum"),
            objective=("objective", "sum"),
            runtime_ms=("runtime_ms", "sum"),
            wave_count=("wave_id", "nunique"),
        )
        .sort_values(["config_id", "seed"], kind="stable")
        .reset_index(drop=True)
    )
    require(len(totals) == 80, "wave aggregation did not produce eighty instance runs")
    require(set(totals["wave_count"]) == {EXPECTED_WAVES}, "incomplete wave aggregation")
    return totals, wave_ids


def truth_value(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def validate_instance_runs(
    path: Path, totals: pd.DataFrame, manifest: pd.DataFrame
) -> dict[str, float]:
    instance = pd.read_csv(path, dtype={"config_id": str})
    require_columns(instance, INSTANCE_COLUMNS, path.name)
    require(
        len(instance) == 80 and not instance.duplicated(["config_id", "seed"]).any(),
        "instance_runs.csv must contain eighty unique config/seed rows",
    )
    require(
        set(instance["solver"].astype(str)) == {"simulated_annealing"},
        "instance_runs.csv contains a non-SA solver",
    )
    require(
        set(instance["config_id"].astype(str)) == set(manifest["config_id"].astype(str)),
        "instance_runs/manifest config sets differ",
    )
    instance["seed"] = finite_numeric(instance, "seed", path.name, integer=True)
    instance["order_count"] = finite_numeric(
        instance, "order_count", path.name, nonnegative=True, integer=True
    )
    instance["objective"] = finite_numeric(
        instance, "objective", path.name, nonnegative=True
    )
    instance["runtime_ms"] = finite_numeric(
        instance, "runtime_ms", path.name, nonnegative=True
    )
    require(
        instance["feasible"].map(truth_value).all(),
        "instance_runs.csv contains an infeasible result",
    )
    merged = totals.merge(
        instance[["config_id", "seed", "order_count", "objective", "runtime_ms"]],
        on=["config_id", "seed"],
        suffixes=("_waves", "_instance"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    require(set(merged["_merge"]) == {"both"}, "instance_runs.csv lacks a config/seed result")
    require(
        np.array_equal(merged["order_count_waves"], merged["order_count_instance"]),
        "instance order counts differ from the sum of wave rows",
    )
    require(
        np.allclose(
            merged["objective_waves"],
            merged["objective_instance"],
            rtol=1e-10,
            atol=1e-6,
        ),
        "instance objective differs from the sum of wave rows",
    )

    # The instance timer surrounds evaluator construction, all wave solves and
    # final validation. Wave timers cover only the individual solver calls.
    # Therefore the former must bound, rather than equal, the wave-time sum.
    overhead = (
        merged["runtime_ms_instance"].to_numpy(dtype=float)
        - merged["runtime_ms_waves"].to_numpy(dtype=float)
    )
    require(
        np.all(overhead >= -1e-6),
        "instance runtime is smaller than the sum of wave solver runtimes",
    )
    return {
        "minimum": float(np.min(overhead)),
        "mean": float(np.mean(overhead)),
        "maximum": float(np.max(overhead)),
    }


def descriptive(values: Sequence[float], prefix: str, *, include_test: bool) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    require(array.size == len(EXPECTED_SEEDS), f"{prefix} requires ten paired observations")
    require(np.isfinite(array).all(), f"{prefix} contains a non-finite observation")
    count = int(array.size)
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    standard_error = sample_sd / math.sqrt(count)
    critical = float(stats.t.ppf(0.975, df=count - 1))
    margin = critical * standard_error
    record: dict[str, Any] = {
        f"{prefix}_n": count,
        f"{prefix}_mean": mean,
        f"{prefix}_sample_sd": sample_sd,
        f"{prefix}_standard_error": standard_error,
        f"{prefix}_t95_low": mean - margin,
        f"{prefix}_t95_high": mean + margin,
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }
    if include_test:
        if sample_sd == 0.0:
            t_statistic = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
            p_raw = 1.0 if mean == 0.0 else 0.0
        else:
            t_statistic = mean / standard_error
            p_raw = float(2.0 * stats.t.sf(abs(t_statistic), df=count - 1))
        record[f"{prefix}_t_statistic"] = t_statistic
        record[f"{prefix}_p_raw"] = p_raw
    return record


def legacy_statistics(values: Sequence[float], prefix: str) -> dict[str, Any]:
    modern = descriptive(values, prefix, include_test=False)
    return {
        f"{prefix}_n": modern[f"{prefix}_n"],
        f"{prefix}_mean": modern[f"{prefix}_mean"],
        f"{prefix}_std": modern[f"{prefix}_sample_sd"],
        f"{prefix}_ci95": modern[f"{prefix}_t95_high"] - modern[f"{prefix}_mean"],
        f"{prefix}_best": modern[f"{prefix}_min"],
    }


def with_factors(totals: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    mapping = manifest[["config_id", *FACTORS]].copy()
    return totals.merge(mapping, on="config_id", how="inner", validate="many_to_one")


def expected_legacy_config(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_id, frame in data.groupby("config_id", sort=False):
        record: dict[str, Any] = {"config_id": config_id}
        for factor in FACTORS:
            require(frame[factor].nunique() == 1, f"factor {factor} varies inside {config_id}")
            record[factor] = str(frame.iloc[0][factor])
        record.update(legacy_statistics(frame["objective"], "objective"))
        record.update(legacy_statistics(frame["runtime_ms"], "runtime_ms"))
        records.append(record)
    return pd.DataFrame(records)


def expected_legacy_main(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for factor in FACTORS:
        marginal = (
            data.groupby(["seed", factor], as_index=False)
            .agg(objective=("objective", "mean"), runtime_ms=("runtime_ms", "mean"))
        )
        reference_level = REFERENCES[factor]
        reference = marginal[marginal[factor] == reference_level]
        for level in LEVELS[factor]:
            selected = marginal[marginal[factor] == level]
            paired = selected.merge(
                reference,
                on="seed",
                suffixes=("_level", "_reference"),
                validate="one_to_one",
            )
            record: dict[str, Any] = {
                "factor": factor,
                "level": level,
                "reference_level": reference_level,
            }
            record.update(legacy_statistics(selected["objective"], "objective"))
            record.update(legacy_statistics(selected["runtime_ms"], "runtime_ms"))
            record.update(
                legacy_statistics(
                    paired["objective_level"] - paired["objective_reference"],
                    "objective_delta_vs_reference",
                )
            )
            record.update(
                legacy_statistics(
                    paired["runtime_ms_level"] - paired["runtime_ms_reference"],
                    "runtime_ms_delta_vs_reference",
                )
            )
            records.append(record)
    return pd.DataFrame(records)


def signed_cell_coefficient(row: Any, effect_factors: Sequence[str]) -> float:
    coefficient = 1.0
    for factor in effect_factors:
        level = str(getattr(row, factor))
        coefficient *= 1.0 if level == ALTERNATIVES[factor] else -1.0
    return coefficient


def expected_legacy_interactions(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for first_index, first in enumerate(FACTORS):
        for second in FACTORS[first_index + 1 :]:
            remaining = next(factor for factor in FACTORS if factor not in {first, second})
            marginal = (
                data.groupby(["seed", first, second], as_index=False)
                .agg(objective=("objective", "mean"), runtime_ms=("runtime_ms", "mean"))
            )
            values: dict[str, list[float]] = {"objective": [], "runtime_ms": []}
            for _, seed_frame in marginal.groupby("seed", sort=True):
                for metric in values:
                    contrast = math.fsum(
                        signed_cell_coefficient(row, (first, second)) * float(getattr(row, metric))
                        for row in seed_frame.itertuples(index=False)
                    )
                    values[metric].append(contrast)
            record: dict[str, Any] = {
                "interaction_order": 2,
                "factors": f"{first}:{second}",
                "averaged_over": remaining,
                "contrast_definition": "alt_alt-alt_ref-ref_alt+ref_ref",
            }
            record.update(legacy_statistics(values["objective"], "objective_contrast"))
            record.update(legacy_statistics(values["runtime_ms"], "runtime_ms_contrast"))
            records.append(record)

    three_values: dict[str, list[float]] = {"objective": [], "runtime_ms": []}
    for _, seed_frame in data.groupby("seed", sort=True):
        for metric in three_values:
            contrast = math.fsum(
                signed_cell_coefficient(row, FACTORS) * float(getattr(row, metric))
                for row in seed_frame.itertuples(index=False)
            )
            three_values[metric].append(contrast)
    record = {
        "interaction_order": 3,
        "factors": ":".join(FACTORS),
        "averaged_over": "",
        "contrast_definition": "three_factor_difference_of_differences",
    }
    record.update(legacy_statistics(three_values["objective"], "objective_contrast"))
    record.update(legacy_statistics(three_values["runtime_ms"], "runtime_ms_contrast"))
    records.append(record)
    return pd.DataFrame(records)


def compare_legacy_table(
    actual_path: Path,
    expected: pd.DataFrame,
    keys: Sequence[str],
    exact_columns: Sequence[str],
) -> None:
    actual = pd.read_csv(actual_path, dtype={key: str for key in keys})
    require_columns(actual, expected.columns, actual_path.name)
    require(len(actual) == len(expected), f"{actual_path.name} has an unexpected row count")
    require(not actual.duplicated(list(keys)).any(), f"{actual_path.name} has duplicate keys")
    merged = expected.merge(
        actual,
        on=list(keys),
        suffixes=("_expected", "_actual"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    require(set(merged["_merge"]) == {"both"}, f"{actual_path.name} keys differ from recomputation")
    for column in exact_columns:
        expected_values = merged[f"{column}_expected"].fillna("").astype(str)
        actual_values = merged[f"{column}_actual"].fillna("").astype(str)
        require(
            expected_values.equals(actual_values),
            f"{actual_path.name} column {column} differs from recomputation",
        )
    numeric_columns = [
        column for column in expected.columns if column not in {*keys, *exact_columns}
    ]
    for column in numeric_columns:
        expected_values = pd.to_numeric(merged[f"{column}_expected"], errors="raise")
        actual_values = pd.to_numeric(merged[f"{column}_actual"], errors="raise")
        require(
            np.allclose(expected_values, actual_values, rtol=1e-8, atol=1e-6, equal_nan=True),
            f"{actual_path.name} numeric column {column} differs from recomputation",
        )


def safe_protocol_path(base: Path, name: str, context: str) -> Path:
    candidate = (base / name).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe path {name!r} in {context}") from exc
    return candidate


def validate_factorial_analysis(
    input_dir: Path,
    data: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, str]]:
    analysis_dir = input_dir / "factorial_analysis"
    protocol_path = analysis_dir / "analysis_protocol.json"
    for filename in FACTORIAL_FILES:
        require_file(analysis_dir / filename)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {protocol_path}: {exc}") from exc
    require(isinstance(protocol, dict), "factorial analysis protocol must be a JSON object")
    require(protocol.get("suite_type") == "sa", "factorial analysis protocol is not for SA")
    details = protocol.get("details")
    require(isinstance(details, dict), "factorial analysis protocol lacks details")
    require(
        details.get("factor_levels") == {factor: list(LEVELS[factor]) for factor in FACTORS},
        "factor levels in the prior analysis protocol differ from the expected design",
    )
    require(
        details.get("reference_levels") == REFERENCES,
        "reference levels in the prior analysis protocol differ",
    )

    registered_inputs = protocol.get("inputs")
    require(isinstance(registered_inputs, dict), "factorial analysis protocol lacks input hashes")
    for required_name in ("runs.csv", "suite_manifest.csv"):
        require(required_name in registered_inputs, f"protocol lacks hash for {required_name}")
    if (input_dir / "convergence.csv").is_file():
        require("convergence.csv" in registered_inputs, "protocol lacks hash for convergence.csv")
    for name, expected_hash in registered_inputs.items():
        require(
            isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and all(character in "0123456789abcdef" for character in expected_hash.casefold()),
            f"invalid SHA-256 value for {name} in factorial analysis protocol",
        )
        path = safe_protocol_path(input_dir, str(name), "factorial analysis inputs")
        require_file(path)
        require(
            sha256_file(path) == expected_hash.casefold(),
            f"SHA-256 mismatch for prior-analysis input {name}",
        )

    artifacts = protocol.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "factorial analysis protocol lacks artifacts")
    require(len(artifacts) == len(set(artifacts)), "factorial analysis protocol repeats an artifact")
    require(FACTORIAL_FILES.issubset(set(artifacts)), "factorial analysis protocol omits a required artifact")
    artifact_hashes: dict[str, str] = {}
    for name in artifacts:
        require(isinstance(name, str) and name, "invalid artifact name in factorial protocol")
        path = safe_protocol_path(analysis_dir, name, "factorial analysis artifacts")
        require_file(path)
        artifact_hashes[name] = sha256_file(path)

    compare_legacy_table(
        analysis_dir / "sa_config_summary.csv",
        expected_legacy_config(data),
        ("config_id",),
        FACTORS,
    )
    compare_legacy_table(
        analysis_dir / "sa_main_effects.csv",
        expected_legacy_main(data),
        ("factor", "level"),
        ("reference_level",),
    )
    compare_legacy_table(
        analysis_dir / "sa_interactions.csv",
        expected_legacy_interactions(data),
        ("factors",),
        ("interaction_order", "averaged_over", "contrast_definition"),
    )
    return protocol, artifact_hashes


def effect_specs() -> list[tuple[str, ...]]:
    return [
        ("cooling",),
        ("neighborhood",),
        ("start_strategy",),
        ("cooling", "neighborhood"),
        ("cooling", "start_strategy"),
        ("neighborhood", "start_strategy"),
        FACTORS,
    ]


def build_seed_effects(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for seed, seed_frame in data.groupby("seed", sort=True):
        require(len(seed_frame) == EXPECTED_CONFIGS, f"seed {seed} lacks a factorial cell")
        for factors in effect_specs():
            order = len(factors)
            divisor = float(2 ** (len(FACTORS) - order))
            objective = math.fsum(
                signed_cell_coefficient(row, factors) * float(row.objective)
                for row in seed_frame.itertuples(index=False)
            ) / divisor
            runtime = math.fsum(
                signed_cell_coefficient(row, factors) * float(row.runtime_ms)
                for row in seed_frame.itertuples(index=False)
            ) / divisor
            records.append(
                {
                    "effect": ":".join(factors),
                    "effect_order": order,
                    "effect_type": "main" if order == 1 else "interaction",
                    "seed": int(seed),
                    "contrast_divisor": divisor,
                    "objective_effect": objective,
                    "runtime_ms_effect": runtime,
                }
            )
    output = pd.DataFrame(records).sort_values(
        ["effect_order", "effect", "seed"], kind="stable"
    )
    require(len(output) == 70, "expected seven effects for each of ten seeds")
    return output


def holm_adjust(p_values: Sequence[float]) -> tuple[list[float], list[int]]:
    values = np.asarray(p_values, dtype=float)
    require(np.isfinite(values).all(), "Holm correction received a non-finite p-value")
    require(((values >= 0.0) & (values <= 1.0)).all(), "p-value outside [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size, dtype=float)
    ranks = np.empty(values.size, dtype=int)
    running_maximum = 0.0
    family_size = int(values.size)
    for zero_rank, index in enumerate(order):
        candidate = (family_size - zero_rank) * float(values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = min(1.0, running_maximum)
        ranks[index] = zero_rank + 1
    return adjusted.tolist(), ranks.tolist()


def build_effect_summary(seed_effects: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for effect, frame in seed_effects.groupby("effect", sort=False):
        first = frame.iloc[0]
        record: dict[str, Any] = {
            "effect": effect,
            "effect_order": int(first["effect_order"]),
            "effect_type": str(first["effect_type"]),
            "contrast_divisor": float(first["contrast_divisor"]),
            "contrast_direction": ":".join(
                f"{ALTERNATIVES[factor]}-{REFERENCES[factor]}"
                for factor in str(effect).split(":")
            ),
        }
        record.update(descriptive(frame["objective_effect"], "objective", include_test=True))
        record.update(descriptive(frame["runtime_ms_effect"], "runtime_ms", include_test=True))
        records.append(record)
    summary = pd.DataFrame(records).sort_values(
        ["effect_order", "effect"], kind="stable"
    ).reset_index(drop=True)
    for prefix in ("objective", "runtime_ms"):
        adjusted, ranks = holm_adjust(summary[f"{prefix}_p_raw"])
        summary[f"{prefix}_holm_rank"] = ranks
        summary[f"{prefix}_p_holm_7"] = adjusted
        summary[f"{prefix}_holm_reject_0p05"] = [value <= 0.05 for value in adjusted]
    return summary


def build_config_summary(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for config_id, frame in data.groupby("config_id", sort=False):
        record: dict[str, Any] = {"config_id": config_id}
        for factor in FACTORS:
            record[factor] = str(frame.iloc[0][factor])
        record.update(descriptive(frame["objective"], "objective", include_test=False))
        record.update(descriptive(frame["runtime_ms"], "runtime_ms", include_test=False))
        records.append(record)
    return pd.DataFrame(records).sort_values(list(FACTORS), kind="stable").reset_index(drop=True)


def display_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".replace(",", "\u202f")


def make_nl_summary(effects: pd.DataFrame) -> str:
    lines = [
        "# Gekoppelde inferentie voor het SA-factorieel experiment",
        "",
        "De analyse omvat acht configuraties uit een volledig 2x2x2-ontwerp en tien gedeelde seeds. "
        "Per configuratie en seed zijn de twaalf waves eerst opgeteld. De zeven contrasten zijn daarna per seed gevormd.",
        "",
        "Een hoofdeffect is het gemiddelde verschil tussen het alternatieve en het referentieniveau. "
        "Interacties zijn gekoppelde verschillen-van-verschillen met dezelfde plus/min-codering.",
        "",
        "| effect | richting | gemiddeld effect op D [95%-BI] | ruwe p | Holm p | gemiddeld effect op tijd in ms [95%-BI] | Holm p tijd |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in effects.itertuples(index=False):
        objective = (
            f"{display_number(row.objective_mean)} "
            f"[{display_number(row.objective_t95_low)}, {display_number(row.objective_t95_high)}]"
        )
        runtime = (
            f"{display_number(row.runtime_ms_mean)} "
            f"[{display_number(row.runtime_ms_t95_low)}, {display_number(row.runtime_ms_t95_high)}]"
        )
        lines.append(
            f"| {row.effect} | {row.contrast_direction} | {objective} | "
            f"{row.objective_p_raw:.6g} | {row.objective_p_holm_7:.6g} | "
            f"{runtime} | {row.runtime_ms_p_holm_7:.6g} |"
        )
    significant = effects[effects["objective_holm_reject_0p05"]]
    if significant.empty:
        conclusion = "Na Holm-correctie is geen van de zeven kwaliteitscontrasten significant op het 5%-niveau."
    else:
        conclusion = (
            "Na Holm-correctie blijven de volgende kwaliteitscontrasten significant op het 5%-niveau: "
            + ", ".join(significant["effect"].astype(str))
            + "."
        )
    lines.extend(
        [
            "",
            conclusion,
            "",
            "De Holm-correctie is afzonderlijk toegepast op de zeven kwaliteitscontrasten en de zeven runtimecontrasten. "
            "De toets is telkens een tweezijdige gekoppelde Student-t-toets op de tien seedverschillen.",
            "",
            "Positieve hoofdeffecten op de doelfunctie betekenen dat het alternatieve niveau gemiddeld slechter scoort, "
            "aangezien een lagere afstand beter is. Interactiecontrasten vereisen een conditionele interpretatie en zijn geen afzonderlijke algoritmevariant.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    input_dir: Path,
    manifest_path: Path,
    runs_path: Path,
    instance_path: Path,
    suite_name: str,
    wave_ids: Sequence[str],
    prior_protocol: dict[str, Any],
    prior_hashes: dict[str, str],
    instance_timer_overhead_ms: dict[str, float],
    seed_effects: pd.DataFrame,
    effects: pd.DataFrame,
    configs: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    seed_path = out_dir / "sa_factorial_paired_seed_effects.csv"
    effect_path = out_dir / "sa_factorial_effects.csv"
    config_path = out_dir / "sa_factorial_config_summary.csv"
    summary_path = out_dir / "sa_factorial_summary_nl.md"
    seed_effects.to_csv(seed_path, index=False)
    effects.to_csv(effect_path, index=False)
    configs.to_csv(config_path, index=False)
    summary_path.write_text(make_nl_summary(effects), encoding="utf-8")

    artifacts = [seed_path, effect_path, config_path, summary_path]
    script_path = Path(__file__).resolve()
    raw_inputs = {
        "suite_manifest.csv": sha256_file(manifest_path),
        "runs.csv": sha256_file(runs_path),
        "instance_runs.csv": sha256_file(instance_path),
    }
    convergence_path = input_dir / "convergence.csv"
    if convergence_path.is_file():
        raw_inputs["convergence.csv"] = sha256_file(convergence_path)
    protocol = {
        "analysis": "retrospective_sa_factorial_paired_inference",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(out_dir),
        "source_suite": suite_name,
        "design": {
            "type": "complete_2x2x2_factorial",
            "configurations": EXPECTED_CONFIGS,
            "paired_seeds": list(EXPECTED_SEEDS),
            "waves_per_complete_instance": EXPECTED_WAVES,
            "wave_ids": list(wave_ids),
            "factors": {factor: list(LEVELS[factor]) for factor in FACTORS},
            "reference_levels": REFERENCES,
            "alternative_levels": ALTERNATIVES,
            "effects": [":".join(factors) for factors in effect_specs()],
        },
        "statistical_unit": "complete_instance_per_configuration_and_seed",
        "pairing_key": "seed",
        "wave_aggregation": "sum objective and internal solver runtime over all twelve waves before contrasts",
        "effect_scaling": "sum of plus/minus-coded cell means divided by 2^(3-effect_order)",
        "effect_sign": "alternative_minus_reference for main effects; corresponding differences-of-differences for interactions",
        "confidence_interval": "two-sided 95% Student-t interval for the paired mean, df=9",
        "test": "two-sided one-sample Student-t test of ten paired factorial effects against zero",
        "multiplicity": {
            "method": "Holm step-down family-wise error correction",
            "alpha": 0.05,
            "families": {
                "objective_effects": 7,
                "internal_runtime_effects": 7,
            },
        },
        "validation": {
            "run_rows": 960,
            "instance_rows": 80,
            "paired_seed_effect_rows": 70,
            "all_instance_runs_feasible": True,
            "instance_objectives_match_wave_sums": True,
            "instance_runtime_bounds_wave_runtime_sum": True,
            "instance_timer_overhead_ms": instance_timer_overhead_ms,
            "prior_factorial_analysis_recomputed": True,
            "prior_analysis_registered_input_hashes_verified": True,
            "prior_analysis_protocol": prior_protocol,
        },
        "limitations": [
            "Ten paired seeds support inference on seed variability, not on a population of independent problem instances.",
            "Equal seed labels define the pairing, but factor changes can alter subsequent random-number consumption.",
            "Holm correction is applied separately to quality and internal-runtime families.",
            "Runtime effects use the sum of twelve wave solver timers and exclude instance-level evaluator setup and final validation.",
            "The analysis validates archived outputs and hashes but does not invoke or revalidate the solver executable.",
        ],
        "sha256": {
            "raw_suite_inputs": raw_inputs,
            "prior_factorial_analysis_artifacts": prior_hashes,
            "analysis_script": sha256_file(script_path),
            "output_artifacts_excluding_protocol": {
                path.name: sha256_file(path) for path in artifacts
            },
        },
        "artifacts": [path.name for path in artifacts] + ["analysis_protocol.json"],
    }
    (out_dir / "analysis_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_analysis(input_dir: Path, out_dir: Path) -> None:
    input_dir = input_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"missing input directory: {input_dir}")

    manifest_path = input_dir / "suite_manifest.csv"
    runs_path = input_dir / "runs.csv"
    instance_path = input_dir / "instance_runs.csv"
    for path in (manifest_path, runs_path, instance_path):
        require_file(path)

    manifest, _parameters, suite_name = load_manifest(manifest_path)
    totals, wave_ids = load_runs(runs_path, manifest)
    instance_timer_overhead_ms = validate_instance_runs(instance_path, totals, manifest)
    data = with_factors(totals, manifest)
    prior_protocol, prior_hashes = validate_factorial_analysis(input_dir, data)
    seed_effects = build_seed_effects(data)
    effects = build_effect_summary(seed_effects)
    configs = build_config_summary(data)
    write_outputs(
        out_dir,
        input_dir,
        manifest_path,
        runs_path,
        instance_path,
        suite_name,
        wave_ids,
        prior_protocol,
        prior_hashes,
        instance_timer_overhead_ms,
        seed_effects,
        effects,
        configs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_analysis(args.input_dir, args.out_dir)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Wrote paired SA factorial inference to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
