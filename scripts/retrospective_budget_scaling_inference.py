"""Inferential reanalysis of the completed Release budget and scaling studies.

The script is deliberately read-only with respect to the source studies.  It
validates the locked CSV grids and their recorded provenance, derives all
statistics from the existing measurements, and refuses to reuse an output
directory.  No solver process is started.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import scipy
from scipy.stats import t as student_t


BUDGET_LABELS: tuple[str, ...] = (
    "sa_i120",
    "sa_i400",
    "sa_i2000",
    "sa_i10000",
    "ga_g10",
    "ga_g20",
    "ga_g40",
)
BUDGET_METADATA: dict[str, tuple[str, int]] = {
    "sa_i120": ("SA", 120),
    "sa_i400": ("SA", 400),
    "sa_i2000": ("SA", 2000),
    "sa_i10000": ("SA", 10000),
    "ga_g10": ("GA", 10),
    "ga_g20": ("GA", 20),
    "ga_g40": ("GA", 40),
}
MARGINAL_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("sa_i2000_minus_sa_i400", "sa_i400", "sa_i2000"),
    ("sa_i10000_minus_sa_i2000", "sa_i2000", "sa_i10000"),
    ("ga_g40_minus_ga_g20", "ga_g20", "ga_g40"),
)
EXPECTED_BUDGET_SEEDS: tuple[int, ...] = tuple(range(10))
EXPECTED_SCALING_LABELS: tuple[str, ...] = (
    "greedy_matched",
    "sa_matched",
    "ga_matched",
)
EXPECTED_SCALING_N: tuple[int, ...] = (50, 100, 200, 400, 800)
EXPECTED_SCALING_REPEATS: tuple[int, ...] = (0, 1, 2)
EXPECTED_SCALING_SEEDS: tuple[int, ...] = (0, 1, 2)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected a JSON object in {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"missing CSV header in {path}")
        return list(reader)


def parse_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer for {context}: {value!r}") from exc


def parse_float(value: Any, context: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number for {context}: {value!r}") from exc
    require(math.isfinite(number), f"non-finite number for {context}: {value!r}")
    if positive:
        require(number > 0.0, f"non-positive number for {context}: {value!r}")
    return number


def parse_bool(value: Any, context: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid Boolean for {context}: {value!r}")


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    require(bool(records), f"refusing to write an empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def check_hash(path: Path, expected: Any, context: str) -> None:
    require(isinstance(expected, str) and len(expected) == 64, f"missing hash for {context}")
    observed = sha256_file(path)
    require(observed == expected, f"SHA-256 mismatch for {context}: {path}")


def paired_statistics(values: Sequence[float], confidence: float) -> dict[str, float | int]:
    require(len(values) >= 2, "a paired statistic requires at least two observations")
    n = len(values)
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    df = n - 1
    critical = float(student_t.ppf(0.5 + confidence / 2.0, df))
    standard_error = sample_sd / math.sqrt(n)
    margin = critical * standard_error
    if sample_sd == 0.0:
        statistic = math.inf if mean != 0.0 else 0.0
        raw_p = 0.0 if mean != 0.0 else 1.0
    else:
        statistic = mean / standard_error
        raw_p = float(2.0 * student_t.sf(abs(statistic), df))
    return {
        "n_paired": n,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "degrees_of_freedom": df,
        "t_statistic": statistic,
        "t_critical": critical,
        "t95_low": mean - margin,
        "t95_high": mean + margin,
        "raw_two_sided_t_p": raw_p,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    count = len(p_values)
    require(count > 0, "Holm correction needs at least one p-value")
    require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in p_values), "invalid p-value")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running_maximum = 0.0
    for rank, (original_index, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return adjusted


def validate_budget_inputs(
    study_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, float]], dict[str, Path]]:
    manifest_path = study_dir / "study_manifest.json"
    analysis_dir = study_dir / "analysis"
    source_protocol_path = analysis_dir / "analysis_protocol.json"
    runs_path = analysis_dir / "full_instance_runs.csv"
    manifest = load_json(manifest_path)
    source_protocol = load_json(source_protocol_path)

    require(manifest.get("state") == "complete", "budget study is not complete")
    require(manifest.get("expected_total_runs") == 960, "budget manifest does not declare 960 runs")
    require(manifest.get("solver_seeds") == list(EXPECTED_BUDGET_SEEDS), "budget seeds differ")
    require(source_protocol.get("state") == "complete", "budget aggregation is not complete")
    require(source_protocol.get("validated_raw_runs") == 960, "budget aggregation did not validate 960 runs")
    require(source_protocol.get("full_instance_rows") == 80, "budget aggregation does not contain 80 totals")
    require(
        source_protocol.get("study_manifest_sha256") == sha256_file(manifest_path),
        "budget study manifest changed after aggregation",
    )
    recorded_outputs = source_protocol.get("outputs")
    require(isinstance(recorded_outputs, dict), "budget aggregation lacks output hashes")
    recorded_runs = recorded_outputs.get("full_instance_runs.csv")
    require(isinstance(recorded_runs, dict), "budget aggregation lacks full_instance_runs provenance")
    check_hash(runs_path, recorded_runs.get("sha256"), "budget full-instance totals")

    experiments = manifest.get("experiments")
    require(isinstance(experiments, list), "budget manifest lacks experiment metadata")
    manifest_by_label = {
        str(item.get("label")): item for item in experiments if isinstance(item, dict)
    }
    expected_labels = {"greedy", *BUDGET_LABELS}
    require(set(manifest_by_label) == expected_labels, "budget experiment grid differs")
    for label in BUDGET_LABELS:
        family, budget = BUDGET_METADATA[label]
        require(manifest_by_label[label].get("family") == family, f"wrong family for {label}")
        require(manifest_by_label[label].get("budget") == budget, f"wrong budget for {label}")

    records = load_csv(runs_path)
    require(len(records) == 80, "budget totals CSV must have 80 rows")
    required_fields = {
        "experiment_label",
        "family",
        "budget",
        "config_id",
        "config_sha256",
        "solver_seed",
        "wave_count",
        "order_count_total",
        "objective_total",
    }
    require(required_fields.issubset(records[0]), "budget totals CSV schema differs")
    objectives: dict[str, dict[int, float]] = defaultdict(dict)
    seen_pairs: set[tuple[str, int]] = set()
    for row_number, row in enumerate(records, start=2):
        label = row["experiment_label"]
        require(label in expected_labels, f"unknown budget label on row {row_number}: {label}")
        seed = parse_int(row["solver_seed"], f"budget row {row_number} solver seed")
        require(seed in EXPECTED_BUDGET_SEEDS, f"unexpected budget seed on row {row_number}")
        pair = (label, seed)
        require(pair not in seen_pairs, f"duplicate budget pair: {pair}")
        seen_pairs.add(pair)
        metadata = manifest_by_label[label]
        require(row["config_id"] == metadata.get("config_id"), f"config id mismatch for {label}")
        require(row["config_sha256"] == metadata.get("config_sha256"), f"config hash mismatch for {label}")
        require(parse_int(row["wave_count"], f"{label}/{seed} wave count") == 12, "wrong wave count")
        require(
            parse_int(row["order_count_total"], f"{label}/{seed} order count") == 10246,
            "wrong full-instance order count",
        )
        objectives[label][seed] = parse_float(row["objective_total"], f"{label}/{seed} objective")
    expected_pairs = {(label, seed) for label in expected_labels for seed in EXPECTED_BUDGET_SEEDS}
    require(seen_pairs == expected_pairs, "budget label/seed grid is incomplete")

    source_paths = {
        "budget_study_manifest": manifest_path,
        "budget_aggregation_protocol": source_protocol_path,
        "budget_full_instance_runs": runs_path,
    }
    return manifest, source_protocol, dict(objectives), source_paths


def budget_comparisons(
    objectives: dict[str, dict[int, float]], confidence: float, alpha: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    greedy = objectives["greedy"]
    comparisons: list[dict[str, Any]] = []
    paired_values: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for label in BUDGET_LABELS:
        family, budget = BUDGET_METADATA[label]
        gains = [greedy[seed] - objectives[label][seed] for seed in EXPECTED_BUDGET_SEEDS]
        percent_gains = [100.0 * gain / greedy[seed] for gain, seed in zip(gains, EXPECTED_BUDGET_SEEDS)]
        stats = paired_statistics(gains, confidence)
        percent_stats = paired_statistics(percent_gains, confidence)
        comparisons.append(
            {
                "comparison_id": f"{label}_vs_greedy",
                "experiment_label": label,
                "family": family,
                "budget": budget,
                "gain_definition": "greedy_objective_minus_candidate_objective",
                **stats,
                "mean_gain_pct": percent_stats["mean"],
                "sample_sd_gain_pct": percent_stats["sample_sd"],
                "t95_low_gain_pct": percent_stats["t95_low"],
                "t95_high_gain_pct": percent_stats["t95_high"],
            }
        )
        raw_p_values.append(float(stats["raw_two_sided_t_p"]))
        for seed, gain, percent_gain in zip(EXPECTED_BUDGET_SEEDS, gains, percent_gains):
            paired_values.append(
                {
                    "comparison_id": f"{label}_vs_greedy",
                    "experiment_label": label,
                    "family": family,
                    "budget": budget,
                    "solver_seed": seed,
                    "greedy_objective": greedy[seed],
                    "candidate_objective": objectives[label][seed],
                    "paired_gain_abs": gain,
                    "paired_gain_pct": percent_gain,
                }
            )
    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(comparisons, adjusted):
        row["holm_family_size"] = len(BUDGET_LABELS)
        row["holm_adjusted_p"] = adjusted_p
        row["holm_reject_at_alpha"] = adjusted_p <= alpha
        row["alpha"] = alpha

    marginal_rows: list[dict[str, Any]] = []
    marginal_raw_p: list[float] = []
    for contrast_id, lower_label, higher_label in MARGINAL_CONTRASTS:
        gains = [
            objectives[lower_label][seed] - objectives[higher_label][seed]
            for seed in EXPECTED_BUDGET_SEEDS
        ]
        stats = paired_statistics(gains, confidence)
        lower_family, lower_budget = BUDGET_METADATA[lower_label]
        higher_family, higher_budget = BUDGET_METADATA[higher_label]
        require(lower_family == higher_family, f"cross-family marginal contrast: {contrast_id}")
        marginal_rows.append(
            {
                "contrast_id": contrast_id,
                "family": lower_family,
                "lower_budget_label": lower_label,
                "lower_budget": lower_budget,
                "higher_budget_label": higher_label,
                "higher_budget": higher_budget,
                "gain_definition": "lower_budget_objective_minus_higher_budget_objective",
                **stats,
            }
        )
        marginal_raw_p.append(float(stats["raw_two_sided_t_p"]))
    marginal_adjusted = holm_adjust(marginal_raw_p)
    for row, adjusted_p in zip(marginal_rows, marginal_adjusted):
        row["holm_family_size"] = len(MARGINAL_CONTRASTS)
        row["holm_adjusted_p"] = adjusted_p
        row["holm_reject_at_alpha"] = adjusted_p <= alpha
        row["alpha"] = alpha
    return comparisons, marginal_rows, paired_values


def validate_scaling_inputs(
    study_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Path]]:
    protocol_path = study_dir / "protocol.json"
    runs_path = study_dir / "runs.csv"
    legacy_fits_path = study_dir / "fits.csv"
    summary_path = study_dir / "summary.csv"
    protocol = load_json(protocol_path)
    require(protocol.get("state") == "complete", "scaling study is not complete")
    require(protocol.get("completed_runs") == 135, "scaling study does not contain 135 runs")
    require(protocol.get("successful_runs") == 135, "scaling study contains unsuccessful runs")
    require(protocol.get("unsuccessful_runs") == 0, "scaling study contains unsuccessful runs")
    measured = protocol.get("protocol")
    require(isinstance(measured, dict), "scaling protocol block is absent")
    require(measured.get("n_values") == list(EXPECTED_SCALING_N), "scaling n grid differs")
    require(measured.get("repeats") == 3, "scaling repeat count differs")
    require(measured.get("solver_seeds") == list(EXPECTED_SCALING_SEEDS), "scaling seeds differ")
    software = protocol.get("software")
    require(isinstance(software, dict), "scaling software provenance is absent")
    require(software.get("release_was_required") is True, "scaling study did not require Release")
    require(str(software.get("cmake_build_type_detected", "")).casefold() == "release", "scaling study is not Release")

    experiments = protocol.get("experiments")
    require(isinstance(experiments, list), "scaling experiment metadata is absent")
    by_label = {str(item.get("label")): item for item in experiments if isinstance(item, dict)}
    require(set(by_label) == set(EXPECTED_SCALING_LABELS), "scaling experiment grid differs")

    records = load_csv(runs_path)
    require(len(records) == 135, "scaling runs CSV must contain 135 rows")
    required_fields = {
        "experiment_label",
        "solver",
        "declared_config_id",
        "observed_config_id",
        "config_sha256",
        "n_orders",
        "repeat_index",
        "solver_seed",
        "status",
        "return_code",
        "timed_out",
        "feasible",
        "solver_runtime_ms",
    }
    require(required_fields.issubset(records[0]), "scaling runs CSV schema differs")
    seen: set[tuple[str, int, int, int]] = set()
    for row_number, row in enumerate(records, start=2):
        label = row["experiment_label"]
        require(label in EXPECTED_SCALING_LABELS, f"unknown scaling label on row {row_number}")
        n_orders = parse_int(row["n_orders"], f"scaling row {row_number} n")
        repeat = parse_int(row["repeat_index"], f"scaling row {row_number} repeat")
        seed = parse_int(row["solver_seed"], f"scaling row {row_number} seed")
        key = (label, n_orders, repeat, seed)
        require(n_orders in EXPECTED_SCALING_N, f"unexpected n on scaling row {row_number}")
        require(repeat in EXPECTED_SCALING_REPEATS, f"unexpected repeat on scaling row {row_number}")
        require(seed in EXPECTED_SCALING_SEEDS, f"unexpected seed on scaling row {row_number}")
        require(key not in seen, f"duplicate scaling cell: {key}")
        seen.add(key)
        metadata = by_label[label]
        require(row["solver"] == metadata.get("solver"), f"scaling solver mismatch for {label}")
        require(row["declared_config_id"] == metadata.get("declared_config_id"), f"declared config mismatch for {label}")
        require(row["observed_config_id"] == metadata.get("declared_config_id"), f"observed config mismatch for {label}")
        require(row["config_sha256"] == metadata.get("config_sha256"), f"config hash mismatch for {label}")
        require(row["status"] == "success", f"failed scaling run on row {row_number}")
        require(parse_int(row["return_code"], f"scaling row {row_number} return code") == 0, "nonzero return code")
        require(not parse_bool(row["timed_out"], f"scaling row {row_number} timed_out"), "timed-out scaling run")
        require(parse_bool(row["feasible"], f"scaling row {row_number} feasible"), "infeasible scaling run")
        parse_float(row["solver_runtime_ms"], f"scaling row {row_number} runtime", positive=True)
    expected = {
        (label, n_orders, repeat, seed)
        for label in EXPECTED_SCALING_LABELS
        for n_orders in EXPECTED_SCALING_N
        for repeat in EXPECTED_SCALING_REPEATS
        for seed in EXPECTED_SCALING_SEEDS
    }
    require(seen == expected, "scaling label/n/repeat/seed grid is incomplete")
    source_paths = {
        "scaling_protocol": protocol_path,
        "scaling_runs": runs_path,
        "scaling_legacy_fits": legacy_fits_path,
        "scaling_summary": summary_path,
    }
    for path in source_paths.values():
        require(path.is_file(), f"missing scaling input: {path}")
    return protocol, records, source_paths


def scaling_fits(
    protocol: dict[str, Any], records: Sequence[dict[str, str]], confidence: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtimes: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in records:
        runtimes[(row["experiment_label"], int(row["n_orders"]))].append(float(row["solver_runtime_ms"]))

    experiments = {
        str(item["label"]): item
        for item in protocol["experiments"]
        if isinstance(item, dict)
    }
    median_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for label in EXPECTED_SCALING_LABELS:
        medians: list[float] = []
        for n_orders in EXPECTED_SCALING_N:
            values = runtimes[(label, n_orders)]
            require(len(values) == 9, f"{label}/n={n_orders} does not contain nine runtimes")
            median = statistics.median(values)
            medians.append(median)
            median_rows.append(
                {
                    "experiment_label": label,
                    "solver": experiments[label]["solver"],
                    "n_orders": n_orders,
                    "runtime_observations": len(values),
                    "median_solver_runtime_ms": median,
                }
            )

        x = [math.log(value) for value in EXPECTED_SCALING_N]
        y = [math.log(value) for value in medians]
        x_mean = statistics.fmean(x)
        y_mean = statistics.fmean(y)
        sxx = sum((value - x_mean) ** 2 for value in x)
        slope = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y)) / sxx
        intercept = y_mean - slope * x_mean
        fitted = [intercept + slope * value for value in x]
        residuals = [observed - estimate for observed, estimate in zip(y, fitted)]
        residual_sum_squares = sum(value * value for value in residuals)
        total_sum_squares = sum((value - y_mean) ** 2 for value in y)
        df = len(x) - 2
        slope_se = math.sqrt((residual_sum_squares / df) / sxx)
        critical = float(student_t.ppf(0.5 + confidence / 2.0, df))
        normal_critical = 1.959963984540054
        fit_rows.append(
            {
                "experiment_label": label,
                "solver": experiments[label]["solver"],
                "runtime_metric": "median solver_runtime_ms per n",
                "fit_model": "T(n) = a * n^b",
                "fit_method": "OLS on ln(median runtime) versus ln(n)",
                "n_levels": len(x),
                "n_min": min(EXPECTED_SCALING_N),
                "n_max": max(EXPECTED_SCALING_N),
                "coefficient_a_ms": math.exp(intercept),
                "empirical_exponent_b": slope,
                "slope_standard_error": slope_se,
                "residual_degrees_of_freedom": df,
                "t_critical": critical,
                "slope_t95_low": slope - critical * slope_se,
                "slope_t95_high": slope + critical * slope_se,
                "normal_critical": normal_critical,
                "slope_normal_approx_95_low": slope - normal_critical * slope_se,
                "slope_normal_approx_95_high": slope + normal_critical * slope_se,
                "r_squared_log_space": 1.0 - residual_sum_squares / total_sum_squares,
                "interpretation": "descriptive finite-range empirical fit, not asymptotic complexity",
            }
        )
    return fit_rows, median_rows


def verify_legacy_scaling_fits(study_dir: Path, fits: Sequence[dict[str, Any]]) -> None:
    legacy = load_csv(study_dir / "fits.csv")
    require(len(legacy) == len(EXPECTED_SCALING_LABELS), "legacy scaling fit count differs")
    by_label = {row["experiment_label"]: row for row in legacy}
    require(set(by_label) == set(EXPECTED_SCALING_LABELS), "legacy scaling labels differ")
    for fit in fits:
        label = str(fit["experiment_label"])
        observed = by_label[label]
        for old_field, new_field in (
            ("coefficient_a_ms", "coefficient_a_ms"),
            ("empirical_exponent_b", "empirical_exponent_b"),
            ("slope_standard_error", "slope_standard_error"),
            ("r_squared_log_space", "r_squared_log_space"),
        ):
            old_value = parse_float(observed[old_field], f"legacy {label}/{old_field}")
            new_value = float(fit[new_field])
            require(
                math.isclose(old_value, new_value, rel_tol=1e-12, abs_tol=1e-12),
                f"recomputed scaling fit differs for {label}/{old_field}",
            )


def file_manifest(paths: dict[str, Path]) -> dict[str, dict[str, str | int]]:
    return {
        label: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in paths.items()
    }


def build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-study-dir",
        type=Path,
        default=root / "results" / "retrospective_budget_release_full_20260806",
    )
    parser.add_argument(
        "--scaling-study-dir",
        type=Path,
        default=root / "results" / "scaling_controlled_matched",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "results" / "retrospective_budget_scaling_inference_20260807",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    args = build_parser(root).parse_args(argv)
    confidence = float(args.confidence)
    alpha = float(args.alpha)
    require(0.0 < confidence < 1.0, "confidence must lie strictly between zero and one")
    require(0.0 < alpha < 1.0, "alpha must lie strictly between zero and one")
    budget_dir = args.budget_study_dir.resolve()
    scaling_dir = args.scaling_study_dir.resolve()
    out_dir = args.out_dir.resolve()
    require(not out_dir.exists(), f"refusing to overwrite existing output directory: {out_dir}")

    budget_manifest, budget_protocol, objectives, budget_paths = validate_budget_inputs(budget_dir)
    comparisons, marginal, paired_values = budget_comparisons(objectives, confidence, alpha)
    scaling_protocol, scaling_records, scaling_paths = validate_scaling_inputs(scaling_dir)
    fits, scaling_medians = scaling_fits(scaling_protocol, scaling_records, confidence)
    verify_legacy_scaling_fits(scaling_dir, fits)

    script_path = Path(__file__).resolve()
    inputs = {**budget_paths, **scaling_paths, "analysis_script": script_path}
    input_manifest = file_manifest(inputs)
    out_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "budget_paired_comparisons": out_dir / "budget_paired_comparisons.csv",
        "budget_marginal_contrasts": out_dir / "budget_marginal_contrasts.csv",
        "budget_paired_seed_values": out_dir / "budget_paired_seed_values.csv",
        "scaling_exponents": out_dir / "scaling_exponents.csv",
        "scaling_runtime_medians": out_dir / "scaling_runtime_medians.csv",
    }
    write_csv(output_paths["budget_paired_comparisons"], comparisons)
    write_csv(output_paths["budget_marginal_contrasts"], marginal)
    write_csv(output_paths["budget_paired_seed_values"], paired_values)
    write_csv(output_paths["scaling_exponents"], fits)
    write_csv(output_paths["scaling_runtime_medians"], scaling_medians)

    source_software = {
        "budget_cli_sha256": budget_manifest.get("software", {}).get("cli_sha256"),
        "budget_cmake_build_type": budget_manifest.get("software", {}).get("cmake_build_type"),
        "scaling_cli_sha256": scaling_protocol.get("software", {}).get("cli_sha256"),
        "scaling_cmake_build_type": scaling_protocol.get("software", {}).get("cmake_build_type_detected"),
    }
    protocol = {
        "schema_version": 1,
        "state": "complete",
        "created_utc": utc_now(),
        "study": "inferential reanalysis of existing Release budget and scaling measurements",
        "execution": "analysis only; no solver process was started",
        "source_studies": {
            "budget": str(budget_dir),
            "scaling": str(scaling_dir),
        },
        "validation": {
            "budget_raw_runs_previously_validated": budget_protocol.get("validated_raw_runs"),
            "budget_full_instance_rows_revalidated": 80,
            "scaling_raw_runs_revalidated": len(scaling_records),
            "scaling_grid": {
                "labels": list(EXPECTED_SCALING_LABELS),
                "n_values": list(EXPECTED_SCALING_N),
                "repeats": list(EXPECTED_SCALING_REPEATS),
                "solver_seeds": list(EXPECTED_SCALING_SEEDS),
            },
            "legacy_scaling_fits_reproduced": True,
        },
        "budget_inference": {
            "sampling_unit": "solver seed after summing objective over all twelve complete waves",
            "pairing": "candidate and greedy share the same solver-seed label",
            "gain_definition": "greedy objective minus candidate objective; positive favours candidate",
            "planned_comparisons": list(BUDGET_LABELS),
            "test": "two-sided one-sample Student-t test of paired gains against zero",
            "confidence_interval": f"two-sided Student-t interval for the mean at confidence={confidence}",
            "multiplicity": "Holm step-down adjustment across the seven predeclared candidate-versus-greedy budgets",
            "marginal_contrasts": [item[0] for item in MARGINAL_CONTRASTS],
            "marginal_multiplicity": "separate Holm family across the three predeclared marginal contrasts",
            "alpha": alpha,
        },
        "scaling_inference": {
            "aggregation": "median solver_runtime_ms over nine runs at each n",
            "model": "OLS of ln(median runtime) on ln(n), T(n)=a*n^b",
            "primary_interval": "Student-t slope interval with n_levels-2 residual degrees of freedom",
            "compatibility_interval": "normal approximation retained to reproduce the original fits.csv convention",
            "scope": "descriptive over n=50..800 and not evidence of asymptotic complexity",
        },
        "source_software": source_software,
        "analysis_software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "scipy": scipy.__version__,
        },
        "inputs": input_manifest,
        "outputs": file_manifest(output_paths),
        "row_counts": {
            "budget_paired_comparisons": len(comparisons),
            "budget_marginal_contrasts": len(marginal),
            "budget_paired_seed_values": len(paired_values),
            "scaling_exponents": len(fits),
            "scaling_runtime_medians": len(scaling_medians),
        },
    }
    protocol_path = out_dir / "analysis_protocol.json"
    with protocol_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote validated inference outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
