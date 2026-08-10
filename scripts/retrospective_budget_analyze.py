"""Validate and aggregate a completed retrospective full-budget study."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from retrospective_budget_common import (
    EXPECTED_RUNS_PER_WAVE,
    EXPECTED_TOTAL_RUNS,
    EXPERIMENTS,
    REPEATS,
    SOLVER_SEEDS,
    WAVES,
    repo_root,
    sha256_file,
    validate_configuration_files,
)


T_CRITICAL_95_DF9 = 2.2621571627409915


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer for {context}: {value!r}") from exc


def as_float(value: Any, context: str, *, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number for {context}: {value!r}") from exc
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"invalid number for {context}: {value!r}")
    return number


def is_true(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def is_false(value: Any) -> bool:
    return str(value).strip().casefold() in {"false", "0", "no", ""}


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def descriptive_fields(
    values: Sequence[float], prefix: str, *, best: str
) -> dict[str, float | int]:
    require(len(values) == len(SOLVER_SEEDS), f"{prefix} requires ten paired values")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    margin = T_CRITICAL_95_DF9 * sample_sd / math.sqrt(len(values))
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_mean": mean,
        f"{prefix}_sample_sd": sample_sd,
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": q3 - q1,
        f"{prefix}_best": min(values) if best == "min" else max(values),
        f"{prefix}_t95_low": mean - margin,
        f"{prefix}_t95_high": mean + margin,
    }


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    require(bool(records), f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def expected_experiment_metadata(root: Path) -> tuple[list[dict[str, str | int]], dict[str, dict[str, str | int]]]:
    metadata = validate_configuration_files(root)
    return metadata, {str(item["label"]): item for item in metadata}


def validate_manifest(
    study_dir: Path,
    root: Path,
    expected_metadata: list[dict[str, str | int]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_json(study_dir / "study_manifest.json")
    require(manifest.get("state") == "complete", "study manifest is not complete")
    require(
        manifest.get("expected_total_runs") == EXPECTED_TOTAL_RUNS,
        f"manifest must declare {EXPECTED_TOTAL_RUNS} runs",
    )
    require(
        manifest.get("expected_runs_per_wave") == EXPECTED_RUNS_PER_WAVE,
        f"manifest must declare {EXPECTED_RUNS_PER_WAVE} runs per wave",
    )
    require(manifest.get("repeats") == REPEATS, "manifest repeats differ from protocol")
    require(manifest.get("solver_seeds") == list(SOLVER_SEEDS), "manifest solver seeds differ")
    require(
        manifest.get("completed_waves") == [wave for wave, _ in WAVES],
        "manifest does not contain the twelve completed waves in protocol order",
    )

    planned_waves = manifest.get("waves")
    require(isinstance(planned_waves, list) and len(planned_waves) == len(WAVES), "invalid wave plan")
    wave_metadata: dict[str, dict[str, Any]] = {}
    schedule_seeds: set[int] = set()
    subset_seeds: set[int] = set()
    for expected, observed in zip(WAVES, planned_waves):
        require(isinstance(observed, dict), "invalid wave entry in manifest")
        wave_id, n_orders = expected
        require(observed.get("wave_id") == wave_id, f"unexpected wave order at {wave_id}")
        require(observed.get("n_orders") == n_orders, f"wrong full order count for {wave_id}")
        schedule_seed = as_int(observed.get("schedule_seed"), f"{wave_id} schedule seed")
        subset_seed = as_int(observed.get("subset_seed"), f"{wave_id} subset seed")
        require(schedule_seed not in schedule_seeds, "schedule seeds must be unique per wave")
        require(subset_seed not in subset_seeds, "subset seeds must be unique per wave")
        schedule_seeds.add(schedule_seed)
        subset_seeds.add(subset_seed)
        wave_metadata[wave_id] = observed

    observed_experiments = manifest.get("experiments")
    require(observed_experiments == expected_metadata, "manifest experiment metadata or hashes differ")
    software = manifest.get("software")
    require(isinstance(software, dict), "missing software metadata")
    require(str(software.get("cmake_build_type", "")).casefold() == "release", "study was not Release")
    runner = root / "scripts" / "run_scaling_experiments.py"
    common = root / "scripts" / "retrospective_budget_common.py"
    wrapper = root / "scripts" / "retrospective_budget_run.py"
    require(software.get("delegated_runner_sha256") == sha256_file(runner), "delegated runner hash changed")
    require(software.get("common_module_sha256") == sha256_file(common), "common module hash changed")
    require(software.get("wrapper_sha256") == sha256_file(wrapper), "wrapper hash changed")
    cli_path = Path(str(software.get("cli_path", "")))
    require(cli_path.is_file(), f"recorded CLI is unavailable: {cli_path}")
    require(software.get("cli_sha256") == sha256_file(cli_path), "recorded CLI hash changed")
    data = manifest.get("data")
    require(isinstance(data, dict), "missing data hashes")
    require(
        data.get("pick_data_sha256") == sha256_file(root / "data" / "pick_data.csv"),
        "pick_data.csv hash differs from the measured input",
    )
    require(
        data.get("dist_matrix_sha256") == sha256_file(root / "data" / "dist_matrix.csv"),
        "dist_matrix.csv hash differs from the measured input",
    )
    return manifest, wave_metadata


def validate_and_load_runs(
    study_dir: Path,
    manifest: dict[str, Any],
    wave_metadata: dict[str, dict[str, Any]],
    expected_by_label: dict[str, dict[str, str | int]],
) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    cli_sha = manifest["software"]["cli_sha256"]
    runner_sha = manifest["software"]["delegated_runner_sha256"]
    pick_sha = manifest["data"]["pick_data_sha256"]
    distance_sha = manifest["data"]["dist_matrix_sha256"]
    expected_labels = {spec.label for spec in EXPERIMENTS}

    for wave_id, n_orders in WAVES:
        wave_dir = study_dir / wave_id
        protocol = load_json(wave_dir / "protocol.json")
        require(protocol.get("state") == "complete", f"{wave_id} protocol is not complete")
        require(protocol.get("completed_runs") == EXPECTED_RUNS_PER_WAVE, f"{wave_id}: wrong run count")
        require(protocol.get("successful_runs") == EXPECTED_RUNS_PER_WAVE, f"{wave_id}: unsuccessful run")
        require(protocol.get("unsuccessful_runs") == 0, f"{wave_id}: unsuccessful run recorded")
        measured = protocol.get("protocol")
        require(isinstance(measured, dict), f"{wave_id}: missing measured protocol")
        require(measured.get("fixed_wave") == wave_id, f"{wave_id}: protocol wave mismatch")
        require(measured.get("n_values") == [n_orders], f"{wave_id}: not the complete wave size")
        require(measured.get("repeats") == REPEATS, f"{wave_id}: wrong repeats")
        require(measured.get("solver_seeds") == list(SOLVER_SEEDS), f"{wave_id}: wrong seeds")
        require(
            measured.get("schedule_seed") == wave_metadata[wave_id]["schedule_seed"],
            f"{wave_id}: schedule seed mismatch",
        )
        require(
            measured.get("subset_seed_base") == wave_metadata[wave_id]["subset_seed"],
            f"{wave_id}: subset seed mismatch",
        )
        software = protocol.get("software")
        require(isinstance(software, dict), f"{wave_id}: missing software block")
        require(software.get("release_was_required") is True, f"{wave_id}: Release was not required")
        require(str(software.get("cmake_build_type_detected", "")).casefold() == "release", f"{wave_id}: not Release")
        require(software.get("cli_sha256") == cli_sha, f"{wave_id}: CLI hash mismatch")
        require(software.get("orchestrator_sha256") == runner_sha, f"{wave_id}: runner hash mismatch")
        data = protocol.get("data")
        require(isinstance(data, dict), f"{wave_id}: missing data block")
        require(data.get("pick_data_sha256") == pick_sha, f"{wave_id}: pick hash mismatch")
        require(data.get("dist_matrix_sha256") == distance_sha, f"{wave_id}: distance hash mismatch")

        protocol_experiments = protocol.get("experiments")
        require(isinstance(protocol_experiments, list), f"{wave_id}: missing experiments")
        require(
            {item.get("label") for item in protocol_experiments if isinstance(item, dict)} == expected_labels,
            f"{wave_id}: experiment labels differ",
        )
        for item in protocol_experiments:
            require(isinstance(item, dict), f"{wave_id}: malformed experiment metadata")
            label = str(item.get("label"))
            expected = expected_by_label[label]
            require(item.get("solver") == expected["solver"], f"{wave_id}/{label}: solver mismatch")
            require(item.get("declared_config_id") == expected["config_id"], f"{wave_id}/{label}: config id mismatch")
            require(item.get("config_sha256") == expected["config_sha256"], f"{wave_id}/{label}: config hash mismatch")

        runs_path = wave_dir / "runs.csv"
        require(runs_path.is_file(), f"missing {runs_path}")
        with runs_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {
                "wave_id", "total_orders_in_wave", "n_orders", "repeat_index", "subset_seed",
                "experiment_label", "solver", "declared_config_id", "observed_config_id",
                "config_sha256", "solver_seed", "status", "return_code", "timed_out",
                "feasible", "objective", "solver_runtime_ms", "wall_process_ms",
                "reported_order_count", "reported_wave_ids",
            }
            require(reader.fieldnames is not None and required_fields.issubset(reader.fieldnames), f"{wave_id}: runs.csv schema differs")
            rows = list(reader)
        require(len(rows) == EXPECTED_RUNS_PER_WAVE, f"{wave_id}: runs.csv must have 80 rows")
        seen_pairs: set[tuple[str, int]] = set()
        for row_number, row in enumerate(rows, start=2):
            context = f"{wave_id} runs.csv line {row_number}"
            label = row["experiment_label"]
            require(label in expected_by_label, f"{context}: unknown experiment {label}")
            expected = expected_by_label[label]
            seed = as_int(row["solver_seed"], f"{context} solver seed")
            require(seed in SOLVER_SEEDS, f"{context}: unexpected solver seed")
            require((label, seed) not in seen_pairs, f"{context}: duplicate experiment/seed")
            seen_pairs.add((label, seed))
            require(row["status"] == "success", f"{context}: status is not success")
            require(as_int(row["return_code"], f"{context} return code") == 0, f"{context}: nonzero return code")
            require(is_false(row["timed_out"]), f"{context}: timed out")
            require(is_true(row["feasible"]), f"{context}: infeasible")
            require(row["wave_id"] == wave_id, f"{context}: wave mismatch")
            require(as_int(row["total_orders_in_wave"], context) == n_orders, f"{context}: total size mismatch")
            require(as_int(row["n_orders"], context) == n_orders, f"{context}: subset is not full wave")
            require(as_int(row["reported_order_count"], context) == n_orders, f"{context}: reported count mismatch")
            require(row["reported_wave_ids"] == wave_id, f"{context}: reported wave mismatch")
            require(as_int(row["repeat_index"], context) == 0, f"{context}: repeat index mismatch")
            require(as_int(row["subset_seed"], context) == wave_metadata[wave_id]["subset_seed"], f"{context}: subset seed mismatch")
            require(row["solver"] == expected["solver"], f"{context}: solver mismatch")
            require(row["declared_config_id"] == expected["config_id"], f"{context}: declared config id mismatch")
            require(row["observed_config_id"] == expected["config_id"], f"{context}: observed config id mismatch")
            require(row["config_sha256"] == expected["config_sha256"], f"{context}: config hash mismatch")
            row["solver_seed"] = seed
            row["objective"] = as_float(row["objective"], f"{context} objective")
            row["solver_runtime_ms"] = as_float(row["solver_runtime_ms"], f"{context} solver runtime", nonnegative=True)
            row["wall_process_ms"] = as_float(row["wall_process_ms"], f"{context} wall runtime", nonnegative=True)
            all_records.append(row)
        require(
            seen_pairs == {(label, seed) for label in expected_labels for seed in SOLVER_SEEDS},
            f"{wave_id}: incomplete experiment/seed grid",
        )
    require(len(all_records) == EXPECTED_TOTAL_RUNS, "combined study does not contain 960 successful runs")
    return all_records


def aggregate_full_instance(
    records: Sequence[dict[str, Any]], expected_by_label: dict[str, dict[str, str | int]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["experiment_label"], int(record["solver_seed"]))].append(record)
    output: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        expected = expected_by_label[spec.label]
        for seed in SOLVER_SEEDS:
            selected = grouped[(spec.label, seed)]
            require(len(selected) == len(WAVES), f"{spec.label}/seed {seed}: not twelve waves")
            require({row["wave_id"] for row in selected} == {wave for wave, _ in WAVES}, f"{spec.label}/seed {seed}: wave set differs")
            output.append(
                {
                    "experiment_label": spec.label,
                    "solver": spec.solver,
                    "family": spec.family,
                    "budget": spec.budget,
                    "budget_label": spec.budget_label,
                    "config_id": expected["config_id"],
                    "config_sha256": expected["config_sha256"],
                    "solver_seed": seed,
                    "wave_count": len(selected),
                    "order_count_total": sum(as_int(row["n_orders"], "n_orders") for row in selected),
                    "objective_total": math.fsum(float(row["objective"]) for row in selected),
                    "solver_runtime_ms_total": math.fsum(float(row["solver_runtime_ms"]) for row in selected),
                    "wall_process_ms_total": math.fsum(float(row["wall_process_ms"]) for row in selected),
                }
            )
    require(len(output) == len(EXPERIMENTS) * len(SOLVER_SEEDS), "expected 80 full-instance rows")
    return output


def make_paired_gains(full_runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["experiment_label"], row["solver_seed"]): row for row in full_runs}
    output: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        for seed in SOLVER_SEEDS:
            row = indexed[(spec.label, seed)]
            greedy = indexed[("greedy", seed)]
            baseline_objective = float(greedy["objective_total"])
            baseline_runtime = float(greedy["solver_runtime_ms_total"])
            require(baseline_objective != 0.0, "greedy objective is zero; percentage gain undefined")
            require(baseline_runtime > 0.0, "greedy runtime is zero; runtime ratio undefined")
            gain = baseline_objective - float(row["objective_total"])
            runtime_delta = float(row["solver_runtime_ms_total"]) - baseline_runtime
            output.append(
                {
                    "experiment_label": spec.label,
                    "family": spec.family,
                    "budget": spec.budget,
                    "budget_label": spec.budget_label,
                    "solver_seed": seed,
                    "objective_total": row["objective_total"],
                    "greedy_objective_total": baseline_objective,
                    "paired_gain_abs": gain,
                    "paired_gain_pct": 100.0 * gain / baseline_objective,
                    "solver_runtime_ms_total": row["solver_runtime_ms_total"],
                    "greedy_solver_runtime_ms_total": baseline_runtime,
                    "paired_runtime_delta_ms": runtime_delta,
                    "paired_runtime_ratio": float(row["solver_runtime_ms_total"]) / baseline_runtime,
                }
            )
    return output


def summarize(
    full_runs: Sequence[dict[str, Any]], paired: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        selected = [row for row in full_runs if row["experiment_label"] == spec.label]
        gains = [row for row in paired if row["experiment_label"] == spec.label]
        summary: dict[str, Any] = {
            "experiment_label": spec.label,
            "solver": spec.solver,
            "family": spec.family,
            "budget": spec.budget,
            "budget_label": spec.budget_label,
            "n_paired_seeds": len(selected),
        }
        summary.update(descriptive_fields([float(row["objective_total"]) for row in selected], "objective", best="min"))
        summary.update(descriptive_fields([float(row["solver_runtime_ms_total"]) for row in selected], "solver_runtime_ms", best="min"))
        summary.update(descriptive_fields([float(row["wall_process_ms_total"]) for row in selected], "wall_process_ms", best="min"))
        summary.update(descriptive_fields([float(row["paired_gain_abs"]) for row in gains], "paired_gain_abs", best="max"))
        summary.update(descriptive_fields([float(row["paired_gain_pct"]) for row in gains], "paired_gain_pct", best="max"))
        summary.update(descriptive_fields([float(row["paired_runtime_delta_ms"]) for row in gains], "paired_runtime_delta_ms", best="min"))
        summary.update(descriptive_fields([float(row["paired_runtime_ratio"]) for row in gains], "paired_runtime_ratio", best="min"))
        summaries.append(summary)
    return summaries


def plot_time_quality(path: Path, summaries: Sequence[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    by_label = {row["experiment_label"]: row for row in summaries}
    fig, ax = plt.subplots(figsize=(6.7, 3.8), constrained_layout=True)
    styles = {"SA": ("#1f77b4", "o"), "GA": ("#d95f02", "s")}
    for family in ("SA", "GA"):
        specs = sorted((spec for spec in EXPERIMENTS if spec.family == family), key=lambda item: item.budget)
        x = [float(by_label[spec.label]["solver_runtime_ms_mean"]) / 1000.0 for spec in specs]
        y = [float(by_label[spec.label]["paired_gain_pct_mean"]) for spec in specs]
        low = [float(by_label[spec.label]["paired_gain_pct_t95_low"]) for spec in specs]
        high = [float(by_label[spec.label]["paired_gain_pct_t95_high"]) for spec in specs]
        yerr = [[center - lower for center, lower in zip(y, low)], [upper - center for center, upper in zip(y, high)]]
        color, marker = styles[family]
        ax.plot(x, y, color=color, marker=marker, linewidth=1.5, markersize=5, label=family)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, capsize=3, linewidth=1)
        for x_value, y_value, spec in zip(x, y, specs):
            ax.annotate(spec.budget_label, (x_value, y_value), xytext=(4, 5), textcoords="offset points", fontsize=7, color=color)
    greedy = by_label["greedy"]
    ax.scatter(
        [float(greedy["solver_runtime_ms_mean"]) / 1000.0],
        [0.0],
        color="#333333",
        marker="*",
        s=65,
        label="greedy",
        zorder=4,
    )
    ax.set_xscale("log")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_xlabel("Gemiddelde rekentijd over twaalf waves (s, log-schaal)")
    ax.set_ylabel("Gekoppelde winst t.o.v. greedy (%)")
    ax.set_title("Tijd-kwaliteitscurve met Student-t 95%-BI over tien seeds")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}".replace(".", ",")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}".replace(".", ",")))
    ax.legend(frameon=False)
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return matplotlib.__version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all 960 Release measurements, aggregate twelve waves per "
            "configuration/seed and compute descriptive and paired statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--study-dir", required=True, help="completed output of retrospective_budget_run.py")
    parser.add_argument(
        "--out-dir",
        help="new analysis directory; defaults to STUDY_DIR/analysis and may not exist",
    )
    return parser


def run(args: argparse.Namespace, root: Path) -> int:
    study_dir = Path(args.study_dir).expanduser().resolve()
    if not study_dir.is_dir():
        raise ValueError(f"study directory does not exist: {study_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else study_dir / "analysis"
    if os.path.lexists(out_dir):
        raise ValueError(f"analysis output already exists: {out_dir}; choose a new directory")
    expected_metadata, expected_by_label = expected_experiment_metadata(root)
    manifest, wave_metadata = validate_manifest(study_dir, root, expected_metadata)
    records = validate_and_load_runs(study_dir, manifest, wave_metadata, expected_by_label)
    full_runs = aggregate_full_instance(records, expected_by_label)
    paired = make_paired_gains(full_runs)
    summaries = summarize(full_runs, paired)

    out_dir.mkdir(parents=True, exist_ok=False)
    full_path = out_dir / "full_instance_runs.csv"
    paired_path = out_dir / "paired_gains.csv"
    summary_path = out_dir / "summary.csv"
    write_csv(full_path, full_runs)
    write_csv(paired_path, paired)
    write_csv(summary_path, summaries)
    matplotlib_version = plot_time_quality(out_dir / "time_quality_curve.png", summaries)
    analysis_protocol = {
        "schema_version": 1,
        "state": "complete",
        "created_utc": utc_now(),
        "study_dir": str(study_dir),
        "study_manifest_sha256": sha256_file(study_dir / "study_manifest.json"),
        "validated_raw_runs": len(records),
        "full_instance_rows": len(full_runs),
        "summary_rows": len(summaries),
        "aggregation": (
            "For each experiment and solver seed, objective, solver_runtime_ms and "
            "wall_process_ms are summed over the twelve complete waves."
        ),
        "pairing": "Each configuration is paired with greedy on the same solver seed.",
        "gain_definition": "greedy objective minus candidate objective; positive is better",
        "percent_gain_definition": "100 * (greedy - candidate) / greedy, computed per seed",
        "dispersion": "sample standard deviation and linearly interpolated quartiles",
        "confidence_interval": {
            "kind": "two-sided Student-t interval for the arithmetic mean",
            "confidence": 0.95,
            "degrees_of_freedom": 9,
            "critical_value": T_CRITICAL_95_DF9,
            "sampling_unit": "paired solver seed after aggregation over twelve waves",
        },
        "matplotlib_version": matplotlib_version,
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                full_path,
                paired_path,
                summary_path,
                out_dir / "time_quality_curve.png",
                out_dir / "time_quality_curve.pdf",
            )
        },
    }
    (out_dir / "analysis_protocol.json").write_text(
        json.dumps(analysis_protocol, indent=2), encoding="utf-8"
    )
    print(f"Validated {len(records)} successful Release runs.")
    print(f"Full-instance seed totals: {full_path}")
    print(f"Statistical summary:       {summary_path}")
    print(f"Time-quality figure:       {out_dir / 'time_quality_curve.pdf'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args, repo_root())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
