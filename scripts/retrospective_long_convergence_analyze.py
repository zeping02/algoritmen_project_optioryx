"""Validate and summarize a completed long-horizon convergence study."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from retrospective_long_convergence_common import (
    EXPERIMENTS,
    WAVES,
    repo_root,
    sha256_file,
    validate_configuration_files,
)
from retrospective_long_convergence_run import trajectory_metrics


T_CRITICAL_95: dict[int, float] = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183564,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.30600413503,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282966,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    21: 2.07961384473,
    22: 2.0738730679,
    23: 2.06865761042,
    24: 2.06389856163,
    25: 2.05953855275,
    26: 2.05552943864,
    27: 2.05183051648,
    28: 2.0484071418,
    29: 2.04522964213,
    30: 2.0422724563,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile of empty data")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def descriptive(values: Sequence[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": "",
            f"{prefix}_sd": "",
            f"{prefix}_min": "",
            f"{prefix}_q1": "",
            f"{prefix}_median": "",
            f"{prefix}_q3": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_min": min(values),
        f"{prefix}_q1": quantile(values, 0.25),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_q3": quantile(values, 0.75),
        f"{prefix}_max": max(values),
    }


def mean_ci95(values: Sequence[float], prefix: str) -> dict[str, Any]:
    fields = descriptive(values, prefix)
    if len(values) < 2:
        fields[f"{prefix}_ci95_low"] = ""
        fields[f"{prefix}_ci95_high"] = ""
        return fields
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom not in T_CRITICAL_95:
        raise ValueError(
            f"no locked Student-t critical value for {len(values)} paired observations"
        )
    half_width = (
        T_CRITICAL_95[degrees_of_freedom]
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    center = statistics.fmean(values)
    fields[f"{prefix}_ci95_low"] = center - half_width
    fields[f"{prefix}_ci95_high"] = center + half_width
    return fields


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_dense_trajectory(path: Path, expected_budget: int) -> list[float]:
    values: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "step",
            "best_objective_total",
            "gain_from_start",
            "fraction_of_final_gain",
        }
        require(reader.fieldnames is not None and set(reader.fieldnames) == required, f"bad schema: {path}")
        for row_number, row in enumerate(reader, start=2):
            step = int(row["step"])
            require(step == len(values), f"non-contiguous step in {path} line {row_number}")
            values.append(float(row["best_objective_total"]))
    require(len(values) == expected_budget + 1, f"wrong trajectory length: {path}")
    return values


def flatten_run_metrics(record: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": record["label"],
        "solver": record["solver"],
        "family": record["family"],
        "budget": record["budget"],
        "seed": record["seed"],
        "config_id": record["config_id"],
        "solver_runtime_ms": record["solver_runtime_ms"],
        "wave_solver_runtime_ms_sum": record["wave_solver_runtime_ms_sum"],
        "instance_overhead_runtime_ms": record["instance_overhead_runtime_ms"],
        "wall_process_ms": record["wall_process_ms"],
    }
    row.update(metrics)
    return row


def load_and_validate_records(
    study_dir: Path, manifest: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int], list[float]],
]:
    specs = {spec.label: spec for spec in EXPERIMENTS}
    seeds = tuple(int(value) for value in manifest["protocol"]["seeds"])
    expected_jobs = {f"{label}__seed-{seed:02d}" for label in specs for seed in seeds}
    require(manifest.get("state") == "complete", "study manifest is not complete")
    require(manifest.get("planned_jobs") == len(expected_jobs), "wrong planned job count")
    require(manifest.get("completed_jobs") == len(expected_jobs), "wrong completed job count")

    run_rows: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, int], list[float]] = {}
    observed_jobs: set[str] = set()
    for job_id in manifest["protocol"]["job_order"]:
        require(job_id in expected_jobs, f"unexpected job id in schedule: {job_id}")
        record_path = study_dir / "records" / f"{job_id}.json"
        require(record_path.is_file(), f"missing record: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        require(record.get("job_id") == job_id, f"record/job mismatch: {record_path}")
        require(job_id not in observed_jobs, f"duplicate job: {job_id}")
        observed_jobs.add(job_id)
        label = str(record["label"])
        spec = specs[label]
        require(record["solver"] == spec.solver, f"solver mismatch: {job_id}")
        require(record["family"] == spec.family, f"family mismatch: {job_id}")
        require(int(record["budget"]) == spec.budget, f"budget mismatch: {job_id}")

        trajectory_info = record["trajectory_file"]
        trajectory_path = Path(trajectory_info["path"]).resolve()
        require(trajectory_path.parent == (study_dir / "trajectories").resolve(), f"trajectory escaped study: {job_id}")
        require(trajectory_path.is_file(), f"missing trajectory: {trajectory_path}")
        require(sha256_file(trajectory_path) == trajectory_info["sha256"], f"trajectory hash mismatch: {job_id}")
        best = load_dense_trajectory(trajectory_path, spec.budget)
        trajectory_key = (label, int(record["seed"]))
        require(trajectory_key not in trajectories, f"duplicate trajectory: {job_id}")
        trajectories[trajectory_key] = best
        metrics = trajectory_metrics(best)
        recorded_metrics = record["full_instance_metrics"]
        for key, value in metrics.items():
            observed = recorded_metrics[key]
            if isinstance(value, float):
                require(math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-9), f"metric mismatch {job_id}/{key}")
            else:
                require(observed == value, f"metric mismatch {job_id}/{key}")
        run_rows.append(flatten_run_metrics(record, metrics))

        changes_info = record["wave_changepoints_file"]
        changes_path = Path(changes_info["path"]).resolve()
        require(changes_path.parent == (study_dir / "wave_changepoints").resolve(), f"change file escaped study: {job_id}")
        require(changes_path.is_file(), f"missing change points: {changes_path}")
        require(sha256_file(changes_path) == changes_info["sha256"], f"change-point hash mismatch: {job_id}")
        metrics_by_wave = record["wave_metrics"]
        require(
            [row["wave_id"] for row in metrics_by_wave] == [wave_id for wave_id, _ in WAVES],
            f"wrong wave set: {job_id}",
        )
        for wave_metric in metrics_by_wave:
            wave_row = {
                "label": label,
                "solver": spec.solver,
                "family": spec.family,
                "budget": spec.budget,
                "seed": int(record["seed"]),
            }
            wave_row.update(wave_metric)
            wave_rows.append(wave_row)
    require(observed_jobs == expected_jobs, "completed job set differs from locked grid")
    require(len(wave_rows) == len(expected_jobs) * len(WAVES), "wrong wave-row count")
    require(len(trajectories) == len(expected_jobs), "wrong dense-trajectory count")
    return run_rows, wave_rows, trajectories


def summarize_runs(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["label"])].append(row)
    summaries: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        selected = grouped[spec.label]
        improvers = [row for row in selected if not row["no_improvement"]]
        summary: dict[str, Any] = {
            "label": spec.label,
            "solver": spec.solver,
            "family": spec.family,
            "budget": spec.budget,
            "seed_count": len(selected),
            "no_improvement_count": sum(bool(row["no_improvement"]) for row in selected),
            "improved_at_edge_count": sum(bool(row["improved_at_edge"]) for row in selected),
            "improved_in_last_ten_percent_count": sum(
                bool(row["improved_in_last_ten_percent"]) for row in selected
            ),
        }
        summary.update(descriptive([float(row["final_objective"]) for row in selected], "final_objective"))
        summary.update(descriptive([float(row["total_gain"]) for row in selected], "total_gain"))
        summary.update(descriptive([float(row["solver_runtime_ms"]) for row in selected], "solver_runtime_ms"))
        summary.update(descriptive([float(row["wall_process_ms"]) for row in selected], "wall_process_ms"))
        summary.update(descriptive([float(row["t95"]) for row in improvers], "t95_improvers"))
        summary.update(descriptive([float(row["t99"]) for row in improvers], "t99_improvers"))
        summary.update(
            descriptive([float(row["last_improvement"]) for row in improvers], "last_improvement_improvers")
        )
        summary.update(
            descriptive(
                [float(row["share_gain_last_ten_percent"]) for row in improvers],
                "share_gain_last_ten_percent_improvers",
            )
        )
        summaries.append(summary)
    return summaries


def summarize_waves(wave_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in wave_rows:
        grouped[(str(row["label"]), str(row["wave_id"]))].append(row)
    order_counts = dict(WAVES)
    summaries: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        for wave_id, _order_count in WAVES:
            selected = grouped[(spec.label, wave_id)]
            improvers = [row for row in selected if not row["no_improvement"]]
            summary: dict[str, Any] = {
                "label": spec.label,
                "solver": spec.solver,
                "family": spec.family,
                "budget": spec.budget,
                "wave_id": wave_id,
                "order_count": order_counts[wave_id],
                "seed_count": len(selected),
                "no_improvement_count": sum(bool(row["no_improvement"]) for row in selected),
                "improved_at_edge_count": sum(bool(row["improved_at_edge"]) for row in selected),
                "improved_in_last_ten_percent_count": sum(
                    bool(row["improved_in_last_ten_percent"]) for row in selected
                ),
            }
            summary.update(
                descriptive([float(row["final_objective"]) for row in selected], "final_objective")
            )
            summary.update(
                descriptive([float(row["total_gain"]) for row in selected], "total_gain")
            )
            summary.update(
                descriptive(
                    [float(row["gain_last_ten_percent"]) for row in selected],
                    "gain_last_ten_percent",
                )
            )
            summary.update(
                descriptive([float(row["solver_runtime_ms"]) for row in selected], "solver_runtime_ms")
            )
            summary.update(
                descriptive([float(row["t95"]) for row in improvers], "t95_improvers")
            )
            summary.update(
                descriptive([float(row["t99"]) for row in improvers], "t99_improvers")
            )
            summary.update(
                descriptive(
                    [float(row["last_improvement"]) for row in improvers],
                    "last_improvement_improvers",
                )
            )
            summary.update(
                descriptive(
                    [float(row["share_gain_last_ten_percent"]) for row in improvers],
                    "share_gain_last_ten_percent_improvers",
                )
            )
            summaries.append(summary)
    return summaries


def paired_budget_rows(run_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(str(row["label"]), int(row["seed"])): row for row in run_rows}
    comparisons = (
        ("SA", "sa_i10000", "sa_i50000", "independent_budget_specific_schedules"),
        ("GA", "ga_g40", "ga_g100", "same_seed_separate_generation_horizons"),
        ("GA", "ga_g100", "ga_g200", "same_seed_separate_generation_horizons"),
        ("GA", "ga_g40", "ga_g200", "same_seed_separate_generation_horizons"),
    )
    seeds = sorted({int(row["seed"]) for row in run_rows})
    output: list[dict[str, Any]] = []
    for family, short_label, long_label, comparison_kind in comparisons:
        for seed in seeds:
            short = indexed[(short_label, seed)]
            long = indexed[(long_label, seed)]
            short_objective = float(short["final_objective"])
            long_objective = float(long["final_objective"])
            added_gain = short_objective - long_objective
            output.append(
                {
                    "family": family,
                    "short_label": short_label,
                    "long_label": long_label,
                    "comparison_kind": comparison_kind,
                    "seed": seed,
                    "short_final_objective": short_objective,
                    "long_final_objective": long_objective,
                    "added_gain_abs": added_gain,
                    "added_gain_pct_vs_short": 100.0 * added_gain / short_objective,
                    "short_solver_runtime_ms": short["solver_runtime_ms"],
                    "long_solver_runtime_ms": long["solver_runtime_ms"],
                    "runtime_ratio": float(long["solver_runtime_ms"]) / float(short["solver_runtime_ms"]),
                }
            )
    return output


def summarize_paired_budgets(paired: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(str(row["short_label"]), str(row["long_label"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for short_label, long_label in (
        ("sa_i10000", "sa_i50000"),
        ("ga_g40", "ga_g100"),
        ("ga_g100", "ga_g200"),
        ("ga_g40", "ga_g200"),
    ):
        selected = grouped[(short_label, long_label)]
        require(bool(selected), f"missing paired comparison {short_label}/{long_label}")
        summary: dict[str, Any] = {
            "family": selected[0]["family"],
            "short_label": short_label,
            "long_label": long_label,
            "comparison_kind": selected[0]["comparison_kind"],
            "paired_seed_count": len(selected),
            "long_budget_better_count": sum(float(row["added_gain_abs"]) > 1e-9 for row in selected),
            "equal_within_tolerance_count": sum(
                abs(float(row["added_gain_abs"])) <= 1e-9 for row in selected
            ),
            "long_budget_worse_count": sum(float(row["added_gain_abs"]) < -1e-9 for row in selected),
        }
        summary.update(
            mean_ci95([float(row["added_gain_abs"]) for row in selected], "added_gain_abs")
        )
        summary.update(
            mean_ci95(
                [float(row["added_gain_pct_vs_short"]) for row in selected],
                "added_gain_pct_vs_short",
            )
        )
        summary.update(
            mean_ci95([float(row["runtime_ratio"]) for row in selected], "runtime_ratio")
        )
        summaries.append(summary)
    return summaries


def audit_ga_objective_prefixes(
    trajectories: dict[tuple[str, int], list[float]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    comparisons = (
        ("ga_g40", "ga_g100"),
        ("ga_g40", "ga_g200"),
        ("ga_g100", "ga_g200"),
    )
    rows: list[dict[str, Any]] = []
    for short_label, long_label in comparisons:
        for seed in seeds:
            short = trajectories[(short_label, seed)]
            long = trajectories[(long_label, seed)]
            require(len(long) >= len(short), f"GA long horizon shorter for seed {seed}")
            differences = [abs(left - right) for left, right in zip(short, long)]
            mismatch = next(
                (step for step, difference in enumerate(differences) if difference > 1e-9),
                None,
            )
            rows.append(
                {
                    "short_label": short_label,
                    "long_label": long_label,
                    "seed": seed,
                    "points_compared": len(short),
                    "max_abs_difference": max(differences, default=0.0),
                    "first_mismatch_step": "" if mismatch is None else mismatch,
                    "prefix_consistent": mismatch is None,
                }
            )
    return rows


def plot_long_convergence(
    path: Path,
    trajectories: dict[tuple[str, int], list[float]],
    seeds: Sequence[int],
) -> tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    styles = {
        "sa_i10000": ("#e66101", "SA 10.000"),
        "sa_i50000": ("#5e3c99", "SA 50.000"),
        "ga_g40": ("#1b9e77", "GA 40"),
        "ga_g100": ("#377eb8", "GA 100"),
        "ga_g200": ("#e41a1c", "GA 200"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.35), constrained_layout=True)
    for ax, family, labels, x_label, title in (
        (
            axes[0],
            "SA",
            ("sa_i10000", "sa_i50000"),
            "iteratie",
            "SA: afzonderlijke koelschema's",
        ),
        (
            axes[1],
            "GA",
            ("ga_g40", "ga_g100", "ga_g200"),
            "generatie",
            "GA: afzonderlijke generatiehorizonten",
        ),
    ):
        for label in labels:
            matrix = np.asarray([trajectories[(label, seed)] for seed in seeds], dtype=float)
            starts = matrix[:, [0]]
            require(bool(np.all(starts > 0.0)), f"non-positive start objective for {label}")
            gain_percent = 100.0 * (starts - matrix) / starts
            mean = np.mean(gain_percent, axis=0)
            if len(seeds) > 1:
                critical = T_CRITICAL_95[len(seeds) - 1]
                half_width = critical * np.std(gain_percent, axis=0, ddof=1) / math.sqrt(len(seeds))
            else:
                half_width = np.zeros_like(mean)
            steps = np.arange(matrix.shape[1])
            color, display_label = styles[label]
            ax.plot(steps, mean, color=color, linewidth=1.5, label=display_label)
            ax.fill_between(
                steps,
                mean - half_width,
                mean + half_width,
                color=color,
                alpha=0.14,
                linewidth=0.0,
            )
            budget = len(steps) - 1
            ax.axvline(0.9 * budget, color=color, linestyle=":", linewidth=0.8, alpha=0.8)
        ax.axhline(0.0, color="#666666", linewidth=0.7)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:g}".replace(".", ","))
        )
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("winst t.o.v. eigen start (%)")
    fig.suptitle(
        "Volledige-instance best-so-far: gemiddelde en Student-t-95%-BI over seeds",
        fontsize=10,
    )
    fig.savefig(path, dpi=240)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    return matplotlib.__version__, np.__version__


def summary_markdown(
    summaries: Sequence[dict[str, Any]], paired_summaries: Sequence[dict[str, Any]]
) -> str:
    lines = [
        "# Long-convergence summary",
        "",
        "All statistics use the complete-instance best-so-far curve obtained by summing "
        "the twelve wave curves at each step.",
        "",
        "| config | seeds | no gain | t95 median [IQR], improvers | t99 median [IQR], improvers | last improvement median | improved in last 10% | edge improvements | solver time mean (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def interval(prefix: str) -> str:
            if int(row.get(f"{prefix}_n", 0)) == 0:
                return "--"
            return (
                f"{row[f'{prefix}_median']:.1f} "
                f"[{row[f'{prefix}_q1']:.1f}, {row[f'{prefix}_q3']:.1f}]"
            )

        last = "--"
        if int(row.get("last_improvement_improvers_n", 0)):
            last = f"{row['last_improvement_improvers_median']:.1f}"
        lines.append(
            "| {label} | {seed_count} | {no_improvement_count} | {t95} | {t99} | "
            "{last} | {last10} | {edge} | {runtime:.2f} |".format(
                label=row["label"],
                seed_count=row["seed_count"],
                no_improvement_count=row["no_improvement_count"],
                t95=interval("t95_improvers"),
                t99=interval("t99_improvers"),
                last=last,
                last10=row["improved_in_last_ten_percent_count"],
                edge=row["improved_at_edge_count"],
                runtime=float(row["solver_runtime_ms_mean"]) / 1000.0,
            )
        )
    lines.extend(
        [
            "",
            "Definitions: t95/t99 are the first steps reaching 95%/99% of each run's "
            "eventual within-budget gain. Runs without a strict gain have blank t95/t99. "
            "An edge improvement occurs at the final allowed step.",
            "",
            "Exact seconds-to-t95/t99 are unavailable because convergence points carry no "
            "per-step timestamps. Only the exact step and complete-run/per-wave times are recorded.",
            "",
            "SA 10k and SA 50k independently cool from 60 to 0.05 with budget-specific "
            "geometric factors. They are paired alternative schedules, not a 10k prefix "
            "continued for another 40k iterations.",
            "",
            "GA uses one RNG stream across the twelve sequential waves. Changing the "
            "generation horizon changes how many draws wave 1 consumes, so later waves "
            "start from different RNG states. The 40/100/200 full-instance curves are "
            "therefore same-seed alternative horizons, not continuations of one shared "
            "full-instance trajectory. Objective-prefix equality is audited descriptively.",
            "",
            "## Paired budget contrasts",
            "",
            "| short | long | kind | n | mean added gain [95% CI] | mean runtime ratio [95% CI] |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in paired_summaries:
        def mean_interval(prefix: str) -> str:
            center = row[f"{prefix}_mean"]
            low = row[f"{prefix}_ci95_low"]
            high = row[f"{prefix}_ci95_high"]
            if low == "" or high == "":
                return f"{float(center):.3f} [--]"
            return f"{float(center):.3f} [{float(low):.3f}, {float(high):.3f}]"

        lines.append(
            "| {short} | {long} | {kind} | {n} | {gain} | {runtime} |".format(
                short=row["short_label"],
                long=row["long_label"],
                kind=row["comparison_kind"],
                n=row["paired_seed_count"],
                gain=mean_interval("added_gain_abs"),
                runtime=mean_interval("runtime_ratio"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a completed long-convergence study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--study-dir", required=True, help="completed study directory")
    parser.add_argument(
        "--out-dir", help="new output directory; defaults to STUDY_DIR/analysis"
    )
    return parser


def run(args: argparse.Namespace, root: Path) -> int:
    study_dir = Path(args.study_dir).expanduser().resolve()
    if not study_dir.is_dir():
        raise ValueError(f"study directory does not exist: {study_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else study_dir / "analysis"
    if os.path.lexists(out_dir):
        raise ValueError(f"analysis directory already exists: {out_dir}")
    manifest_path = study_dir / "study_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing study manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    current_metadata = validate_configuration_files(root)
    recorded_metadata = manifest.get("experiments")
    require(recorded_metadata == current_metadata, "current locked configs differ from measured configs")
    software = manifest.get("software", {})
    cli_path = Path(str(software.get("cli_path", "")))
    require(cli_path.is_file(), "recorded CLI no longer exists")
    require(sha256_file(cli_path) == software.get("cli_sha256"), "recorded CLI hash changed")
    data = manifest.get("data", {})
    require(
        sha256_file(root / "data" / "pick_data.csv") == data.get("pick_data_sha256"),
        "pick_data.csv hash changed",
    )
    require(
        sha256_file(root / "data" / "dist_matrix.csv") == data.get("dist_matrix_sha256"),
        "dist_matrix.csv hash changed",
    )

    run_rows, wave_rows, trajectories = load_and_validate_records(study_dir, manifest)
    seeds = tuple(int(value) for value in manifest["protocol"]["seeds"])
    summaries = summarize_runs(run_rows)
    wave_summaries = summarize_waves(wave_rows)
    paired = paired_budget_rows(run_rows)
    paired_summaries = summarize_paired_budgets(paired)
    ga_prefix_rows = audit_ga_objective_prefixes(trajectories, seeds)
    out_dir.mkdir(parents=True, exist_ok=False)
    run_path = out_dir / "run_metrics.csv"
    wave_path = out_dir / "wave_metrics.csv"
    wave_summary_path = out_dir / "wave_summary.csv"
    summary_path = out_dir / "summary.csv"
    paired_path = out_dir / "paired_budget_comparisons.csv"
    paired_summary_path = out_dir / "paired_budget_summary.csv"
    ga_prefix_path = out_dir / "ga_objective_prefix_audit.csv"
    figure_path = out_dir / "long_convergence.png"
    markdown_path = out_dir / "summary.md"
    write_csv(run_path, run_rows)
    write_csv(wave_path, wave_rows)
    write_csv(wave_summary_path, wave_summaries)
    write_csv(summary_path, summaries)
    write_csv(paired_path, paired)
    write_csv(paired_summary_path, paired_summaries)
    write_csv(ga_prefix_path, ga_prefix_rows)
    matplotlib_version, numpy_version = plot_long_convergence(
        figure_path, trajectories, seeds
    )
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(summary_markdown(summaries, paired_summaries))

    protocol_path = out_dir / "analysis_protocol.json"
    protocol = {
        "schema_version": 1,
        "state": "complete",
        "created_utc": utc_now(),
        "study_dir": str(study_dir),
        "study_manifest_sha256": sha256_file(manifest_path),
        "validated_runs": len(run_rows),
        "validated_wave_runs": len(wave_rows),
        "wave_summary_rows": len(wave_summaries),
        "ga_objective_prefix_checks": len(ga_prefix_rows),
        "ga_objective_prefix_mismatches": sum(
            not bool(row["prefix_consistent"]) for row in ga_prefix_rows
        ),
        "matplotlib_version": matplotlib_version,
        "numpy_version": numpy_version,
        "definitions": {
            "sampling_unit": "solver seed after exact aggregation over twelve waves",
            "t95_t99": "first step reaching 95%/99% of the within-budget total gain",
            "last_improvement": "last strict best-so-far decrease larger than 1e-9",
            "edge_improvement": "last strict decrease occurs at the final budget step",
            "last_ten_percent_share": (
                "gain from floor(0.9*horizon) through the horizon divided by total gain"
            ),
            "quartiles": "linear interpolation at positions (n-1)*p",
            "runtime": (
                "internal complete-instance timer including evaluator setup and validation; "
                "the sum of twelve wave timers is retained separately"
            ),
            "seconds_to_t95_t99": (
                "unavailable: convergence points contain no per-step timestamps; exact t95/t99 "
                "are reported only in iterations/generations"
            ),
            "sa_budget_relation": (
                "SA 10k and 50k are independent schedules, each cooling from 60 to 0.05 "
                "with its own geometric alpha; SA 10k is not a prefix of SA 50k"
            ),
            "ga_budget_relation": (
                "GA 40/100/200 use the same seed labels but one RNG stream is consumed "
                "sequentially over waves. A longer horizon in an earlier wave shifts the "
                "RNG state for later waves, so full-instance curves are alternative "
                "horizons rather than shared prefixes. Best-objective prefix equality is "
                "recorded as a descriptive audit and is not required."
            ),
            "paired_budget_confidence_intervals": (
                "two-sided Student-t 95% intervals over paired solver-seed contrasts"
            ),
            "figure": (
                "mean percentage gain from each run's own start with two-sided Student-t "
                "95% intervals; dotted vertical lines mark 90% of each budget"
            ),
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "path": str(path)}
            for path in (
                run_path,
                wave_path,
                wave_summary_path,
                summary_path,
                paired_path,
                paired_summary_path,
                ga_prefix_path,
                figure_path,
                figure_path.with_suffix(".pdf"),
                markdown_path,
            )
        },
    }
    exclusive_path = protocol_path
    with exclusive_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    print(f"Validated {len(run_rows)} runs and {len(wave_rows)} wave trajectories.")
    print(f"Summary: {summary_path}")
    print(f"Wave summary: {wave_summary_path}")
    print(f"Paired contrasts: {paired_summary_path}")
    print(f"GA objective-prefix audit: {ga_prefix_path}")
    print(f"Convergence figure: {figure_path.with_suffix('.pdf')}")
    print(f"Readable summary: {markdown_path}")
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
