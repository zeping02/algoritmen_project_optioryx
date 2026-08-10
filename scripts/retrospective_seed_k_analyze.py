"""Audit and summarize the deterministic seed-selection timing/K study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from retrospective_seed_k_run import EXPERIMENTS, root_dir, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_float(value: Any, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number for {context}: {value!r}") from exc
    require(math.isfinite(parsed), f"non-finite number for {context}")
    return parsed


def linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "quantile requires observations")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Sequence[float], prefix: str) -> dict[str, float | int]:
    require(bool(values), f"no observations for {prefix}")
    q1 = linear_quantile(values, 0.25)
    q3 = linear_quantile(values, 0.75)
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_sample_sd": statistics.stdev(values) if len(values) > 1 else math.nan,
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": q3 - q1,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    require(bool(rows), f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate(study: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol_path = study / "protocol.json"
    runs_path = study / "runs.csv"
    require(protocol_path.is_file(), f"missing {protocol_path}")
    require(runs_path.is_file(), f"missing {runs_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("state") == "complete", "study protocol is not complete")
    repetitions = int(protocol.get("repetitions", 0))
    require(repetitions >= 2, "study needs repeated timings")
    expected_count = len(EXPERIMENTS) * repetitions
    require(protocol.get("expected_runs") == expected_count, "protocol run count differs")
    require(protocol.get("successful_runs") == expected_count, "protocol contains failed runs")
    software = protocol.get("software")
    require(isinstance(software, dict), "protocol lacks software metadata")
    cli_path = Path(str(software.get("cli", "")))
    require(cli_path.is_file(), f"measured CLI no longer exists: {cli_path}")
    require(software.get("cli_sha256") == sha256_file(cli_path), "measured CLI hash changed")
    data = protocol.get("data")
    require(isinstance(data, dict), "protocol lacks data metadata")
    require(
        data.get("pick_data_sha256") == sha256_file(root / "data" / "pick_data.csv"),
        "pick data hash changed",
    )
    require(
        data.get("dist_matrix_sha256") == sha256_file(root / "data" / "dist_matrix.csv"),
        "distance data hash changed",
    )
    metadata = protocol.get("experiments")
    require(isinstance(metadata, list) and len(metadata) == len(EXPERIMENTS), "experiment metadata differs")
    expected_labels = {item.label for item in EXPERIMENTS}
    observed_labels = {str(item.get("label")) for item in metadata if isinstance(item, dict)}
    require(observed_labels == expected_labels, "experiment labels differ")
    for item in metadata:
        require(isinstance(item, dict), "malformed experiment metadata")
        config_path = Path(str(item.get("config_path", "")))
        require(config_path.is_file(), f"missing measured config: {config_path}")
        require(item.get("config_sha256") == sha256_file(config_path), f"config changed: {config_path}")

    with runs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "schedule_index", "label", "seed_rule", "selection_rule", "lookahead_width",
            "repeat_index", "config_id", "config_sha256", "status", "return_code",
            "timed_out", "feasible", "objective", "solver_runtime_ms", "wall_process_ms",
            "wave_count",
        }
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames), "runs schema differs")
        raw_rows = list(reader)
    require(len(raw_rows) == expected_count, "runs.csv row count differs")
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for line_number, row in enumerate(raw_rows, start=2):
        context = f"runs.csv line {line_number}"
        label = row["label"]
        require(label in expected_labels, f"{context}: unknown label")
        repeat = int(row["repeat_index"])
        require(0 <= repeat < repetitions, f"{context}: repeat out of range")
        require((label, repeat) not in seen, f"{context}: duplicate label/repeat")
        seen.add((label, repeat))
        require(row["status"] == "success", f"{context}: failed status")
        require(int(row["return_code"]) == 0, f"{context}: nonzero return code")
        require(row["timed_out"].casefold() == "false", f"{context}: timed out")
        require(row["feasible"].casefold() == "true", f"{context}: infeasible")
        require(int(row["wave_count"]) == 12, f"{context}: not twelve waves")
        converted = dict(row)
        converted["repeat_index"] = repeat
        converted["objective"] = as_float(row["objective"], f"{context} objective")
        converted["solver_runtime_ms"] = as_float(row["solver_runtime_ms"], f"{context} internal time")
        converted["wall_process_ms"] = as_float(row["wall_process_ms"], f"{context} wall time")
        rows.append(converted)
    require(
        seen == {(label, repeat) for label in expected_labels for repeat in range(repetitions)},
        "incomplete label/repetition grid",
    )
    return protocol, rows


def summarize(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    experiment_by_label = {item.label: item for item in EXPERIMENTS}
    summaries: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        selected = grouped[experiment.label]
        objectives = [float(row["objective"]) for row in selected]
        require(max(objectives) == min(objectives), f"deterministic objective changed for {experiment.label}")
        summary: dict[str, Any] = {
            "label": experiment.label,
            "seed_rule": experiment.seed,
            "selection_rule": experiment.rule,
            "lookahead_width": experiment.k if experiment.k is not None else "",
            "objective": objectives[0],
        }
        summary.update(describe([float(row["solver_runtime_ms"]) for row in selected], "solver_runtime_ms"))
        summary.update(describe([float(row["wall_process_ms"]) for row in selected], "wall_process_ms"))
        summaries.append(summary)

    by_label = {str(row["label"]): row for row in summaries}
    for row in summaries:
        experiment = experiment_by_label[str(row["label"])]
        baseline = by_label[f"{experiment.seed}_immediate"]
        baseline_objective = float(baseline["objective"])
        row["gain_vs_immediate_abs"] = baseline_objective - float(row["objective"])
        row["gain_vs_immediate_pct"] = 100.0 * (
            baseline_objective - float(row["objective"])
        ) / baseline_objective
        row["median_internal_time_ratio_vs_immediate"] = (
            float(row["solver_runtime_ms_median"]) / float(baseline["solver_runtime_ms_median"])
        )
        row["median_wall_time_ratio_vs_immediate"] = (
            float(row["wall_process_ms_median"]) / float(baseline["wall_process_ms_median"])
        )
    k_rows = [dict(row) for row in summaries if row["selection_rule"] == "two_step"]
    k_rows.sort(key=lambda row: (str(row["seed_rule"]), int(row["lookahead_width"])))
    return summaries, k_rows


def make_plot(path: Path, k_rows: Sequence[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    styles = {"largest": ("#1f77b4", "o", "grootste span"), "smallest": ("#d95f02", "s", "kleinste span")}
    for seed, (color, marker, label) in styles.items():
        selected = sorted(
            (row for row in k_rows if row["seed_rule"] == seed),
            key=lambda row: int(row["lookahead_width"]),
        )
        k = [int(row["lookahead_width"]) for row in selected]
        gain = [float(row["gain_vs_immediate_pct"]) for row in selected]
        runtime = [float(row["solver_runtime_ms_median"]) / 1000.0 for row in selected]
        low = [float(row["solver_runtime_ms_q1"]) / 1000.0 for row in selected]
        high = [float(row["solver_runtime_ms_q3"]) / 1000.0 for row in selected]
        axes[0].plot(k, gain, color=color, marker=marker, label=label)
        axes[1].errorbar(
            k, runtime,
            yerr=[[center - lower for center, lower in zip(runtime, low)], [upper - center for center, upper in zip(runtime, high)]],
            color=color, marker=marker, capsize=3, label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_title("(a) Kwaliteitswinst t.o.v. directe kost")
    axes[0].set_ylabel("winst (%)")
    axes[1].set_title("(b) Interne uitvoeringstijd")
    axes[1].set_ylabel("mediaan [IQR] (s)")
    for axis in axes:
        axis.set_xlabel("voorselectiebreedte K")
        axis.set_xticks([2, 4, 8, 16])
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}".replace(".", ",")))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return matplotlib.__version__


def run(args: argparse.Namespace, root: Path) -> int:
    study = Path(args.study_dir).expanduser().resolve()
    require(study.is_dir(), f"study directory does not exist: {study}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else study / "analysis"
    if os.path.lexists(out_dir):
        raise ValueError(f"analysis output already exists: {out_dir}")
    protocol, rows = load_and_validate(study, root)
    summaries, k_rows = summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=False)
    summary_path = out_dir / "summary.csv"
    k_path = out_dir / "k_sensitivity.csv"
    plot_path = out_dir / "k_sensitivity.png"
    write_csv(summary_path, summaries)
    write_csv(k_path, k_rows)
    matplotlib_version = make_plot(plot_path, k_rows)
    analysis = {
        "schema_version": 1,
        "state": "complete",
        "study_dir": str(study),
        "study_protocol_sha256": sha256_file(study / "protocol.json"),
        "runs_sha256": sha256_file(study / "runs.csv"),
        "validated_runs": len(rows),
        "sampling_unit": "independent process invocation of a deterministic full-instance configuration",
        "quality_check": "objective must be exactly identical over all repetitions",
        "timing_summary": "median and linearly interpolated IQR over randomized-order repetitions",
        "quality_comparator": "same-seed-rule minimum incremental route-cost selection",
        "runtime_ratio": "ratio of configuration median to matched immediate-rule median",
        "matplotlib_version": matplotlib_version,
        "outputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in (summary_path, k_path, plot_path, plot_path.with_suffix(".pdf"))
        },
    }
    (out_dir / "analysis_protocol.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Validated {len(rows)} repeated deterministic runs.")
    print(f"Summary: {summary_path}")
    print(f"Figure:  {plot_path.with_suffix('.pdf')}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--study-dir", required=True)
    result.add_argument("--out-dir")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args, root_dir())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
