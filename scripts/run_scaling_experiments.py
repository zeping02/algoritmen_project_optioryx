"""Run a controlled empirical runtime-versus-order-count experiment.

The experiment deliberately samples every problem size from one competition
wave.  For each repeat, one reproducible random permutation of that wave's
orders is made; the instances for increasing ``n`` are nested prefixes of this
permutation.  The same subset is then presented to every solver/configuration
and solver seed.  This prevents wave identity from being confounded with the
number of orders.

Outputs are ``runs.csv`` (all invocations), ``summary.csv``, ``fits.csv``
(descriptive power-law fits), ``subset_manifest.csv``, ``protocol.json`` and
``runtime_vs_n.png`` plus a vector-quality ``runtime_vs_n.pdf``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MANAGED_OUTPUTS = (
    "runs.csv",
    "summary.csv",
    "fits.csv",
    "subset_manifest.csv",
    "protocol.json",
    "runtime_vs_n.png",
    "runtime_vs_n.pdf",
)


@dataclass(frozen=True)
class Experiment:
    label: str
    solver: str
    config_path: Path | None
    declared_config_id: str
    config_sha256: str


@dataclass(frozen=True)
class SubsetInstance:
    repeat_index: int
    subset_seed: int
    n_orders: int
    subset_id: str
    data_dir: Path
    static_materialization: str


def parse_integer_list(value: str, *, minimum: int, option_name: str) -> list[int]:
    try:
        result = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a comma-separated list of integers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError(f"{option_name} may not be empty")
    if any(number < minimum for number in result):
        raise argparse.ArgumentTypeError(
            f"every value in {option_name} must be at least {minimum}"
        )
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"{option_name} contains duplicate values")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def read_config_id(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if line.startswith("config_id="):
                value = line.partition("=")[2].strip()
                if value:
                    return value
    return fallback


def detect_build_type(cli_path: Path) -> tuple[str | None, str | None]:
    """Return CMake build type and cache path when the adjacent cache exists."""
    candidates = [cli_path.parent / "CMakeCache.txt", cli_path.parent.parent / "CMakeCache.txt"]
    for cache_path in candidates:
        if not cache_path.is_file():
            continue
        with cache_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("CMAKE_BUILD_TYPE:"):
                    return line.partition("=")[2].strip() or None, str(cache_path)
        return None, str(cache_path)
    return None, None


def total_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
        except (AttributeError, OSError):
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None


def available_cpu_count() -> int | None:
    try:
        affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]
        return len(affinity)
    except (AttributeError, OSError):
        return os.cpu_count()


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "changed_paths": status.splitlines() if status else [],
    }


def machine_metadata() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "hostname": uname.node,
        "operating_system": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_cpu_count": os.cpu_count(),
        "cpus_available_to_process": available_cpu_count(),
        "total_memory_bytes": total_memory_bytes(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "perf_counter": vars(time.get_clock_info("perf_counter")),
        "relevant_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBER_OF_PROCESSORS",
                "PROCESSOR_IDENTIFIER",
            )
        },
    }


def load_pick_data(
    pick_path: Path, wave_id: str
) -> tuple[list[str], dict[str, list[list[str]]], dict[str, int]]:
    required_columns = ("WAVE_ID", "ORDER_ID", "LOCATION_ID", "ASIS_SEQUENCE")
    wave_order_ids: dict[str, set[str]] = {}
    selected_rows: dict[str, list[list[str]]] = {}

    with pick_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"pick data is empty: {pick_path}") from exc
        missing = [column for column in required_columns if column not in header]
        if missing:
            raise ValueError(f"pick data is missing columns: {', '.join(missing)}")
        wave_column = header.index("WAVE_ID")
        order_column = header.index("ORDER_ID")

        for row_number, row in enumerate(reader, start=2):
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            if len(row) > len(header):
                raise ValueError(f"row {row_number} in pick_data.csv has too many fields")
            current_wave = row[wave_column].strip()
            order_id = row[order_column].strip()
            if not current_wave or not order_id:
                raise ValueError(
                    f"row {row_number} in pick_data.csv has an empty wave or order id"
                )
            wave_order_ids.setdefault(current_wave, set()).add(order_id)
            if current_wave == wave_id:
                selected_rows.setdefault(order_id, []).append(row)

    counts = {wave: len(order_ids) for wave, order_ids in wave_order_ids.items()}
    if not selected_rows:
        available = ", ".join(f"{wave} ({count})" for wave, count in sorted(counts.items()))
        raise ValueError(f"wave {wave_id!r} was not found; available waves: {available}")
    return header, selected_rows, counts


def materialize_static_files(data_dir: Path, destination: Path) -> str:
    methods: set[str] = set()
    for source in sorted(data_dir.iterdir()):
        if not source.is_file() or source.name.casefold() == "pick_data.csv":
            continue
        target = destination / source.name
        try:
            os.link(source, target)
            methods.add("hardlink")
        except OSError:
            shutil.copy2(source, target)
            methods.add("copy")
    if not (destination / "dist_matrix.csv").is_file():
        raise ValueError(f"{data_dir} does not contain required dist_matrix.csv")
    return "+".join(sorted(methods)) or "none"


def make_subset_instances(
    *,
    data_dir: Path,
    temp_root: Path,
    header: list[str],
    rows_by_order: dict[str, list[list[str]]],
    wave_id: str,
    n_values: Sequence[int],
    repeats: int,
    subset_seed_base: int,
    manifest_path: Path,
) -> list[SubsetInstance]:
    canonical_order_ids = list(rows_by_order)
    canonical_position = {order_id: index for index, order_id in enumerate(canonical_order_ids)}
    instances: list[SubsetInstance] = []

    manifest_fields = [
        "wave_id",
        "repeat_index",
        "subset_seed",
        "n_orders",
        "subset_id",
        "selection_rank",
        "canonical_input_rank",
        "order_id",
        "pick_row_count",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle:
        manifest = csv.DictWriter(manifest_handle, fieldnames=manifest_fields)
        manifest.writeheader()

        for repeat_index in range(repeats):
            subset_seed = subset_seed_base + repeat_index
            permutation = canonical_order_ids.copy()
            random.Random(subset_seed).shuffle(permutation)
            selection_rank = {order_id: rank for rank, order_id in enumerate(permutation, start=1)}

            for n_orders in n_values:
                selected_ids = permutation[:n_orders]
                selected_set = set(selected_ids)
                subset_id = sha256_text(
                    f"{wave_id}\n{subset_seed}\n{n_orders}\n" + "\n".join(selected_ids)
                )[:16]
                instance_dir = temp_root / f"repeat_{repeat_index:03d}" / f"n_{n_orders:06d}"
                instance_dir.mkdir(parents=True, exist_ok=False)
                materialization = materialize_static_files(data_dir, instance_dir)

                with (instance_dir / "pick_data.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as pick_handle:
                    writer = csv.writer(pick_handle, lineterminator="\n")
                    writer.writerow(header)
                    # Preserve canonical input order. Randomness determines membership only.
                    for order_id in canonical_order_ids:
                        if order_id in selected_set:
                            writer.writerows(rows_by_order[order_id])

                for order_id in selected_ids:
                    manifest.writerow(
                        {
                            "wave_id": wave_id,
                            "repeat_index": repeat_index,
                            "subset_seed": subset_seed,
                            "n_orders": n_orders,
                            "subset_id": subset_id,
                            "selection_rank": selection_rank[order_id],
                            "canonical_input_rank": canonical_position[order_id] + 1,
                            "order_id": order_id,
                            "pick_row_count": len(rows_by_order[order_id]),
                        }
                    )

                instances.append(
                    SubsetInstance(
                        repeat_index=repeat_index,
                        subset_seed=subset_seed,
                        n_orders=n_orders,
                        subset_id=subset_id,
                        data_dir=instance_dir,
                        static_materialization=materialization,
                    )
                )
    return instances


def normalize_captured_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_cli_result(solution_path: Path, stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "objective": None,
        "runtime_ms": None,
        "feasible": None,
        "observed_solver": None,
        "observed_config_id": None,
        "reported_order_count": None,
        "reported_wave_ids": None,
        "parse_source": None,
        "parse_error": "",
    }
    if solution_path.is_file():
        try:
            with solution_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            parsed.update(
                {
                    "objective": float(payload["objective"]),
                    "runtime_ms": float(payload["runtime_ms"]),
                    "feasible": bool(payload["feasible"]),
                    "observed_solver": str(payload.get("solver", "")),
                    "observed_config_id": str(payload.get("config_id", "")),
                    "parse_source": "solution_json",
                }
            )
            waves = payload.get("waves", [])
            parsed["reported_wave_ids"] = ";".join(
                str(wave.get("wave_id", "")) for wave in waves
            )
            parsed["reported_order_count"] = sum(
                len(batch.get("orders", []))
                for wave in waves
                for batch in wave.get("batches", [])
            )
            return parsed
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parsed["parse_error"] = f"solution JSON could not be parsed: {exc}"

    objective_match = re.search(rf"^Objective:\s*({NUMBER_PATTERN})\s*$", stdout, re.MULTILINE)
    runtime_match = re.search(rf"^Runtime ms:\s*({NUMBER_PATTERN})\s*$", stdout, re.MULTILINE)
    feasible_match = re.search(r"^Feasible:\s*(yes|no)\s*$", stdout, re.MULTILINE | re.IGNORECASE)
    solver_match = re.search(r"^Solver:\s*(\S.*?)\s*$", stdout, re.MULTILINE)
    config_match = re.search(r"^Config:\s*(\S.*?)\s*$", stdout, re.MULTILINE)
    if objective_match and runtime_match and feasible_match:
        parsed.update(
            {
                "objective": float(objective_match.group(1)),
                "runtime_ms": float(runtime_match.group(1)),
                "feasible": feasible_match.group(1).casefold() == "yes",
                "observed_solver": solver_match.group(1) if solver_match else None,
                "observed_config_id": config_match.group(1) if config_match else None,
                "parse_source": "stdout",
            }
        )
        return parsed

    if not parsed["parse_error"]:
        parsed["parse_error"] = "objective, runtime, or feasibility was absent from CLI output"
    return parsed


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def descriptive_stats(values: Sequence[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_stdev": None,
            f"{prefix}_q1": None,
            f"{prefix}_q3": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
        f"{prefix}_q1": percentile(values, 0.25),
        f"{prefix}_q3": percentile(values, 0.75),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def summarize_runs(
    records: Sequence[dict[str, Any]], experiments: Sequence[Experiment], n_values: Sequence[int]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        for n_orders in n_values:
            selected = [
                record
                for record in records
                if record["experiment_label"] == experiment.label
                and record["n_orders"] == n_orders
            ]
            successful = [record for record in selected if record["status"] == "success"]
            internal = [float(record["solver_runtime_ms"]) for record in successful]
            wall = [float(record["wall_process_ms"]) for record in successful]
            objective = [float(record["objective"]) for record in successful]
            observed_ids = sorted(
                {
                    str(record["observed_config_id"])
                    for record in successful
                    if record["observed_config_id"]
                }
            )
            summary: dict[str, Any] = {
                "experiment_label": experiment.label,
                "solver": experiment.solver,
                "declared_config_id": experiment.declared_config_id,
                "observed_config_ids": ";".join(observed_ids),
                "n_orders": n_orders,
                "expected_runs": len(selected),
                "successful_runs": len(successful),
                "feasible_runs": sum(record["feasible"] is True for record in selected),
                "timed_out_runs": sum(record["status"] == "timeout" for record in selected),
                "failed_runs": sum(record["status"] != "success" for record in selected),
                "success_rate": len(successful) / len(selected) if selected else 0.0,
                "feasible_rate": (
                    sum(record["feasible"] is True for record in selected) / len(selected)
                    if selected
                    else 0.0
                ),
            }
            summary.update(descriptive_stats(internal, "solver_runtime_ms"))
            summary.update(descriptive_stats(wall, "wall_process_ms"))
            summary.update(descriptive_stats(objective, "objective"))
            summaries.append(summary)
    return summaries


def fit_power_law(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    usable = [(n, runtime) for n, runtime in points if n > 0.0 and runtime > 0.0]
    result: dict[str, Any] = {
        "coefficient_a_ms": None,
        "empirical_exponent_b": None,
        "r_squared_log_space": None,
        "slope_standard_error": None,
        "slope_approx_95pct_low": None,
        "slope_approx_95pct_high": None,
        "n_levels_used": len(usable),
        "n_min": min((n for n, _ in usable), default=None),
        "n_max": max((n for n, _ in usable), default=None),
    }
    if len(usable) < 2:
        return result

    x = [math.log(n) for n, _ in usable]
    y = [math.log(runtime) for _, runtime in usable]
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    sxx = sum((value - mean_x) ** 2 for value in x)
    if sxx == 0.0:
        return result
    slope = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y)) / sxx
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x_i for x_i in x]
    residual_sum = sum((y_i - fitted_i) ** 2 for y_i, fitted_i in zip(y, fitted))
    total_sum = sum((y_i - mean_y) ** 2 for y_i in y)
    r_squared = None if total_sum == 0.0 else 1.0 - residual_sum / total_sum

    result.update(
        {
            "coefficient_a_ms": math.exp(intercept),
            "empirical_exponent_b": slope,
            "r_squared_log_space": r_squared,
        }
    )
    if len(usable) > 2:
        standard_error = math.sqrt(max(0.0, residual_sum / (len(usable) - 2)) / sxx)
        result.update(
            {
                "slope_standard_error": standard_error,
                "slope_approx_95pct_low": slope - 1.96 * standard_error,
                "slope_approx_95pct_high": slope + 1.96 * standard_error,
            }
        )
    return result


def make_fits(
    summaries: Sequence[dict[str, Any]], experiments: Sequence[Experiment]
) -> list[dict[str, Any]]:
    fits: list[dict[str, Any]] = []
    for experiment in experiments:
        points = [
            (float(summary["n_orders"]), float(summary["solver_runtime_ms_median"]))
            for summary in summaries
            if summary["experiment_label"] == experiment.label
            and summary["solver_runtime_ms_median"] is not None
            and float(summary["solver_runtime_ms_median"]) > 0.0
        ]
        fit = {
            "experiment_label": experiment.label,
            "solver": experiment.solver,
            "declared_config_id": experiment.declared_config_id,
            "runtime_metric": "median solver_runtime_ms per n",
            "fit_model": "T(n) = a * n^b",
            "fit_method": "ordinary least squares on ln(median runtime) versus ln(n)",
            "interpretation": "empirical descriptive fit; not a proof of asymptotic complexity",
        }
        fit.update(fit_power_law(points))
        fits.append(fit)
    return fits


def write_dict_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV without a schema: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_runtime(
    path: Path,
    records: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    fits: Sequence[dict[str, Any]],
    experiments: Sequence[Experiment],
    wave_id: str,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create runtime_vs_n.png") from exc

    fig, axis = plt.subplots(figsize=(9.0, 5.8))
    colors = plt.get_cmap("tab10")
    fit_by_label = {fit["experiment_label"]: fit for fit in fits}

    for experiment_index, experiment in enumerate(experiments):
        color = colors(experiment_index % 10)
        group_summaries = sorted(
            (
                summary
                for summary in summaries
                if summary["experiment_label"] == experiment.label
                and summary["solver_runtime_ms_median"] is not None
                and float(summary["solver_runtime_ms_median"]) > 0.0
            ),
            key=lambda summary: int(summary["n_orders"]),
        )
        if not group_summaries:
            continue
        x = [int(summary["n_orders"]) for summary in group_summaries]
        medians = [float(summary["solver_runtime_ms_median"]) for summary in group_summaries]
        q1 = [float(summary["solver_runtime_ms_q1"]) for summary in group_summaries]
        q3 = [float(summary["solver_runtime_ms_q3"]) for summary in group_summaries]

        raw = [
            record
            for record in records
            if record["experiment_label"] == experiment.label
            and record["status"] == "success"
            and float(record["solver_runtime_ms"]) > 0.0
        ]
        jitter_rng = random.Random(97_531 + experiment_index)
        raw_x = [
            float(record["n_orders"]) * math.exp(jitter_rng.uniform(-0.018, 0.018))
            for record in raw
        ]
        raw_y = [float(record["solver_runtime_ms"]) for record in raw]
        axis.scatter(raw_x, raw_y, s=15, alpha=0.22, color=color, edgecolors="none")
        axis.fill_between(x, q1, q3, alpha=0.16, color=color)

        fit = fit_by_label[experiment.label]
        exponent = fit["empirical_exponent_b"]
        r_squared = fit["r_squared_log_space"]
        suffix = ""
        if exponent is not None:
            suffix = f" (empirisch b={float(exponent):.2f}"
            suffix += f", R²={float(r_squared):.3f})" if r_squared is not None else ")"
        axis.plot(x, medians, marker="o", linewidth=2.0, color=color, label=experiment.label + suffix)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(f"Aantal orders n (geneste willekeurige subsets van {wave_id})")
    axis.set_ylabel("Uitvoeringstijd solver (ms, mediaan; band = IQR)")
    axis.set_title("Gecontroleerde empirische runtime-schaalstudie")
    axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return matplotlib.__version__


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    default_cli = repo_root / "build_release" / (
        "optioryx_cli.exe" if os.name == "nt" else "optioryx_cli"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Measure empirical runtime scaling on reproducible order-count subsets "
            "from one fixed competition wave."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cli", default=str(default_cli), help="path to the Release CLI executable")
    parser.add_argument("--data-dir", default=str(repo_root / "data"), help="competition data directory")
    parser.add_argument("--wave", required=True, help="single WAVE_ID from which every subset is sampled")
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=3,
        required=True,
        metavar=("LABEL", "SOLVER", "CONFIG"),
        help=(
            "experiment triple; repeat the option for multiple solver/config pairs. "
            "Use '-' as CONFIG to run without a config file"
        ),
    )
    parser.add_argument(
        "--n-list",
        required=True,
        help="comma-separated positive order counts, for example 50,100,200,400",
    )
    parser.add_argument("--repeats", type=int, default=5, help="independent random subsets per n")
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="comma-separated non-negative solver seeds run on every subset",
    )
    parser.add_argument("--subset-seed", type=int, default=20260803, help="base seed for subset sampling")
    parser.add_argument("--schedule-seed", type=int, default=20260804, help="seed for randomized run order")
    parser.add_argument(
        "--timeout-seconds", type=float, default=900.0, help="timeout for each CLI invocation"
    )
    parser.add_argument("--out-dir", required=True, help="directory for CSV, JSON and PNG outputs")
    parser.add_argument(
        "--require-release",
        action="store_true",
        help="abort unless an adjacent CMake cache identifies the CLI as Release",
    )
    parser.add_argument(
        "--keep-temporary-data",
        action="store_true",
        help="retain the generated filtered competition directories below out-dir",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace this script's managed output files")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first unsuccessful run")
    parser.add_argument("--quiet", action="store_true", help="suppress per-run progress messages")
    return parser


def validate_experiments(
    raw_experiments: Sequence[Sequence[str]], repo_root: Path
) -> list[Experiment]:
    experiments: list[Experiment] = []
    labels: set[str] = set()
    for label, solver, raw_config in raw_experiments:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(
                f"experiment label {label!r} is invalid; use letters, digits, '.', '_' or '-'"
            )
        if label in labels:
            raise ValueError(f"duplicate experiment label: {label}")
        labels.add(label)
        config_path = None if raw_config == "-" else Path(raw_config).expanduser()
        if config_path is not None:
            if not config_path.is_absolute():
                config_path = repo_root / config_path
            config_path = config_path.resolve()
            if not config_path.is_file():
                raise ValueError(f"config does not exist: {config_path}")
        experiments.append(
            Experiment(
                label=label,
                solver=solver,
                config_path=config_path,
                declared_config_id=read_config_id(config_path, label),
                config_sha256=sha256_file(config_path) if config_path is not None else "",
            )
        )
    return experiments


def run_study(args: argparse.Namespace, repo_root: Path) -> int:
    cli_path = Path(args.cli).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not cli_path.is_file():
        raise ValueError(f"CLI executable does not exist: {cli_path}")
    if not data_dir.is_dir():
        raise ValueError(f"data directory does not exist: {data_dir}")
    pick_path = data_dir / "pick_data.csv"
    distance_path = data_dir / "dist_matrix.csv"
    if not pick_path.is_file() or not distance_path.is_file():
        raise ValueError("data-dir must contain pick_data.csv and dist_matrix.csv")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.subset_seed < 0 or args.schedule_seed < 0:
        raise ValueError("subset and schedule seeds must be non-negative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be a finite positive number")

    n_values = sorted(parse_integer_list(args.n_list, minimum=1, option_name="--n-list"))
    solver_seeds = parse_integer_list(args.seeds, minimum=0, option_name="--seeds")
    experiments = validate_experiments(args.experiment, repo_root)
    build_type, cmake_cache = detect_build_type(cli_path)
    if args.require_release and (build_type or "").casefold() != "release":
        raise ValueError(
            "--require-release was specified, but an adjacent CMakeCache.txt did not "
            f"identify this CLI as Release (detected: {build_type!r})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    collisions = [out_dir / name for name in MANAGED_OUTPUTS if (out_dir / name).exists()]
    if collisions and not args.overwrite:
        joined = ", ".join(path.name for path in collisions)
        raise ValueError(f"output files already exist ({joined}); pass --overwrite to replace them")

    header, rows_by_order, wave_counts = load_pick_data(pick_path, args.wave)
    total_wave_orders = len(rows_by_order)
    if n_values[-1] > total_wave_orders:
        raise ValueError(
            f"largest requested n ({n_values[-1]}) exceeds the {total_wave_orders} orders "
            f"in wave {args.wave}"
        )

    started_utc = utc_now()
    protocol_path = out_dir / "protocol.json"
    static_files = [
        path for path in sorted(data_dir.iterdir()) if path.is_file() and path.name != "pick_data.csv"
    ]
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "study": "controlled empirical runtime versus order count",
        "state": "running",
        "started_utc": started_utc,
        "finished_utc": None,
        "claim_scope": (
            "Power-law exponents are descriptive empirical fits over the measured n range; "
            "they are not proofs of asymptotic time complexity."
        ),
        "protocol": {
            "fixed_wave": args.wave,
            "available_orders_in_wave": total_wave_orders,
            "n_values": n_values,
            "repeats": args.repeats,
            "solver_seeds": solver_seeds,
            "subset_seed_base": args.subset_seed,
            "subset_seed_rule": "subset_seed = subset_seed_base + zero_based_repeat_index",
            "subset_sampling": (
                "For each repeat, Python random.Random(subset_seed).shuffle creates one "
                "permutation of all order IDs in the fixed wave. Each n is its prefix, so "
                "subsets are nested within a repeat. Rows are written in original canonical "
                "CSV order so sampling does not randomize solver input order."
            ),
            "pairing": "every solver/config and solver seed receives every identical subset",
            "run_process": "one fresh CLI process per measured run; no in-process warm-up",
            "schedule": "all run jobs are shuffled once with Python random.Random(schedule_seed)",
            "schedule_seed": args.schedule_seed,
            "timeout_seconds_per_process": args.timeout_seconds,
            "primary_runtime_metric": (
                "solver_runtime_ms parsed from the CLI solution JSON; excludes Python orchestration "
                "but includes the C++ solve_instance interval"
            ),
            "secondary_runtime_metric": "wall_process_ms measured around subprocess.run",
            "summary_dispersion": "median and linearly interpolated 25th/75th percentiles (IQR)",
            "fit": "OLS of ln(median solver_runtime_ms) on ln(n): T(n)=a*n^b",
            "execution_order_caveat": (
                "Thermal state, dynamic CPU frequency, background load, and OS scheduling are not controlled."
            ),
        },
        "experiments": [
            {
                "label": experiment.label,
                "solver": experiment.solver,
                "config_path": str(experiment.config_path) if experiment.config_path else None,
                "declared_config_id": experiment.declared_config_id,
                "config_sha256": experiment.config_sha256 or None,
            }
            for experiment in experiments
        ],
        "software": {
            "command_argv": sys.argv,
            "orchestrator_path": str(Path(__file__).resolve()),
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
            "working_directory": str(Path.cwd()),
            "repo_root": str(repo_root),
            "git": git_metadata(repo_root),
            "cli_path": str(cli_path),
            "cli_sha256": sha256_file(cli_path),
            "cmake_build_type_detected": build_type,
            "cmake_cache_path": cmake_cache,
            "release_was_required": bool(args.require_release),
        },
        "hardware_and_os": machine_metadata(),
        "data": {
            "data_directory": str(data_dir),
            "pick_data_sha256": sha256_file(pick_path),
            "dist_matrix_sha256": sha256_file(distance_path),
            "static_file_sha256": {path.name: sha256_file(path) for path in static_files},
            "wave_order_counts": wave_counts,
        },
        "outputs": {name: str(out_dir / name) for name in MANAGED_OUTPUTS},
        "completed_runs": 0,
        "successful_runs": 0,
        "unsuccessful_runs": 0,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temporary_data:
        temp_root = out_dir / "temporary_instances" / (
            dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        )
        temp_root.mkdir(parents=True, exist_ok=False)
    else:
        temporary_context = tempfile.TemporaryDirectory(prefix="optioryx_scaling_")
        temp_root = Path(temporary_context.name)

    records: list[dict[str, Any]] = []
    try:
        instances = make_subset_instances(
            data_dir=data_dir,
            temp_root=temp_root,
            header=header,
            rows_by_order=rows_by_order,
            wave_id=args.wave,
            n_values=n_values,
            repeats=args.repeats,
            subset_seed_base=args.subset_seed,
            manifest_path=out_dir / "subset_manifest.csv",
        )
        protocol["temporary_data"] = {
            "retained": bool(args.keep_temporary_data),
            "path": str(temp_root) if args.keep_temporary_data else None,
            "static_materialization_methods": sorted(
                {instance.static_materialization for instance in instances}
            ),
        }
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

        jobs = [
            (instance, experiment, solver_seed)
            for instance in instances
            for experiment in experiments
            for solver_seed in solver_seeds
        ]
        random.Random(args.schedule_seed).shuffle(jobs)

        raw_fields = [
            "schedule_index",
            "run_started_utc",
            "wave_id",
            "total_orders_in_wave",
            "n_orders",
            "repeat_index",
            "subset_seed",
            "subset_id",
            "experiment_label",
            "solver",
            "declared_config_id",
            "observed_config_id",
            "config_path",
            "config_sha256",
            "solver_seed",
            "status",
            "return_code",
            "timed_out",
            "feasible",
            "objective",
            "solver_runtime_ms",
            "wall_process_ms",
            "runtime_parse_source",
            "reported_order_count",
            "reported_wave_ids",
            "static_materialization",
            "command_json",
            "parse_error",
            "stdout",
            "stderr",
        ]
        runs_path = out_dir / "runs.csv"
        with runs_path.open("w", encoding="utf-8", newline="") as runs_handle:
            writer = csv.DictWriter(runs_handle, fieldnames=raw_fields)
            writer.writeheader()

            for schedule_index, (instance, experiment, solver_seed) in enumerate(jobs, start=1):
                solution_path = temp_root / "solutions" / f"run_{schedule_index:06d}.json"
                solution_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(cli_path),
                    "solve",
                    "--solver",
                    experiment.solver,
                    "--input",
                    str(instance.data_dir),
                    "--output",
                    str(solution_path),
                    "--seed",
                    str(solver_seed),
                ]
                if experiment.config_path is not None:
                    command.extend(["--config", str(experiment.config_path)])

                run_started_utc = utc_now()
                wall_started = time.perf_counter()
                timed_out = False
                return_code: int | None = None
                stdout = ""
                stderr = ""
                try:
                    completed = subprocess.run(
                        command,
                        cwd=repo_root,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=args.timeout_seconds,
                    )
                    return_code = completed.returncode
                    stdout = completed.stdout
                    stderr = completed.stderr
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout = normalize_captured_text(exc.stdout)
                    stderr = normalize_captured_text(exc.stderr)
                except OSError as exc:
                    stderr = str(exc)
                wall_process_ms = (time.perf_counter() - wall_started) * 1000.0
                parsed = parse_cli_result(solution_path, stdout)

                status = "success"
                if timed_out:
                    status = "timeout"
                elif return_code is None:
                    status = "launch_error"
                elif parsed["parse_source"] is None:
                    status = "parse_error"
                elif parsed["reported_order_count"] is not None and parsed["reported_order_count"] != instance.n_orders:
                    status = "output_mismatch"
                    parsed["parse_error"] = (
                        f"solution contains {parsed['reported_order_count']} orders; "
                        f"expected {instance.n_orders}"
                    )
                elif parsed["reported_wave_ids"] not in (None, "", args.wave):
                    status = "output_mismatch"
                    parsed["parse_error"] = (
                        f"solution reports wave(s) {parsed['reported_wave_ids']!r}; expected {args.wave!r}"
                    )
                elif parsed["feasible"] is not True:
                    status = "infeasible"
                elif return_code != 0:
                    status = "process_error"
                elif not (
                    math.isfinite(float(parsed["objective"]))
                    and math.isfinite(float(parsed["runtime_ms"]))
                    and float(parsed["runtime_ms"]) >= 0.0
                ):
                    status = "invalid_measurement"

                record = {
                    "schedule_index": schedule_index,
                    "run_started_utc": run_started_utc,
                    "wave_id": args.wave,
                    "total_orders_in_wave": total_wave_orders,
                    "n_orders": instance.n_orders,
                    "repeat_index": instance.repeat_index,
                    "subset_seed": instance.subset_seed,
                    "subset_id": instance.subset_id,
                    "experiment_label": experiment.label,
                    "solver": experiment.solver,
                    "declared_config_id": experiment.declared_config_id,
                    "observed_config_id": parsed["observed_config_id"],
                    "config_path": str(experiment.config_path) if experiment.config_path else "",
                    "config_sha256": experiment.config_sha256,
                    "solver_seed": solver_seed,
                    "status": status,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "feasible": parsed["feasible"],
                    "objective": parsed["objective"],
                    "solver_runtime_ms": parsed["runtime_ms"],
                    "wall_process_ms": wall_process_ms,
                    "runtime_parse_source": parsed["parse_source"],
                    "reported_order_count": parsed["reported_order_count"],
                    "reported_wave_ids": parsed["reported_wave_ids"],
                    "static_materialization": instance.static_materialization,
                    "command_json": json.dumps(command),
                    "parse_error": parsed["parse_error"],
                    "stdout": stdout,
                    "stderr": stderr,
                }
                records.append(record)
                writer.writerow(record)
                runs_handle.flush()
                if not args.quiet:
                    runtime_display = (
                        f"{float(parsed['runtime_ms']):.3f} ms"
                        if parsed["runtime_ms"] is not None
                        else "no runtime"
                    )
                    print(
                        f"[{schedule_index}/{len(jobs)}] {experiment.label} "
                        f"n={instance.n_orders} repeat={instance.repeat_index} seed={solver_seed}: "
                        f"{status} ({runtime_display})",
                        flush=True,
                    )
                if status != "success" and args.fail_fast:
                    raise RuntimeError(f"run {schedule_index} failed with status {status}")

        summaries = summarize_runs(records, experiments, n_values)
        fits = make_fits(summaries, experiments)
        write_dict_csv(out_dir / "summary.csv", summaries)
        write_dict_csv(out_dir / "fits.csv", fits)
        matplotlib_version = plot_runtime(
            out_dir / "runtime_vs_n.png",
            records,
            summaries,
            fits,
            experiments,
            args.wave,
        )
        unsuccessful = sum(record["status"] != "success" for record in records)
        protocol.update(
            {
                "state": "complete" if unsuccessful == 0 else "complete_with_failures",
                "finished_utc": utc_now(),
                "completed_runs": len(records),
                "successful_runs": len(records) - unsuccessful,
                "unsuccessful_runs": unsuccessful,
                "matplotlib_version": matplotlib_version,
            }
        )
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        print(f"Raw runs: {out_dir / 'runs.csv'}")
        print(f"Summary:  {out_dir / 'summary.csv'}")
        print(f"Fits:     {out_dir / 'fits.csv'}")
        print(f"Figure:   {out_dir / 'runtime_vs_n.png'}")
        return 0 if unsuccessful == 0 else 2
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args = parser.parse_args()
    try:
        return run_study(args, repo_root)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
