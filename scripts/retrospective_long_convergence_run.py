"""Run a locked, lossless long-horizon convergence study on the Release CLI.

Each CLI invocation solves the complete twelve-wave instance once and writes its
normal JSON output to a temporary directory.  The script streams that JSON into
compact, exact study artefacts:

* one dense full-instance best-so-far trajectory per configuration and seed;
* one lossless change-point trajectory for every underlying wave;
* exact internal solver time and process wall time;
* per-run and per-wave convergence metrics.

The requested result root must not exist.  Existing result directories are never
overwritten.  Completed job artefacts are also opened exclusively.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from retrospective_long_convergence_common import (
    DEFAULT_SEEDS,
    EXPERIMENTS,
    WAVES,
    ExperimentSpec,
    config_path,
    count_orders_by_wave,
    parse_seed_list,
    repo_root,
    sha256_file,
    validate_configuration_files,
)


IMPROVEMENT_TOLERANCE = 1e-9
SCHEMA_VERSION = 1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def exclusive_write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def detect_cmake_build_type(cli_path: Path) -> tuple[str | None, Path | None]:
    candidates = (cli_path.parent / "CMakeCache.txt", cli_path.parent.parent / "CMakeCache.txt")
    for cache_path in candidates:
        if not cache_path.is_file():
            continue
        with cache_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("CMAKE_BUILD_TYPE:"):
                    return line.partition("=")[2].strip() or None, cache_path.resolve()
        return None, cache_path.resolve()
    return None, None


def git_text(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def parse_cli_stdout(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in stdout.splitlines():
        key, separator, value = raw_line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def first_threshold_step(best: Sequence[float], fraction: float) -> int | None:
    gain = best[0] - best[-1]
    if gain <= IMPROVEMENT_TOLERANCE:
        return None
    target = fraction * gain
    for step, objective in enumerate(best):
        if best[0] - objective + IMPROVEMENT_TOLERANCE >= target:
            return step
    raise AssertionError("final trajectory point does not reach its own gain threshold")


def trajectory_metrics(best: Sequence[float]) -> dict[str, Any]:
    if not best:
        raise ValueError("empty convergence trajectory")
    horizon = len(best) - 1
    for step in range(1, len(best)):
        if best[step] > best[step - 1] + IMPROVEMENT_TOLERANCE:
            raise ValueError(f"best-so-far trajectory increases at step {step}")
    improvement_steps = [
        step
        for step in range(1, len(best))
        if best[step - 1] - best[step] > IMPROVEMENT_TOLERANCE
    ]
    start = best[0]
    final = best[-1]
    gain = start - final
    last_improvement = improvement_steps[-1] if improvement_steps else None
    last_ten_start = int(math.floor(0.9 * horizon))
    gain_last_ten = best[last_ten_start] - final
    return {
        "horizon": horizon,
        "start_objective": start,
        "final_objective": final,
        "total_gain": gain,
        "no_improvement": not improvement_steps,
        "improvement_count": len(improvement_steps),
        "t95": first_threshold_step(best, 0.95),
        "t99": first_threshold_step(best, 0.99),
        "last_improvement": last_improvement,
        "last_improvement_fraction": (
            None if last_improvement is None or horizon == 0 else last_improvement / horizon
        ),
        "edge_step_gain": 0.0 if horizon == 0 else best[-2] - best[-1],
        "improved_at_edge": bool(improvement_steps and improvement_steps[-1] == horizon),
        "last_ten_percent_start_step": last_ten_start,
        "gain_last_ten_percent": gain_last_ten,
        "share_gain_last_ten_percent": 0.0 if gain <= IMPROVEMENT_TOLERANCE else gain_last_ten / gain,
        "improved_in_last_ten_percent": gain_last_ten > IMPROVEMENT_TOLERANCE,
    }


def parse_scalar(line: str) -> Any:
    _key, separator, value = line.strip().partition(":")
    if not separator:
        raise ValueError(f"expected JSON scalar line, got {line!r}")
    return json.loads(value.rstrip(","))


def parse_solution_json(
    path: Path, spec: ExperimentSpec, seed: int
) -> tuple[dict[str, Any], list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    top: dict[str, Any] = {}
    expected_wave_ids = [wave_id for wave_id, _count in WAVES]
    full_best = [0.0] * (spec.budget + 1)
    wave_metrics: list[dict[str, Any]] = []
    change_points: list[dict[str, Any]] = []

    current_wave: str | None = None
    current_objective: float | None = None
    current_runtime_ms: float | None = None
    current_best: list[float] = []
    in_convergence = False

    def finish_wave() -> None:
        nonlocal current_wave, current_objective, current_runtime_ms, current_best
        if current_wave is None or current_objective is None or current_runtime_ms is None:
            raise ValueError("incomplete wave metadata in solution JSON")
        if len(current_best) != spec.budget + 1:
            raise ValueError(
                f"{current_wave}: expected {spec.budget + 1} points, got {len(current_best)}"
            )
        if not math.isclose(current_best[-1], current_objective, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"{current_wave}: final best {current_best[-1]} differs from wave objective "
                f"{current_objective}"
            )
        for step, value in enumerate(current_best):
            full_best[step] += value
        metrics = trajectory_metrics(current_best)
        metrics.update(
            {
                "wave_id": current_wave,
                "solver_runtime_ms": current_runtime_ms,
            }
        )
        wave_metrics.append(metrics)
        previous: float | None = None
        for step, value in enumerate(current_best):
            if previous is None or previous - value > IMPROVEMENT_TOLERANCE:
                change_points.append(
                    {"wave_id": current_wave, "step": step, "best_objective": value}
                )
            previous = value
        current_wave = None
        current_objective = None
        current_runtime_ms = None
        current_best = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith('  "solver":'):
                top["solver"] = parse_scalar(line)
            elif line.startswith('  "config_id":'):
                top["config_id"] = parse_scalar(line)
            elif line.startswith('  "seed":'):
                top["seed"] = parse_scalar(line)
            elif line.startswith('  "objective":'):
                top["objective"] = float(parse_scalar(line))
            elif line.startswith('  "runtime_ms":'):
                top["runtime_ms"] = float(parse_scalar(line))
            elif line.startswith('  "feasible":'):
                top["feasible"] = bool(parse_scalar(line))
            elif line.startswith('      "wave_id":'):
                if in_convergence:
                    raise ValueError(f"new wave encountered inside convergence at line {line_number}")
                current_wave = str(parse_scalar(line))
            elif line.startswith('      "objective":'):
                current_objective = float(parse_scalar(line))
            elif line.startswith('      "runtime_ms":'):
                current_runtime_ms = float(parse_scalar(line))
            elif line.startswith('      "convergence": ['):
                if current_wave is None:
                    raise ValueError(f"convergence without wave at line {line_number}")
                in_convergence = True
                current_best = []
            elif in_convergence and line.startswith("        {"):
                point = json.loads(line.strip().rstrip(","))
                step = int(point["step"])
                if step != len(current_best):
                    raise ValueError(
                        f"{current_wave}: expected step {len(current_best)}, got {step}"
                    )
                current_best.append(float(point["best_objective"]))
            elif in_convergence and line.startswith("      ]"):
                in_convergence = False
                finish_wave()

    required_top = {"solver", "config_id", "seed", "objective", "runtime_ms", "feasible"}
    if set(top) != required_top:
        raise ValueError(f"solution JSON top-level metadata differs: {sorted(top)}")
    if top["solver"] != spec.solver:
        raise ValueError(f"observed solver {top['solver']!r}, expected {spec.solver!r}")
    if int(top["seed"]) != seed:
        raise ValueError(f"observed seed {top['seed']!r}, expected {seed}")
    if not top["feasible"]:
        raise ValueError("solver reported an infeasible full-instance solution")
    if [row["wave_id"] for row in wave_metrics] != expected_wave_ids:
        raise ValueError("solution JSON does not contain the twelve waves in canonical order")
    if not math.isclose(full_best[-1], float(top["objective"]), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("sum of final wave objectives differs from full-instance objective")
    wave_runtime_sum = math.fsum(float(row["solver_runtime_ms"]) for row in wave_metrics)
    if float(top["runtime_ms"]) + 1e-6 < wave_runtime_sum:
        raise ValueError("full-instance solver time is smaller than the sum of wave timers")
    top["wave_runtime_ms_sum"] = wave_runtime_sum
    top["instance_overhead_runtime_ms"] = float(top["runtime_ms"]) - wave_runtime_sum
    return top, full_best, wave_metrics, change_points


def write_gzip_csv_exclusive(
    temporary_path: Path, final_path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_path}")
    with temporary_path.open("xb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle:
                writer = csv.DictWriter(text_handle, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
    temporary_path.rename(final_path)


def full_trajectory_rows(best: Sequence[float]) -> Iterable[dict[str, Any]]:
    start = best[0]
    gain = start - best[-1]
    for step, objective in enumerate(best):
        improvement = start - objective
        yield {
            "step": step,
            "best_objective_total": format(objective, ".17g"),
            "gain_from_start": format(improvement, ".17g"),
            "fraction_of_final_gain": (
                "" if gain <= IMPROVEMENT_TOLERANCE else format(improvement / gain, ".17g")
            ),
        }


def execute_job(
    *,
    root: Path,
    cli_path: Path,
    data_dir: Path,
    out_dir: Path,
    spec: ExperimentSpec,
    config_id: str,
    config_sha256: str,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    job_id = f"{spec.label}__seed-{seed:02d}"
    record_path = out_dir / "records" / f"{job_id}.json"
    trajectory_path = out_dir / "trajectories" / f"{job_id}.csv.gz"
    changes_path = out_dir / "wave_changepoints" / f"{job_id}.csv.gz"
    for path in (record_path, trajectory_path, changes_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite completed job artefact: {path}")

    with tempfile.TemporaryDirectory(prefix=f"optioryx_{job_id}_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        solution_path = temporary_root / "solution.json"
        command = [
            str(cli_path),
            "solve",
            "--solver",
            spec.solver,
            "--input",
            str(data_dir),
            "--output",
            str(solution_path),
            "--config",
            str(config_path(root, spec).resolve()),
            "--seed",
            str(seed),
        ]
        started_utc = utc_now()
        start_ns = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{job_id} exceeded {timeout_seconds:.1f} seconds") from exc
        wall_process_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        finished_utc = utc_now()
        if completed.returncode != 0:
            raise RuntimeError(
                f"{job_id} failed with return code {completed.returncode}: {completed.stderr.strip()}"
            )
        if not solution_path.is_file():
            raise RuntimeError(f"{job_id} did not create its temporary solution JSON")

        stdout_values = parse_cli_stdout(completed.stdout)
        if stdout_values.get("Feasible", "").casefold() != "yes":
            raise RuntimeError(f"{job_id} did not report a feasible solution")
        top, full_best, wave_metrics, change_points = parse_solution_json(solution_path, spec, seed)
        if top["config_id"] != config_id:
            raise RuntimeError(
                f"{job_id} observed config {top['config_id']!r}, expected {config_id!r}"
            )

        trajectory_temp = temporary_root / "full_trajectory.csv.gz"
        write_gzip_csv_exclusive(
            trajectory_temp,
            trajectory_path,
            ("step", "best_objective_total", "gain_from_start", "fraction_of_final_gain"),
            full_trajectory_rows(full_best),
        )
        changes_temp = temporary_root / "wave_changepoints.csv.gz"
        write_gzip_csv_exclusive(
            changes_temp,
            changes_path,
            ("wave_id", "step", "best_objective"),
            change_points,
        )

    metrics = trajectory_metrics(full_best)
    record = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "label": spec.label,
        "solver": spec.solver,
        "family": spec.family,
        "budget": spec.budget,
        "seed": seed,
        "config_id": config_id,
        "config_sha256": config_sha256,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "solver_runtime_ms": float(top["runtime_ms"]),
        "wave_solver_runtime_ms_sum": float(top["wave_runtime_ms_sum"]),
        "instance_overhead_runtime_ms": float(top["instance_overhead_runtime_ms"]),
        "wall_process_ms": wall_process_ms,
        "full_instance_metrics": metrics,
        "wave_metrics": wave_metrics,
        "cli_stdout": completed.stdout,
        "cli_stderr": completed.stderr,
        "trajectory_file": {
            "path": str(trajectory_path),
            "sha256": sha256_file(trajectory_path),
            "rows": spec.budget + 1,
            "representation": "dense exact sum of twelve per-wave best-so-far values",
        },
        "wave_changepoints_file": {
            "path": str(changes_path),
            "sha256": sha256_file(changes_path),
            "rows": len(change_points),
            "representation": "lossless start plus every strict per-wave best-so-far decrease",
        },
    }
    exclusive_write_json(record_path, record)
    return record


def record_manifest_entry(out_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": record["job_id"],
        "record_path": str(out_dir / "records" / f"{record['job_id']}.json"),
        "solver_runtime_ms": record["solver_runtime_ms"],
        "wall_process_ms": record["wall_process_ms"],
    }


def load_completed_records(out_dir: Path, expected_jobs: set[str]) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    records_dir = out_dir / "records"
    if not records_dir.is_dir():
        raise ValueError(f"missing records directory in resumable study: {records_dir}")
    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        job_id = str(record.get("job_id", ""))
        if job_id not in expected_jobs:
            raise ValueError(f"unexpected completed record during resume: {record_path}")
        if job_id in completed:
            raise ValueError(f"duplicate completed record for {job_id}")
        expected_record_path = records_dir / f"{job_id}.json"
        if record_path.resolve() != expected_record_path.resolve():
            raise ValueError(f"record filename does not match job id: {record_path}")
        for key, directory in (
            ("trajectory_file", "trajectories"),
            ("wave_changepoints_file", "wave_changepoints"),
        ):
            info = record.get(key)
            if not isinstance(info, dict):
                raise ValueError(f"{job_id}: missing {key}")
            artefact = Path(str(info.get("path", ""))).resolve()
            expected_parent = (out_dir / directory).resolve()
            if artefact.parent != expected_parent or not artefact.is_file():
                raise ValueError(f"{job_id}: missing or misplaced {key}")
            if sha256_file(artefact) != info.get("sha256"):
                raise ValueError(f"{job_id}: hash mismatch in {key}")
        completed[job_id] = record
    return completed


def preserve_incomplete_artefacts(out_dir: Path, job_id: str) -> None:
    candidates = (
        out_dir / "trajectories" / f"{job_id}.csv.gz",
        out_dir / "wave_changepoints" / f"{job_id}.csv.gz",
    )
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    partial_root = out_dir / "partials"
    partial_root.mkdir(exist_ok=True)
    suffix = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = partial_root / f"{job_id}__{suffix}"
    destination.mkdir(exist_ok=False)
    for path in existing:
        path.rename(destination / path.name)


def validate_resume_manifest(
    *,
    manifest: dict[str, Any],
    out_dir: Path,
    cli_path: Path,
    pick_path: Path,
    distance_path: Path,
    config_metadata: list[dict[str, object]],
    seeds: tuple[int, ...],
    schedule_seed: int,
    expected_order: list[str],
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume manifest schema differs")
    if manifest.get("state") not in {"running", "failed"}:
        raise ValueError(f"study state {manifest.get('state')!r} is not resumable")
    if Path(str(manifest.get("output_root", ""))).resolve() != out_dir:
        raise ValueError("resume output root differs from manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("resume manifest has no protocol block")
    if protocol.get("seeds") != list(seeds):
        raise ValueError("resume seeds differ from manifest")
    if protocol.get("schedule_seed") != schedule_seed:
        raise ValueError("resume schedule seed differs from manifest")
    if protocol.get("job_order") != expected_order:
        raise ValueError("resume job order differs from manifest")
    if manifest.get("experiments") != config_metadata:
        raise ValueError("resume configs differ from manifest")
    software = manifest.get("software")
    if not isinstance(software, dict):
        raise ValueError("resume manifest has no software block")
    if software.get("cli_sha256") != sha256_file(cli_path):
        raise ValueError("resume CLI hash differs from manifest")
    if software.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("resume runner hash differs from manifest")
    common_path = Path(__file__).with_name("retrospective_long_convergence_common.py").resolve()
    if software.get("common_sha256") != sha256_file(common_path):
        raise ValueError("resume common-specification hash differs from manifest")
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("resume manifest has no data block")
    if data.get("pick_data_sha256") != sha256_file(pick_path):
        raise ValueError("resume pick_data.csv hash differs from manifest")
    if data.get("dist_matrix_sha256") != sha256_file(distance_path):
        raise ValueError("resume dist_matrix.csv hash differs from manifest")


def build_parser(root: Path) -> argparse.ArgumentParser:
    executable = "optioryx_cli.exe" if os.name == "nt" else "optioryx_cli"
    parser = argparse.ArgumentParser(
        description=(
            "Run five locked long-convergence configurations over all twelve waves "
            "using a fresh result directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cli", default=str(root / "build_budget_release" / executable), help="Release CLI"
    )
    parser.add_argument("--data-dir", default=str(root / "data"), help="competition data")
    parser.add_argument("--out-dir", required=True, help="new, non-existing result root")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated coupled solver seeds",
    )
    parser.add_argument(
        "--schedule-seed", type=int, default=2026080701, help="fixed serial job-order seed"
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=900.0, help="timeout for each full-instance run"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only a matching failed/running study and preserve incomplete artefacts",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate without creating output")
    return parser


def run(args: argparse.Namespace, root: Path) -> int:
    cli_path = Path(args.cli).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    seeds = parse_seed_list(args.seeds)
    if not cli_path.is_file():
        raise ValueError(f"CLI does not exist: {cli_path}")
    build_type, cache_path = detect_cmake_build_type(cli_path)
    if (build_type or "").casefold() != "release" or cache_path is None:
        raise ValueError(
            f"selected CLI is not identified as Release (detected {build_type!r})"
        )
    if not data_dir.is_dir():
        raise ValueError(f"data directory does not exist: {data_dir}")
    pick_path = data_dir / "pick_data.csv"
    distance_path = data_dir / "dist_matrix.csv"
    if not pick_path.is_file() or not distance_path.is_file():
        raise ValueError("data directory must contain pick_data.csv and dist_matrix.csv")
    observed_counts = count_orders_by_wave(pick_path)
    if observed_counts != dict(WAVES):
        raise ValueError(
            f"wave/order grid differs from locked protocol: observed {observed_counts}"
        )
    config_metadata = validate_configuration_files(root)
    metadata_by_label = {str(row["label"]): row for row in config_metadata}
    jobs = [(spec, seed) for spec in EXPERIMENTS for seed in seeds]
    random.Random(args.schedule_seed).shuffle(jobs)
    solver_only_minutes = 3.824 * len(seeds)
    print(f"Preflight valid: {len(jobs)} jobs, {len(seeds)} coupled seeds, twelve waves per job.")
    print(
        f"Rough solver-only estimate: {solver_only_minutes:.1f} minutes; "
        "allow approximately 15-40% extra for lossless trajectory JSON."
    )
    print(f"Release CLI SHA-256: {sha256_file(cli_path)}")
    if args.dry_run:
        print("Dry run complete; no output directory created.")
        return 0
    manifest_path = out_dir / "study_manifest.json"
    expected_order = [f"{spec.label}__seed-{seed:02d}" for spec, seed in jobs]
    expected_jobs = set(expected_order)
    if args.resume:
        if not out_dir.is_dir() or not manifest_path.is_file():
            raise ValueError("--resume requires an existing study with study_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(
            manifest=manifest,
            out_dir=out_dir,
            cli_path=cli_path,
            pick_path=pick_path,
            distance_path=distance_path,
            config_metadata=config_metadata,
            seeds=seeds,
            schedule_seed=args.schedule_seed,
            expected_order=expected_order,
        )
        completed_records = load_completed_records(out_dir, expected_jobs)
        for job_id in expected_order:
            if job_id not in completed_records:
                preserve_incomplete_artefacts(out_dir, job_id)
        manifest["state"] = "running"
        manifest.pop("failure", None)
        manifest.pop("failure_utc", None)
        manifest.setdefault("resume_utc", []).append(utc_now())
        manifest["completed_jobs"] = len(completed_records)
        manifest["records"] = [
            record_manifest_entry(out_dir, completed_records[job_id])
            for job_id in expected_order
            if job_id in completed_records
        ]
        atomic_write_json(manifest_path, manifest)
        print(f"Resuming after {len(completed_records)} verified jobs.")
    else:
        if os.path.lexists(out_dir):
            raise ValueError(f"output directory already exists: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=False)
        for name in ("records", "trajectories", "wave_changepoints", "partials"):
            (out_dir / name).mkdir(exist_ok=False)
        completed_records: dict[str, dict[str, Any]] = {}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "created_utc": utc_now(),
            "completed_utc": None,
            "root": str(root.resolve()),
            "output_root": str(out_dir),
            "protocol": {
                "seeds": list(seeds),
                "wave_counts": dict(WAVES),
                "schedule_seed": args.schedule_seed,
                "job_order": expected_order,
                "serial_execution": True,
                "trajectory_unit": "best objective summed over all twelve waves at each step",
                "timing": (
                    "solver_runtime_ms is the internal complete-instance timer including "
                    "evaluator setup and validation; per-wave timers are recorded separately; "
                    "wall_process_ms includes input loading and temporary JSON serialization"
                ),
                "improvement_tolerance": IMPROVEMENT_TOLERANCE,
                "t95_t99": (
                    "first step reaching respectively 95% or 99% of that run's total "
                    "within-budget full-instance gain"
                ),
                "seconds_to_t95_t99": (
                    "not available: the current Release convergence records contain no "
                    "per-step timestamps; only exact iteration/generation steps and total "
                    "run/per-wave times are recorded"
                ),
                "sa_budget_relation": (
                    "SA 10k and SA 50k independently cool from 60 to 0.05 using "
                    "budget-specific geometric factors; SA 10k is not a prefix of SA 50k"
                ),
                "ga_budget_relation": (
                    "GA 40/100/200 differ only in the generation horizon, but one RNG stream "
                    "is consumed sequentially across waves. The horizons are same-seed "
                    "alternatives, not shared full-instance prefixes"
                ),
                "last_ten_percent": "steps floor(0.9 * horizon) through horizon, inclusive",
            },
            "experiments": config_metadata,
            "software": {
                "cli_path": str(cli_path),
                "cli_sha256": sha256_file(cli_path),
                "cmake_build_type": build_type,
                "cmake_cache_path": str(cache_path),
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "common_path": str(Path(__file__).with_name("retrospective_long_convergence_common.py").resolve()),
                "common_sha256": sha256_file(
                    Path(__file__).with_name("retrospective_long_convergence_common.py").resolve()
                ),
                "python": sys.version,
                "platform": platform.platform(),
                "git_head": git_text(root, "rev-parse", "HEAD"),
                "git_status_porcelain_at_start": git_text(root, "status", "--porcelain"),
            },
            "hardware": {
                "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
                "logical_cpu_count": os.cpu_count(),
            },
            "data": {
                "data_dir": str(data_dir),
                "pick_data_sha256": sha256_file(pick_path),
                "dist_matrix_sha256": sha256_file(distance_path),
            },
            "planned_jobs": len(jobs),
            "completed_jobs": 0,
            "records": [],
        }
        atomic_write_json(manifest_path, manifest)

    try:
        for job_index, (spec, seed) in enumerate(jobs, start=1):
            job_id = f"{spec.label}__seed-{seed:02d}"
            if job_id in completed_records:
                print(f"[{job_index:02d}/{len(jobs):02d}] {job_id}: verified, skipped", flush=True)
                continue
            metadata = metadata_by_label[spec.label]
            print(
                f"[{job_index:02d}/{len(jobs):02d}] {spec.label}, seed {seed} "
                f"(budget {spec.budget})",
                flush=True,
            )
            record = execute_job(
                root=root,
                cli_path=cli_path,
                data_dir=data_dir,
                out_dir=out_dir,
                spec=spec,
                config_id=str(metadata["config_id"]),
                config_sha256=str(metadata["config_sha256"]),
                seed=seed,
                timeout_seconds=args.timeout_seconds,
            )
            completed_records[job_id] = record
            manifest["completed_jobs"] = len(completed_records)
            manifest["records"].append(record_manifest_entry(out_dir, record))
            atomic_write_json(manifest_path, manifest)
            print(
                f"    D={record['full_instance_metrics']['final_objective']:.6f}; "
                f"solver={record['solver_runtime_ms'] / 1000.0:.2f}s; "
                f"wall={record['wall_process_ms'] / 1000.0:.2f}s",
                flush=True,
            )
    except BaseException as exc:
        manifest["state"] = "failed"
        manifest["failure_utc"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(manifest_path, manifest)
        raise

    manifest["state"] = "complete"
    manifest["completed_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    print(f"Study complete: {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    root = repo_root()
    parser = build_parser(root)
    args = parser.parse_args(argv)
    try:
        return run(args, root)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
