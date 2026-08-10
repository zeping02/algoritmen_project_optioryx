"""Run a locked, randomized timing and K-sensitivity study for seed-selection.

Every configuration is deterministic. Repetition therefore estimates timing
variation only; solution quality must be identical across repetitions. The
schedule randomizes configuration/repetition order to spread temporal machine
effects across all variants.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Experiment:
    label: str
    filename: str
    seed: str
    rule: str
    k: int | None = None


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("largest_overlap", "largest_overlap.cfg", "largest", "overlap"),
    Experiment("largest_span_expansion", "largest_span_expansion.cfg", "largest", "span"),
    Experiment("smallest_overlap", "smallest_overlap.cfg", "smallest", "overlap"),
    Experiment("smallest_span_expansion", "smallest_span_expansion.cfg", "smallest", "span"),
    Experiment("largest_immediate", "largest_immediate.cfg", "largest", "immediate"),
    Experiment("smallest_immediate", "smallest_immediate.cfg", "smallest", "immediate"),
    Experiment("largest_k2", "largest_k2.cfg", "largest", "two_step", 2),
    Experiment("largest_k4", "largest_k4.cfg", "largest", "two_step", 4),
    Experiment("largest_k8", "largest_k8.cfg", "largest", "two_step", 8),
    Experiment("largest_k16", "largest_k16.cfg", "largest", "two_step", 16),
    Experiment("smallest_k2", "smallest_k2.cfg", "smallest", "two_step", 2),
    Experiment("smallest_k4", "smallest_k4.cfg", "smallest", "two_step", 4),
    Experiment("smallest_k8", "smallest_k8.cfg", "smallest", "two_step", 8),
    Experiment("smallest_k16", "smallest_k16.cfg", "smallest", "two_step", 16),
)


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            key, separator, value = line.partition("=")
            if not separator or not key.strip() or not value.strip():
                raise ValueError(f"invalid config line {line_number} in {path}")
            key = key.strip()
            if key in values:
                raise ValueError(f"duplicate config key {key!r} in {path}")
            values[key] = value.strip()
    return values


def detect_release(cli: Path) -> tuple[str | None, Path | None]:
    for cache in (cli.parent / "CMakeCache.txt", cli.parent.parent / "CMakeCache.txt"):
        if not cache.is_file():
            continue
        with cache.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("CMAKE_BUILD_TYPE:"):
                    return line.partition("=")[2].strip() or None, cache.resolve()
        return None, cache.resolve()
    return None, None


def expected_config(experiment: Experiment) -> dict[str, str]:
    seed_rule = (
        "largest_snake_span" if experiment.seed == "largest" else "smallest_snake_span"
    )
    rule_names = {
        "overlap": "maximum_location_overlap",
        "span": "minimum_span_expansion",
        "immediate": "minimum_incremental_cost",
        "two_step": "bounded_two_step_lookahead",
    }
    expected = {
        "config_id": f"retrospective_seed_{experiment.label.replace('_k', '_two_step_k')}",
        "seed_rule": seed_rule,
        "selection_rule": rule_names[experiment.rule],
    }
    if experiment.k is not None:
        expected["lookahead_width"] = str(experiment.k)
    return expected


def validate_inputs(root: Path, cli: Path, data: Path) -> list[dict[str, Any]]:
    if not cli.is_file():
        raise ValueError(f"missing CLI: {cli}")
    build_type, cache = detect_release(cli)
    if (build_type or "").casefold() != "release" or cache is None:
        raise ValueError(f"CLI is not identified as a Release build: {build_type!r}")
    for filename in ("pick_data.csv", "dist_matrix.csv"):
        if not (data / filename).is_file():
            raise ValueError(f"missing data file: {data / filename}")
    metadata: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for experiment in EXPERIMENTS:
        path = root / "configs" / "retrospective_seed_k" / experiment.filename
        observed = read_config(path)
        expected = expected_config(experiment)
        if observed != expected:
            raise ValueError(
                f"locked config mismatch for {experiment.label}: expected {expected}, observed {observed}"
            )
        if observed["config_id"] in seen_ids:
            raise ValueError(f"duplicate config id: {observed['config_id']}")
        seen_ids.add(observed["config_id"])
        metadata.append(
            {
                "label": experiment.label,
                "seed": experiment.seed,
                "rule": experiment.rule,
                "k": experiment.k,
                "config_id": observed["config_id"],
                "config_path": str(path.resolve()),
                "config_sha256": sha256_file(path),
            }
        )
    return metadata


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_runs(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "schedule_index", "started_utc", "label", "seed_rule", "selection_rule",
        "lookahead_width", "repeat_index", "config_id", "config_sha256", "status",
        "return_code", "timed_out", "feasible", "objective", "solver_runtime_ms",
        "wall_process_ms", "wave_count", "parse_error", "stdout", "stderr",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_study(args: argparse.Namespace, root: Path) -> int:
    cli = Path(args.cli).expanduser().resolve()
    data = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.repeats < 2:
        raise ValueError("at least two repetitions are required for timing variation")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    if os.path.lexists(out_dir):
        raise ValueError(f"output path already exists: {out_dir}")
    metadata = validate_inputs(root, cli, data)
    metadata_by_label = {str(item["label"]): item for item in metadata}
    experiment_by_label = {item.label: item for item in EXPERIMENTS}
    jobs = [(item.label, repeat) for item in EXPERIMENTS for repeat in range(args.repeats)]
    random.Random(args.schedule_seed).shuffle(jobs)
    build_type, cache = detect_release(cli)
    if args.dry_run:
        print(
            f"Preflight passed: {len(EXPERIMENTS)} configurations x {args.repeats} "
            f"repetitions = {len(jobs)} full-instance runs."
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=False)
    temp_dir = out_dir / "temporary_solutions"
    representative_dir = out_dir / "representative_solutions"
    temp_dir.mkdir()
    representative_dir.mkdir()
    protocol_path = out_dir / "protocol.json"
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "study": "deterministic seed-selection timing and lookahead-width sensitivity",
        "state": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "repetitions": args.repeats,
        "schedule_seed": args.schedule_seed,
        "expected_runs": len(jobs),
        "completed_runs": 0,
        "experiments": metadata,
        "software": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "cli": str(cli),
            "cli_sha256": sha256_file(cli),
            "cmake_build_type": build_type,
            "cmake_cache": str(cache),
            "python": sys.executable,
        },
        "data": {
            "directory": str(data),
            "pick_data_sha256": sha256_file(data / "pick_data.csv"),
            "dist_matrix_sha256": sha256_file(data / "dist_matrix.csv"),
        },
    }
    atomic_json(protocol_path, protocol)
    records: list[dict[str, Any]] = []

    try:
        for schedule_index, (label, repeat_index) in enumerate(jobs, start=1):
            experiment = experiment_by_label[label]
            meta = metadata_by_label[label]
            config_path = Path(str(meta["config_path"]))
            output_path = temp_dir / f"{schedule_index:03d}_{label}_r{repeat_index:02d}.json"
            command = [
                str(cli), "solve", "--solver", "seed_selection", "--input", str(data),
                "--output", str(output_path), "--seed", "0", "--config", str(config_path),
            ]
            started_utc = utc_now()
            started = time.perf_counter()
            timed_out = False
            payload: dict[str, Any] | None = None
            parse_error = ""
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                return_code = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                return_code = -1
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
            wall_ms = 1000.0 * (time.perf_counter() - started)
            if return_code == 0 and not timed_out and output_path.is_file():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    parse_error = str(exc)
            elif not timed_out:
                parse_error = "missing solution JSON" if not output_path.is_file() else "nonzero return code"

            feasible = bool(payload and payload.get("feasible") is True)
            objective = payload.get("objective") if payload else ""
            solver_runtime = payload.get("runtime_ms") if payload else ""
            waves = payload.get("waves") if payload else None
            wave_count = len(waves) if isinstance(waves, list) else ""
            observed_id = payload.get("config_id") if payload else None
            if payload and observed_id != meta["config_id"]:
                parse_error = f"observed config_id {observed_id!r} differs"
            if payload and wave_count != 12:
                parse_error = f"expected 12 waves, observed {wave_count!r}"
            status = (
                "success"
                if return_code == 0 and not timed_out and feasible and not parse_error
                else "failed"
            )
            record = {
                "schedule_index": schedule_index,
                "started_utc": started_utc,
                "label": label,
                "seed_rule": experiment.seed,
                "selection_rule": experiment.rule,
                "lookahead_width": experiment.k if experiment.k is not None else "",
                "repeat_index": repeat_index,
                "config_id": meta["config_id"],
                "config_sha256": meta["config_sha256"],
                "status": status,
                "return_code": return_code,
                "timed_out": timed_out,
                "feasible": feasible,
                "objective": objective,
                "solver_runtime_ms": solver_runtime,
                "wall_process_ms": wall_ms,
                "wave_count": wave_count,
                "parse_error": parse_error,
                "stdout": stdout,
                "stderr": stderr,
            }
            records.append(record)
            write_runs(out_dir / "runs.csv", records)
            protocol["completed_runs"] = len(records)
            protocol["last_schedule_index"] = schedule_index
            atomic_json(protocol_path, protocol)
            if status != "success":
                raise RuntimeError(f"run {schedule_index}/{len(jobs)} failed: {record}")
            if repeat_index == 0:
                shutil.copy2(output_path, representative_dir / f"{label}.json")
            output_path.unlink(missing_ok=True)
            if not args.quiet:
                print(
                    f"[{schedule_index:03d}/{len(jobs)}] {label} r{repeat_index}: "
                    f"D={float(objective):.2f}, internal={float(solver_runtime)/1000.0:.3f}s, "
                    f"wall={wall_ms/1000.0:.3f}s",
                    flush=True,
                )
    except BaseException as exc:
        protocol["state"] = "failed"
        protocol["finished_utc"] = utc_now()
        protocol["error"] = str(exc)
        atomic_json(protocol_path, protocol)
        raise

    temp_dir.rmdir()
    protocol["state"] = "complete"
    protocol["finished_utc"] = utc_now()
    protocol["successful_runs"] = len(records)
    atomic_json(protocol_path, protocol)
    print(f"Completed {len(records)} deterministic full-instance runs in {out_dir}")
    return 0


def parser(root: Path) -> argparse.ArgumentParser:
    executable = "optioryx_cli.exe" if os.name == "nt" else "optioryx_cli"
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument(
        "--cli", default=str(root / "build_research_release" / executable)
    )
    result.add_argument("--data-dir", default=str(root / "data"))
    result.add_argument("--out-dir", required=True)
    result.add_argument("--repeats", type=int, default=10)
    result.add_argument("--schedule-seed", type=int, default=2026080701)
    result.add_argument("--timeout-seconds", type=float, default=900.0)
    result.add_argument("--quiet", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = root_dir()
    args = parser(root).parse_args(argv)
    try:
        return run_study(args, root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
