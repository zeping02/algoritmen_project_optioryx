"""Run the locked Release budget study on all twelve complete competition waves.

This wrapper delegates every measured invocation to ``run_scaling_experiments.py``.
It deliberately refuses an existing output root and never passes ``--overwrite``.
Use ``--dry-run`` for a complete preflight without creating files or running solvers.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from retrospective_budget_common import (
    EXPECTED_RUNS_PER_WAVE,
    EXPECTED_TOTAL_RUNS,
    EXPERIMENTS,
    REPEATS,
    SOLVER_SEEDS,
    WAVES,
    config_path,
    repo_root,
    sha256_file,
    validate_configuration_files,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def count_orders_by_wave(pick_path: Path) -> dict[str, int]:
    order_ids: dict[str, set[str]] = {}
    with pick_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"WAVE_ID", "ORDER_ID"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{pick_path} must contain WAVE_ID and ORDER_ID")
        for line_number, row in enumerate(reader, start=2):
            wave_id = (row.get("WAVE_ID") or "").strip()
            order_id = (row.get("ORDER_ID") or "").strip()
            if not wave_id or not order_id:
                raise ValueError(f"empty wave or order id at line {line_number} in {pick_path}")
            order_ids.setdefault(wave_id, set()).add(order_id)
    return {wave_id: len(ids) for wave_id, ids in order_ids.items()}


def validate_preflight(
    *, root: Path, cli_path: Path, data_dir: Path, out_root: Path
) -> tuple[Path, list[dict[str, str | int]], str, Path]:
    runner = (root / "scripts" / "run_scaling_experiments.py").resolve()
    if not runner.is_file():
        raise ValueError(f"missing delegated runner: {runner}")
    if not cli_path.is_file():
        raise ValueError(f"CLI executable does not exist: {cli_path}")
    build_type, cache_path = detect_cmake_build_type(cli_path)
    if (build_type or "").casefold() != "release" or cache_path is None:
        raise ValueError(
            "the selected CLI is not identified as a Release build by an adjacent "
            f"CMakeCache.txt (detected {build_type!r})"
        )
    if not data_dir.is_dir():
        raise ValueError(f"data directory does not exist: {data_dir}")
    pick_path = data_dir / "pick_data.csv"
    distance_path = data_dir / "dist_matrix.csv"
    if not pick_path.is_file() or not distance_path.is_file():
        raise ValueError("data directory must contain pick_data.csv and dist_matrix.csv")
    observed_counts = count_orders_by_wave(pick_path)
    expected_counts = dict(WAVES)
    if observed_counts != expected_counts:
        raise ValueError(
            "competition wave counts differ from the locked study specification: "
            f"expected {expected_counts}, observed {observed_counts}"
        )
    if os.path.lexists(out_root):
        raise ValueError(
            f"output root already exists: {out_root}; choose a new unique directory"
        )
    config_metadata = validate_configuration_files(root)
    return runner, config_metadata, build_type, cache_path


def make_wave_command(
    *,
    root: Path,
    runner: Path,
    cli_path: Path,
    data_dir: Path,
    out_root: Path,
    wave_id: str,
    n_orders: int,
    wave_index: int,
    subset_seed_base: int,
    schedule_seed_base: int,
    timeout_seconds: float,
    quiet: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "--cli",
        str(cli_path),
        "--data-dir",
        str(data_dir),
        "--wave",
        wave_id,
        "--n-list",
        str(n_orders),
        "--repeats",
        str(REPEATS),
        "--seeds",
        ",".join(str(seed) for seed in SOLVER_SEEDS),
        "--subset-seed",
        str(subset_seed_base + wave_index),
        "--schedule-seed",
        str(schedule_seed_base + wave_index),
        "--timeout-seconds",
        str(timeout_seconds),
        "--out-dir",
        str(out_root / wave_id),
        "--require-release",
        "--fail-fast",
    ]
    for spec in EXPERIMENTS:
        command.extend(
            [
                "--experiment",
                spec.label,
                spec.solver,
                str(config_path(root, spec).resolve()),
            ]
        )
    if quiet:
        command.append("--quiet")
    return command


def validate_completed_wave(wave_dir: Path, wave_id: str) -> None:
    protocol_path = wave_dir / "protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError(f"runner did not create {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("state") != "complete":
        raise RuntimeError(f"{wave_id} ended in state {protocol.get('state')!r}")
    if protocol.get("completed_runs") != EXPECTED_RUNS_PER_WAVE:
        raise RuntimeError(f"{wave_id} did not record {EXPECTED_RUNS_PER_WAVE} runs")
    if protocol.get("successful_runs") != EXPECTED_RUNS_PER_WAVE:
        raise RuntimeError(f"{wave_id} did not record only successful runs")
    if protocol.get("unsuccessful_runs") != 0:
        raise RuntimeError(f"{wave_id} contains unsuccessful runs")


def build_parser(root: Path) -> argparse.ArgumentParser:
    executable = "optioryx_cli.exe" if os.name == "nt" else "optioryx_cli"
    parser = argparse.ArgumentParser(
        description=(
            "Run 960 randomized, fail-fast Release measurements: eight locked "
            "configurations, ten solver seeds and all twelve complete waves."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cli",
        default=str(root / "build_budget_release" / executable),
        help="path to the dedicated Release CLI",
    )
    parser.add_argument("--data-dir", default=str(root / "data"), help="competition data")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="new, non-existing study output root; existing paths are always refused",
    )
    parser.add_argument(
        "--subset-seed-base",
        type=int,
        default=2026081600,
        help="wave ordinal is added to obtain a unique subset seed",
    )
    parser.add_argument(
        "--schedule-seed-base",
        type=int,
        default=2026080600,
        help="wave ordinal is added to obtain a unique randomized schedule seed",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=900.0, help="timeout per solver process"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress individual run messages")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate everything and print commands without creating output or running solvers",
    )
    return parser


def run(args: argparse.Namespace, root: Path) -> int:
    cli_path = Path(args.cli).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    if args.subset_seed_base < 0 or args.schedule_seed_base < 0:
        raise ValueError("seed bases must be non-negative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be a finite positive number")

    runner, config_metadata, build_type, cache_path = validate_preflight(
        root=root,
        cli_path=cli_path,
        data_dir=data_dir,
        out_root=out_root,
    )
    commands = [
        make_wave_command(
            root=root,
            runner=runner,
            cli_path=cli_path,
            data_dir=data_dir,
            out_root=out_root,
            wave_id=wave_id,
            n_orders=n_orders,
            wave_index=index,
            subset_seed_base=args.subset_seed_base,
            schedule_seed_base=args.schedule_seed_base,
            timeout_seconds=args.timeout_seconds,
            quiet=args.quiet,
        )
        for index, (wave_id, n_orders) in enumerate(WAVES, start=1)
    ]

    if args.dry_run:
        print(
            f"Preflight passed: {len(WAVES)} waves, {len(EXPERIMENTS)} configurations, "
            f"{len(SOLVER_SEEDS)} seeds, {EXPECTED_TOTAL_RUNS} expected runs."
        )
        for command in commands:
            print(subprocess.list2cmdline(command))
        return 0

    out_root.mkdir(parents=True, exist_ok=False)
    manifest_path = out_root / "study_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "retrospective full-instance time-quality budget study",
        "state": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "claim_scope": (
            "Objectives and runtimes are aggregated over all twelve complete waves. "
            "Intervals describe variation across the ten paired solver seeds."
        ),
        "expected_total_runs": EXPECTED_TOTAL_RUNS,
        "expected_runs_per_wave": EXPECTED_RUNS_PER_WAVE,
        "waves": [
            {
                "wave_id": wave_id,
                "n_orders": n_orders,
                "subset_seed": args.subset_seed_base + index,
                "schedule_seed": args.schedule_seed_base + index,
            }
            for index, (wave_id, n_orders) in enumerate(WAVES, start=1)
        ],
        "repeats": REPEATS,
        "solver_seeds": list(SOLVER_SEEDS),
        "experiments": config_metadata,
        "completed_waves": [],
        "software": {
            "wrapper_path": str(Path(__file__).resolve()),
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "common_module_path": str(Path(__file__).with_name("retrospective_budget_common.py").resolve()),
            "common_module_sha256": sha256_file(
                Path(__file__).with_name("retrospective_budget_common.py").resolve()
            ),
            "delegated_runner_path": str(runner),
            "delegated_runner_sha256": sha256_file(runner),
            "python_executable": sys.executable,
            "cli_path": str(cli_path),
            "cli_sha256": sha256_file(cli_path),
            "cmake_build_type": build_type,
            "cmake_cache_path": str(cache_path),
        },
        "data": {
            "data_dir": str(data_dir),
            "pick_data_sha256": sha256_file(data_dir / "pick_data.csv"),
            "dist_matrix_sha256": sha256_file(data_dir / "dist_matrix.csv"),
        },
        "commands": commands,
    }
    atomic_write_json(manifest_path, manifest)

    try:
        for index, ((wave_id, _), command) in enumerate(zip(WAVES, commands), start=1):
            print(f"[{index}/{len(WAVES)}] Running complete wave {wave_id}", flush=True)
            completed = subprocess.run(command, cwd=root, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"delegated runner failed for {wave_id} with exit code {completed.returncode}"
                )
            validate_completed_wave(out_root / wave_id, wave_id)
            manifest["completed_waves"].append(wave_id)
            atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["state"] = "failed"
        manifest["finished_utc"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(manifest_path, manifest)
        raise

    manifest["state"] = "complete"
    manifest["finished_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    print(f"Study complete: {out_root}")
    print(
        "Analyze with: "
        + subprocess.list2cmdline(
            [
                sys.executable,
                str(root / "scripts" / "retrospective_budget_analyze.py"),
                "--study-dir",
                str(out_root),
            ]
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    root = repo_root()
    parser = build_parser(root)
    args = parser.parse_args(argv)
    try:
        return run(args, root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
