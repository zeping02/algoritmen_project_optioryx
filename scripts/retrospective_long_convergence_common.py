"""Locked specification and validation helpers for the long convergence study."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


WAVES: tuple[tuple[str, int], ...] = (
    ("W000001", 1041),
    ("W000002", 1076),
    ("W000003", 1219),
    ("W000004", 800),
    ("W000005", 909),
    ("W000006", 1076),
    ("W000007", 974),
    ("W000008", 945),
    ("W000009", 1148),
    ("W000010", 858),
    ("W000011", 50),
    ("W000012", 150),
)
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    solver: str
    family: str
    budget: int
    config_filename: str


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("sa_i10000", "simulated_annealing", "SA", 10_000, "sa_i10000.cfg"),
    ExperimentSpec("sa_i50000", "simulated_annealing", "SA", 50_000, "sa_i50000.cfg"),
    ExperimentSpec("ga_g40", "genetic", "GA", 40, "ga_g40.cfg"),
    ExperimentSpec("ga_g100", "genetic", "GA", 100, "ga_g100.cfg"),
    ExperimentSpec("ga_g200", "genetic", "GA", 200, "ga_g200.cfg"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path(root: Path, spec: ExperimentSpec) -> Path:
    return root / "configs" / "retrospective_long_convergence" / spec.config_filename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_key_value_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            key, separator, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not separator or not key or not value:
                raise ValueError(f"invalid config line {line_number} in {path}")
            if key in values:
                raise ValueError(f"duplicate config key {key!r} in {path}")
            values[key] = value
    return values


def expected_config(spec: ExperimentSpec) -> dict[str, str]:
    if spec.family == "SA":
        iterations = spec.budget
        alpha = (0.05 / 60.0) ** (1.0 / iterations)
        beta = (60.0 / 0.05 - 1.0) / (60.0 * iterations)
        return {
            "config_id": (
                "retrospective_long_convergence_sa_c-geometric_n-relocate_"
                f"start-greedy-ls-1-win-8_i-{iterations}_t0-60_tf-0p05"
            ),
            "start_strategy": "greedy",
            "greedy_start_local_search_passes": "1",
            "greedy_start_local_search_neighbor_window": "8",
            "neighborhood": "relocate",
            "cooling": "geometric",
            "iterations": str(iterations),
            "initial_temperature": "60",
            "final_temperature": "0.05",
            "geometric_alpha": repr(alpha),
            "lundy_beta": repr(beta),
        }
    if spec.family == "GA":
        generations = spec.budget
        return {
            "config_id": (
                "retrospective_long_convergence_ga_start-greedy-ls-1-win-8_"
                f"pop-12_gen-{generations}_tour-3_mut-0p20_insert-0p50_elite-0p08"
            ),
            "greedy_start_local_search_passes": "1",
            "greedy_start_local_search_neighbor_window": "8",
            "population_size": "12",
            "generations": str(generations),
            "tournament_size": "3",
            "mutation_rate": "0.20",
            "insertion_rate": "0.50",
            "elitism_fraction": "0.08",
        }
    raise AssertionError(f"unsupported family: {spec.family}")


def validate_configuration_files(root: Path) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for spec in EXPERIMENTS:
        path = config_path(root, spec).resolve()
        if not path.is_file():
            raise ValueError(f"missing convergence config: {path}")
        observed = read_key_value_config(path)
        expected = expected_config(spec)
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            changed = sorted(
                key for key in set(expected) & set(observed) if expected[key] != observed[key]
            )
            raise ValueError(
                f"config {path} differs from the locked specification; "
                f"missing={missing}, extra={extra}, changed={changed}"
            )
        config_id = observed["config_id"]
        if config_id in seen_ids:
            raise ValueError(f"duplicate config_id: {config_id}")
        seen_ids.add(config_id)
        if spec.family == "SA":
            expected_alpha = (0.05 / 60.0) ** (1.0 / spec.budget)
            expected_beta = (60.0 / 0.05 - 1.0) / (60.0 * spec.budget)
            if not math.isclose(
                float(observed["geometric_alpha"]), expected_alpha, rel_tol=0.0, abs_tol=2e-15
            ):
                raise ValueError(f"incorrect geometric_alpha in {path}")
            if not math.isclose(
                float(observed["lundy_beta"]), expected_beta, rel_tol=0.0, abs_tol=2e-15
            ):
                raise ValueError(f"incorrect lundy_beta in {path}")
        metadata.append(
            {
                "label": spec.label,
                "solver": spec.solver,
                "family": spec.family,
                "budget": spec.budget,
                "config_path": str(path),
                "config_id": config_id,
                "config_sha256": sha256_file(path),
            }
        )
    return metadata


def count_orders_by_wave(pick_path: Path) -> dict[str, int]:
    orders: dict[str, set[str]] = {}
    with pick_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"WAVE_ID", "ORDER_ID"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{pick_path} must contain WAVE_ID and ORDER_ID")
        for line_number, row in enumerate(reader, start=2):
            wave_id = (row.get("WAVE_ID") or "").strip()
            order_id = (row.get("ORDER_ID") or "").strip()
            if not wave_id or not order_id:
                raise ValueError(f"empty wave/order id at line {line_number} in {pick_path}")
            orders.setdefault(wave_id, set()).add(order_id)
    return {wave_id: len(order_ids) for wave_id, order_ids in orders.items()}


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        seed = int(token)
        if seed < 0:
            raise ValueError("solver seeds must be non-negative")
        if seed in seeds:
            raise ValueError(f"duplicate solver seed: {seed}")
        seeds.append(seed)
    if not seeds:
        raise ValueError("at least one solver seed is required")
    return tuple(seeds)
