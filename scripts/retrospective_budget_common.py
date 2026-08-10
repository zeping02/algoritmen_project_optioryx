"""Shared, immutable specification for the retrospective full-budget study."""

from __future__ import annotations

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
SOLVER_SEEDS: tuple[int, ...] = tuple(range(10))
REPEATS = 1
EXPECTED_RUNS_PER_WAVE = 8 * len(SOLVER_SEEDS) * REPEATS
EXPECTED_TOTAL_RUNS = len(WAVES) * EXPECTED_RUNS_PER_WAVE


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    solver: str
    config_filename: str
    family: str
    budget: int
    budget_label: str


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("greedy", "greedy", "greedy_ls1_win8.cfg", "greedy", 0, "greedy"),
    ExperimentSpec(
        "sa_i120", "simulated_annealing", "sa_i120.cfg", "SA", 120, "120 iter."
    ),
    ExperimentSpec(
        "sa_i400", "simulated_annealing", "sa_i400.cfg", "SA", 400, "400 iter."
    ),
    ExperimentSpec(
        "sa_i2000", "simulated_annealing", "sa_i2000.cfg", "SA", 2000, "2.000 iter."
    ),
    ExperimentSpec(
        "sa_i10000",
        "simulated_annealing",
        "sa_i10000.cfg",
        "SA",
        10000,
        "10.000 iter.",
    ),
    ExperimentSpec("ga_g10", "genetic", "ga_g10.cfg", "GA", 10, "10 gen."),
    ExperimentSpec("ga_g20", "genetic", "ga_g20.cfg", "GA", 20, "20 gen."),
    ExperimentSpec("ga_g40", "genetic", "ga_g40.cfg", "GA", 40, "40 gen."),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path(root: Path, spec: ExperimentSpec) -> Path:
    return root / "configs" / "retrospective_budget" / spec.config_filename


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


def _expected_config(spec: ExperimentSpec) -> dict[str, str]:
    if spec.label == "greedy":
        return {
            "config_id": "retrospective_budget_greedy_ls-1_win-8",
            "local_search_passes": "1",
            "local_search_neighbor_window": "8",
            "route_elimination_passes": "0",
            "route_elimination_max_batch_size": "12",
            "route_elimination_beam_width": "1",
            "route_elimination_candidates_per_order": "1",
            "route_elimination_ordering_trials": "1",
            "post_elimination_local_search_passes": "0",
            "pair_rebuild_passes": "0",
            "pair_rebuild_neighbor_window": "8",
            "triple_rebuild_passes": "0",
            "triple_rebuild_neighbor_window": "8",
            "triple_rebuild_combination_window": "0",
            "post_triple_pair_rebuild_passes": "0",
            "post_triple_pair_rebuild_neighbor_window": "8",
            "post_pair_triple_rebuild_passes": "0",
            "post_pair_triple_rebuild_neighbor_window": "8",
            "post_pair_triple_rebuild_combination_window": "0",
            "post_second_triple_pair_rebuild_passes": "0",
            "post_second_triple_pair_rebuild_neighbor_window": "8",
            "final_local_search_passes": "0",
        }
    if spec.family == "SA":
        iterations = spec.budget
        alpha = (0.05 / 60.0) ** (1.0 / iterations)
        beta_text = {
            120: "0.1665277777777778",
            400: "0.0499583333333333",
            2000: "0.009991666666666667",
            10000: "0.0019983333333333333",
        }[iterations]
        return {
            "config_id": (
                "retrospective_budget_sa_c-geometric_n-relocate_start-greedy-"
                f"ls-1-win-8_i-{iterations}_t0-60_tf-0p05"
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
            "lundy_beta": beta_text,
        }
    if spec.family == "GA":
        generations = spec.budget
        return {
            "config_id": (
                "retrospective_budget_ga_start-greedy-ls-1-win-8_pop-12_"
                f"gen-{generations}_tour-3_mut-0p20_insert-0p50_elite-0p08"
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
    raise AssertionError(f"unhandled experiment {spec.label}")


def validate_configuration_files(root: Path) -> list[dict[str, str | int]]:
    metadata: list[dict[str, str | int]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for spec in EXPERIMENTS:
        path = config_path(root, spec).resolve()
        if not path.is_file():
            raise ValueError(f"missing study config: {path}")
        observed = read_key_value_config(path)
        expected = _expected_config(spec)
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            changed = sorted(
                key for key in set(expected) & set(observed) if expected[key] != observed[key]
            )
            raise ValueError(
                f"config {path} differs from the locked study specification; "
                f"missing={missing}, extra={extra}, changed={changed}"
            )
        if spec.family == "SA":
            alpha = float(observed["geometric_alpha"])
            beta = float(observed["lundy_beta"])
            expected_alpha = (0.05 / 60.0) ** (1.0 / spec.budget)
            expected_beta = (60.0 / 0.05 - 1.0) / (60.0 * spec.budget)
            if not math.isclose(alpha, expected_alpha, rel_tol=0.0, abs_tol=2e-15):
                raise ValueError(f"incorrect geometric alpha in {path}")
            if not math.isclose(beta, expected_beta, rel_tol=0.0, abs_tol=2e-15):
                raise ValueError(f"incorrect Lundy beta in {path}")
        config_id = observed["config_id"]
        digest = sha256_file(path)
        if config_id in seen_ids:
            raise ValueError(f"duplicate config_id: {config_id}")
        if digest in seen_hashes:
            raise ValueError(f"duplicate config file content: {path}")
        seen_ids.add(config_id)
        seen_hashes.add(digest)
        metadata.append(
            {
                "label": spec.label,
                "solver": spec.solver,
                "family": spec.family,
                "budget": spec.budget,
                "budget_label": spec.budget_label,
                "config_path": str(path),
                "config_id": config_id,
                "config_sha256": digest,
            }
        )
    return metadata
