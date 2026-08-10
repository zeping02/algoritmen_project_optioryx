"""Synthetic smoke test for the paired SA-factorial inference script."""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_ANALYZER = ROOT / "scripts" / "analyze_experiment_suite.py"
INFERENCE = ROOT / "scripts" / "retrospective_sa_factorial_inference.py"

LEVELS = {
    "cooling": ("geometric", "lundy_mees"),
    "neighborhood": ("relocate", "swap"),
    "start_strategy": ("greedy", "randomized"),
}


def sign(level: str, alternatives: str) -> int:
    return 1 if level == alternatives else -1


def make_fixture(input_dir: Path) -> None:
    input_dir.mkdir(parents=True)
    manifests: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    instances: list[dict[str, object]] = []
    for cooling, neighborhood, start_strategy in itertools.product(*LEVELS.values()):
        config_id = f"sa_c-{cooling}_n-{neighborhood}_start-{start_strategy}"
        parameters = {
            "config_id": config_id,
            "cooling": cooling,
            "neighborhood": neighborhood,
            "start_strategy": start_strategy,
            "iterations": "400",
            "initial_temperature": "60",
            "final_temperature": "0.05",
        }
        manifests.append(
            {
                "suite": "synthetic_sa_factorial",
                "solver": "simulated_annealing",
                "config_id": config_id,
                "seed_count": 10,
                "instance_runs": 10,
                "wave_runs": 120,
                "seeds": ";".join(str(seed) for seed in range(10)),
                "parameters": ";".join(f"{key}={value}" for key, value in parameters.items()),
            }
        )
        a = sign(cooling, "lundy_mees")
        b = sign(neighborhood, "swap")
        c = sign(start_strategy, "randomized")
        for seed in range(10):
            objective_total = (
                100_000.0
                + 100.0 * seed
                + 40.0 * a
                + 20.0 * b
                + 10.0 * c
                + 8.0 * a * b
                + 6.0 * a * c
                + 4.0 * b * c
                + 2.0 * a * b * c
            )
            runtime_total = (
                1_000.0
                + seed
                + 3.0 * a
                + 2.0 * b
                + c
                + 0.5 * a * b
                + 0.25 * a * c
                + 0.125 * b * c
                + 0.0625 * a * b * c
            )
            config_seed_rows: list[dict[str, object]] = []
            for wave_index in range(12):
                config_seed_rows.append(
                    {
                        "solver": "simulated_annealing",
                        "config_id": config_id,
                        "wave_id": f"W{wave_index + 1:06d}",
                        "seed": seed,
                        "order_count": 100 + wave_index,
                        "objective": objective_total / 12.0,
                        "runtime_ms": runtime_total / 12.0,
                    }
                )
            runs.extend(config_seed_rows)
            instances.append(
                {
                    "solver": "simulated_annealing",
                    "config_id": config_id,
                    "seed": seed,
                    "order_count": sum(int(row["order_count"]) for row in config_seed_rows),
                    "objective": math.fsum(float(row["objective"]) for row in config_seed_rows),
                    # De volledige-instantietimer bevat ook evaluatoropbouw en
                    # eindvalidatie buiten de twaalf wave-solvertimers.
                    "runtime_ms": math.fsum(float(row["runtime_ms"]) for row in config_seed_rows) + 5.0,
                    "feasible": True,
                }
            )
    pd.DataFrame(manifests).to_csv(input_dir / "suite_manifest.csv", index=False)
    pd.DataFrame(runs).to_csv(input_dir / "runs.csv", index=False)
    pd.DataFrame(instances).to_csv(input_dir / "instance_runs.csv", index=False)


class RetrospectiveSaFactorialInferenceTest(unittest.TestCase):
    def test_complete_synthetic_cli_and_guards(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sa_factorial_smoke_") as temporary:
            temporary_path = Path(temporary)
            input_dir = temporary_path / "suite"
            prior_dir = input_dir / "factorial_analysis"
            output_dir = temporary_path / "inference"
            make_fixture(input_dir)
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = str(temporary_path / "matplotlib")
            prior = subprocess.run(
                [
                    sys.executable,
                    str(BASE_ANALYZER),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(prior_dir),
                    "--suite-type",
                    "sa",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(prior.returncode, 0, prior.stderr)

            result = subprocess.run(
                [
                    sys.executable,
                    str(INFERENCE),
                    "--input-dir",
                    str(input_dir),
                    "--out-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            effects = pd.read_csv(output_dir / "sa_factorial_effects.csv").set_index("effect")
            self.assertEqual(len(effects), 7)
            expected_objective_effects = {
                "cooling": 80.0,
                "neighborhood": 40.0,
                "start_strategy": 20.0,
                "cooling:neighborhood": 32.0,
                "cooling:start_strategy": 24.0,
                "neighborhood:start_strategy": 16.0,
                "cooling:neighborhood:start_strategy": 16.0,
            }
            for effect, expected in expected_objective_effects.items():
                self.assertAlmostEqual(float(effects.loc[effect, "objective_mean"]), expected, places=6)
            self.assertTrue((effects["objective_p_holm_7"] == 0.0).all())
            self.assertTrue((effects["objective_holm_reject_0p05"] == True).all())  # noqa: E712
            protocol = json.loads((output_dir / "analysis_protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["design"]["configurations"], 8)
            self.assertTrue(protocol["validation"]["prior_factorial_analysis_recomputed"])
            self.assertEqual(len(protocol["sha256"]["prior_factorial_analysis_artifacts"]), 6)

            overwrite = subprocess.run(
                [
                    sys.executable,
                    str(INFERENCE),
                    "--input-dir",
                    str(input_dir),
                    "--out-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(overwrite.returncode, 2)
            self.assertIn("refusing to overwrite", overwrite.stderr)

            with (input_dir / "runs.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            hash_failure = subprocess.run(
                [
                    sys.executable,
                    str(INFERENCE),
                    "--input-dir",
                    str(input_dir),
                    "--out-dir",
                    str(temporary_path / "second_inference"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(hash_failure.returncode, 2)
            self.assertIn("SHA-256 mismatch", hash_failure.stderr)


if __name__ == "__main__":
    unittest.main()
