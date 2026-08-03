import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_experiment_suite.py"


def manifest_row(suite, solver, config_id, seeds, parameters, control="none"):
    serialized = ";".join(
        f"{key}={value}" for key, value in {"config_id": config_id, **parameters}.items()
    )
    return {
        "suite": suite,
        "solver": solver,
        "config_id": config_id,
        "seed_count": len(seeds),
        "instance_runs": len(seeds),
        "wave_runs": 2 * len(seeds),
        "seeds": ";".join(map(str, seeds)),
        "parameters": serialized,
        "convergence_control": control,
        "convergence_points_per_wave": 0,
    }


def wave_rows(solver, config_id, seed, objective, runtime):
    return [
        {
            "solver": solver,
            "config_id": config_id,
            "wave_id": "wave_1",
            "seed": seed,
            "objective": 0.4 * objective,
            "runtime_ms": 0.4 * runtime,
        },
        {
            "solver": solver,
            "config_id": config_id,
            "wave_id": "wave_2",
            "seed": seed,
            "objective": 0.6 * objective,
            "runtime_ms": 0.6 * runtime,
        },
    ]


def run_analysis(input_dir, output_dir, suite_type, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--suite-type",
            suite_type,
            *extra,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class ExperimentSuiteAnalysisTest(unittest.TestCase):
    def test_help(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--suite-type {sa,ga,seed}", completed.stdout)
        self.assertIn("--baseline-config", completed.stdout)

    def test_sa_factorial_outputs_known_paired_effects(self):
        manifest = []
        runs = []
        for cooling, move, start in itertools.product(
            ["geometric", "lundy_mees"], ["relocate", "swap"], ["greedy", "randomized"]
        ):
            config_id = f"sa_{cooling}_{move}_{start}"
            manifest.append(
                manifest_row(
                    "sa_factorial",
                    "simulated_annealing",
                    config_id,
                    [0, 1, 2],
                    {
                        "cooling": cooling,
                        "neighborhood": move,
                        "start_strategy": start,
                        "iterations": "400",
                    },
                    "temperature",
                )
            )
            for seed in [0, 1, 2]:
                cooling_alt = cooling == "lundy_mees"
                move_alt = move == "swap"
                start_alt = start == "randomized"
                objective = (
                    1000
                    + 5 * seed
                    + 40 * cooling_alt
                    - 20 * move_alt
                    + 30 * start_alt
                    + 12 * cooling_alt * move_alt
                )
                runtime = 100 + seed + 10 * cooling_alt + 4 * move_alt + 2 * start_alt
                runs.extend(
                    wave_rows("simulated_annealing", config_id, seed, objective, runtime)
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            pd.DataFrame(runs).to_csv(input_dir / "runs.csv", index=False)
            pd.DataFrame(manifest).sample(frac=1, random_state=3).to_csv(
                input_dir / "suite_manifest.csv", index=False
            )
            completed = run_analysis(input_dir, output_dir, "sa")
            self.assertEqual(completed.returncode, 0, completed.stderr)

            main = pd.read_csv(output_dir / "sa_main_effects.csv")
            cooling = main[
                (main["factor"] == "cooling") & (main["level"] == "lundy_mees")
            ].iloc[0]
            self.assertAlmostEqual(cooling["objective_delta_vs_reference_mean"], 46.0)
            interactions = pd.read_csv(output_dir / "sa_interactions.csv")
            cooling_move = interactions[
                interactions["factors"] == "cooling:neighborhood"
            ].iloc[0]
            self.assertAlmostEqual(cooling_move["objective_contrast_mean"], 12.0)
            for filename in [
                "sa_config_summary.csv",
                "sa_effects.png",
                "sa_effects.pdf",
                "analysis_protocol.json",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

    def test_ga_ofat_uses_paired_seeds_and_final_diversity(self):
        baseline = {
            "population_size": "40",
            "generations": "120",
            "tournament_size": "3",
            "mutation_rate": "0.20",
            "insertion_rate": "0.50",
            "elitism_fraction": "0.05",
        }
        variants = [
            baseline,
            {**baseline, "population_size": "20"},
            {**baseline, "population_size": "80"},
            {**baseline, "generations": "60"},
            {**baseline, "generations": "240"},
            {**baseline, "mutation_rate": "0.05"},
            {**baseline, "mutation_rate": "0.40"},
            {**baseline, "elitism_fraction": "0.01"},
            {**baseline, "elitism_fraction": "0.15"},
        ]
        offsets = {
            ("population_size", "20"): -12.0,
            ("population_size", "80"): -16.0,
            ("generations", "60"): 8.0,
            ("generations", "240"): -10.0,
            ("mutation_rate", "0.05"): 5.0,
            ("mutation_rate", "0.40"): -4.0,
            ("elitism_fraction", "0.01"): 3.0,
            ("elitism_fraction", "0.15"): 7.0,
        }
        manifest = []
        runs = []
        convergence = []
        baseline_id = None
        for parameters in variants:
            config_id = (
                f"ga_p{parameters['population_size']}_g{parameters['generations']}_"
                f"m{parameters['mutation_rate']}_e{parameters['elitism_fraction']}"
            )
            if parameters == baseline:
                baseline_id = config_id
                offset = 0.0
            else:
                changed = next(
                    (factor, parameters[factor])
                    for factor in [
                        "population_size",
                        "generations",
                        "mutation_rate",
                        "elitism_fraction",
                    ]
                    if parameters[factor] != baseline[factor]
                )
                offset = offsets[changed]
            manifest.append(
                manifest_row(
                    "ga_sensitivity",
                    "genetic",
                    config_id,
                    ["0", "1", "2"],
                    parameters,
                    "population_diversity_mean_normalized_hamming",
                )
            )
            generation = int(parameters["generations"])
            for seed in [0, 1, 2]:
                runs.extend(
                    wave_rows(
                        "genetic",
                        config_id,
                        seed,
                        1000 + 3 * seed + offset,
                        100 + seed + generation / 10,
                    )
                )
                for wave_index, wave in enumerate(["wave_1", "wave_2"]):
                    for step, diversity in [(0, 0.8), (generation, 0.4 + 0.01 * seed + 0.02 * wave_index)]:
                        convergence.append(
                            {
                                "solver": "genetic",
                                "config_id": config_id,
                                "wave_id": wave,
                                "seed": str(seed),
                                "step": step,
                                "best_objective": 1,
                                "current_objective": 1,
                                "control_value": diversity,
                            }
                        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            pd.DataFrame(runs).sample(frac=1, random_state=4).to_csv(
                input_dir / "runs.csv", index=False
            )
            pd.DataFrame(manifest).sample(frac=1, random_state=5).to_csv(
                input_dir / "suite_manifest.csv", index=False
            )
            pd.DataFrame(convergence).to_csv(input_dir / "convergence.csv", index=False)
            completed = run_analysis(input_dir, output_dir, "ga")
            self.assertEqual(completed.returncode, 0, completed.stderr)

            summary = pd.read_csv(output_dir / "ga_sensitivity_summary.csv")
            self.assertEqual(set(summary["baseline_config_id"]), {baseline_id})
            population_20 = summary[
                (summary["changed_factor"] == "population_size")
                & (pd.to_numeric(summary["variant_value"], errors="coerce") == 20)
            ].iloc[0]
            self.assertAlmostEqual(population_20["objective_delta_mean"], -12.0)
            self.assertEqual(population_20["paired_seed_count"], 3)
            self.assertEqual(population_20["final_diversity_n"], 3)
            for filename in [
                "ga_paired_seed_deltas.csv",
                "ga_sensitivity.png",
                "ga_sensitivity.pdf",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            (input_dir / "convergence.csv").unlink()
            without_diversity = root / "output_without_diversity"
            completed = run_analysis(
                input_dir,
                without_diversity,
                "ga",
                "--baseline-config",
                baseline_id,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            no_diversity = pd.read_csv(
                without_diversity / "ga_sensitivity_summary.csv"
            )
            self.assertTrue((no_diversity["final_diversity_n"] == 0).all())

    def test_seed_outputs_paper_design_and_matched_ablation(self):
        seed_rules = ["smallest_snake_span", "largest_snake_span"]
        selections = [
            "maximum_location_overlap",
            "minimum_span_expansion",
            "bounded_two_step_lookahead",
            "minimum_incremental_cost",
        ]
        objectives = {
            "maximum_location_overlap": 1100,
            "minimum_span_expansion": 1080,
            "bounded_two_step_lookahead": 1000,
            "minimum_incremental_cost": 1010,
        }
        manifest = []
        runs = []
        for seed_index, seed_rule in enumerate(seed_rules):
            for selection in selections:
                config_id = f"seed_{seed_rule}_{selection}"
                parameters = {"seed_rule": seed_rule, "selection_rule": selection}
                if selection == "bounded_two_step_lookahead":
                    parameters["lookahead_width"] = "8"
                manifest.append(
                    manifest_row("seed_factorial", "seed_selection", config_id, [0], parameters)
                )
                runs.extend(
                    wave_rows(
                        "seed_selection",
                        config_id,
                        0,
                        objectives[selection] + 5 * seed_index,
                        20 + seed_index,
                    )
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            pd.DataFrame(runs).to_csv(input_dir / "runs.csv", index=False)
            pd.DataFrame(manifest).to_csv(input_dir / "suite_manifest.csv", index=False)
            completed = run_analysis(input_dir, output_dir, "seed")
            self.assertEqual(completed.returncode, 0, completed.stderr)

            paper = pd.read_csv(output_dir / "seed_paper_2x2.csv")
            self.assertEqual(len(paper), 4)
            ablation = pd.read_csv(output_dir / "seed_custom_ablation.csv")
            self.assertEqual(len(ablation), 2)
            self.assertTrue(
                (ablation["objective_delta_custom_minus_immediate_mean"] == -10).all()
            )
            self.assertTrue(ablation["objective_delta_custom_minus_immediate_std"].isna().all())
            for filename in ["seed_factorial.png", "seed_factorial.pdf"]:
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
