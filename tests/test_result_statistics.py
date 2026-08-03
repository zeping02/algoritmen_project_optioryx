import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_results import (  # noqa: E402
    _ci95_half_width,
    build_config_summary,
    build_convergence_seed_totals,
    build_convergence_summary,
    build_seed_totals,
    compute_gain_table,
)
from scripts.analyze_experiment_suite import ci95 as suite_ci95  # noqa: E402


def make_runs() -> pd.DataFrame:
    rows = [
        ("greedy", "greedy_default", "wave_1", 0, 100.0, 10.0, 0.10, 1.0),
        ("greedy", "greedy_default", "wave_2", 0, 200.0, 20.0, 0.20, 2.0),
        ("simulated_annealing", "sa_test", "wave_1", 0, 90.0, 15.0, 0.10, 1.0),
        ("simulated_annealing", "sa_test", "wave_2", 0, 180.0, 25.0, 0.20, 2.0),
        ("simulated_annealing", "sa_test", "wave_1", 1, 80.0, 17.0, 0.10, 1.0),
        ("simulated_annealing", "sa_test", "wave_2", 1, 170.0, 27.0, 0.20, 2.0),
        ("genetic", "ga_test", "wave_1", 0, 92.0, 21.0, 0.10, 1.0),
        ("genetic", "ga_test", "wave_2", 0, 185.0, 31.0, 0.20, 2.0),
        ("genetic", "ga_test", "wave_1", 1, 88.0, 23.0, 0.10, 1.0),
        ("genetic", "ga_test", "wave_2", 1, 175.0, 33.0, 0.20, 2.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "solver",
            "config_id",
            "wave_id",
            "seed",
            "objective",
            "runtime_ms",
            "overlap_density",
            "span_dispersion",
        ],
    )


def make_convergence() -> pd.DataFrame:
    rows = []
    traces = {
        ("simulated_annealing", "sa_test", 0, "wave_1"): [(0, 100, 105, 10), (1, 90, 95, 5)],
        ("simulated_annealing", "sa_test", 0, "wave_2"): [(0, 200, 205, 12), (1, 180, 185, 7)],
        ("simulated_annealing", "sa_test", 1, "wave_1"): [(0, 110, 115, 11), (1, 80, 85, 6)],
        ("simulated_annealing", "sa_test", 1, "wave_2"): [(0, 210, 215, 13), (1, 180, 185, 8)],
        ("genetic", "ga_test", 0, "wave_1"): [(0, 105, 105, 0.4), (1, 92, 94, 0.3)],
        ("genetic", "ga_test", 0, "wave_2"): [(0, 205, 205, 0.6), (1, 185, 188, 0.5)],
        ("genetic", "ga_test", 1, "wave_1"): [(0, 100, 100, 0.5), (1, 88, 90, 0.4)],
        ("genetic", "ga_test", 1, "wave_2"): [(0, 200, 200, 0.7), (1, 175, 177, 0.6)],
    }
    for (solver, config_id, seed, wave_id), points in traces.items():
        for step, best, current, control in points:
            rows.append(
                (solver, config_id, wave_id, seed, step, best, current, control)
            )
    return pd.DataFrame(
        rows,
        columns=[
            "solver",
            "config_id",
            "wave_id",
            "seed",
            "step",
            "best_objective",
            "current_objective",
            "control_value",
        ],
    )


class ResultStatisticsTest(unittest.TestCase):
    def test_student_t_ci_uses_full_precision_critical_value(self) -> None:
        expected = 2.262157162854099 / math.sqrt(10.0)
        aggregate = _ci95_half_width(pd.Series([1.0]), pd.Series([10])).iloc[0]
        self.assertAlmostEqual(aggregate, expected, places=14)
        self.assertAlmostEqual(suite_ci95(1.0, 10), expected, places=14)

    def test_complete_instance_statistics_are_over_seeds(self) -> None:
        seed_totals = build_seed_totals(make_runs())
        sa_totals = seed_totals[seed_totals["config_id"] == "sa_test"]
        self.assertEqual(sa_totals["total_objective"].tolist(), [270.0, 250.0])
        self.assertEqual(sa_totals["total_runtime_ms"].tolist(), [40.0, 44.0])

        summary = build_config_summary(seed_totals)
        sa = summary[summary["config_id"] == "sa_test"].iloc[0]
        self.assertEqual(sa["runs"], 2)
        self.assertEqual(sa["mean_objective"], 260.0)
        self.assertAlmostEqual(sa["std_objective"], math.sqrt(200.0))
        self.assertEqual(sa["best_objective"], 250.0)
        self.assertEqual(sa["mean_runtime_ms"], 42.0)

    def test_gain_table_collapses_seeds_before_correlation_level(self) -> None:
        gains = compute_gain_table(make_runs())
        sa = gains[gains["config_id"] == "sa_test"].sort_values("wave_id")
        self.assertEqual(len(sa), 2)
        self.assertAlmostEqual(sa.iloc[0]["relative_gain_vs_greedy"], 0.15)
        self.assertAlmostEqual(sa.iloc[1]["relative_gain_vs_greedy"], 0.125)

    def test_convergence_sums_waves_and_keeps_control_non_additive(self) -> None:
        totals = build_convergence_seed_totals(make_convergence())
        sa_seed_zero = totals[
            (totals["config_id"] == "sa_test") & (totals["seed"] == 0)
        ].sort_values("step")
        self.assertEqual(sa_seed_zero["total_best_objective"].tolist(), [300, 270])
        self.assertEqual(sa_seed_zero["mean_control_value"].tolist(), [11.0, 6.0])

        summary = build_convergence_summary(totals)
        sa_step_zero = summary[
            (summary["config_id"] == "sa_test") & (summary["step"] == 0)
        ].iloc[0]
        self.assertEqual(sa_step_zero["mean_best_objective"], 310.0)
        self.assertAlmostEqual(sa_step_zero["std_best_objective"], math.sqrt(200.0))
        self.assertEqual(sa_step_zero["mean_control_value"], 11.5)
        self.assertEqual(sa_step_zero["seed_count"], 2)

    def test_command_line_scripts_write_legacy_and_new_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            analysis_dir = root / "analysis"
            plot_dir = root / "plots"
            input_dir.mkdir()
            make_runs().to_csv(input_dir / "runs.csv", index=False)
            make_convergence().to_csv(input_dir / "convergence.csv", index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "aggregate_results.py"),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(analysis_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "plot_results.py"),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(plot_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            expected = [
                analysis_dir / "summary.csv",
                analysis_dir / "seed_totals.csv",
                analysis_dir / "wave_summary.csv",
                analysis_dir / "convergence_summary.csv",
                analysis_dir / "control_summary.csv",
                analysis_dir / "control_wave_summary.csv",
                analysis_dir / "ga_diversity_summary.csv",
                plot_dir / "runtime.png",
                plot_dir / "quality_vs_runtime.png",
                plot_dir / "convergence.png",
                plot_dir / "wave_by_wave.png",
                plot_dir / "extra_analysis.png",
                plot_dir / "ga_diversity.png",
                plot_dir / "sa_temperature.png",
            ]
            self.assertEqual([path for path in expected if not path.exists()], [])


if __name__ == "__main__":
    unittest.main()
