import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_runtime_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        runs.groupby(["solver", "config_id"], as_index=False)
        .agg(mean_runtime_ms=("runtime_ms", "mean"))
        .sort_values("mean_runtime_ms")
    )
    plt.figure(figsize=(10, 5))
    plt.bar(summary["config_id"], summary["mean_runtime_ms"], color="#4F7CAC")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean runtime (ms)")
    plt.title("Runtime by configuration")
    plt.tight_layout()
    plt.savefig(output_dir / "runtime.png", dpi=150)
    plt.close()


def save_quality_runtime_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        runs.groupby(["solver", "config_id"], as_index=False)
        .agg(mean_runtime_ms=("runtime_ms", "mean"), mean_objective=("objective", "mean"))
    )
    plt.figure(figsize=(7, 5))
    for solver, frame in summary.groupby("solver"):
        plt.scatter(frame["mean_runtime_ms"], frame["mean_objective"], label=solver, s=80)
        for _, row in frame.iterrows():
            plt.annotate(row["config_id"], (row["mean_runtime_ms"], row["mean_objective"]), fontsize=8)
    plt.xlabel("Mean runtime (ms)")
    plt.ylabel("Mean objective")
    plt.title("Runtime versus objective quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "quality_vs_runtime.png", dpi=150)
    plt.close()


def save_convergence_plot(convergence: pd.DataFrame, output_dir: Path) -> None:
    if convergence.empty:
        return
    summary = (
        convergence.groupby(["solver", "config_id", "step"], as_index=False)
        .agg(mean_best_objective=("best_objective", "mean"))
    )
    plt.figure(figsize=(9, 5))
    for config_id, frame in summary.groupby("config_id"):
        plt.plot(frame["step"], frame["mean_best_objective"], label=config_id)
    plt.xlabel("Step")
    plt.ylabel("Mean best objective")
    plt.title("Convergence traces")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=150)
    plt.close()


def save_wave_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        runs.groupby(["wave_id", "solver"], as_index=False)
        .agg(mean_objective=("objective", "mean"))
    )
    pivot = summary.pivot(index="wave_id", columns="solver", values="mean_objective")
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("Mean objective")
    plt.title("Wave-by-wave performance")
    plt.tight_layout()
    plt.savefig(output_dir / "wave_by_wave.png", dpi=150)
    plt.close()


def save_extra_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    greedy = (
        runs[runs["solver"] == "greedy"][["wave_id", "objective"]]
        .groupby("wave_id", as_index=False)
        .mean()
        .rename(columns={"objective": "greedy_objective"})
    )
    gains = runs.merge(greedy, on="wave_id", how="left")
    gains = gains[gains["solver"].isin(["simulated_annealing", "genetic"])].copy()
    if gains.empty:
        return
    gains["relative_gain_vs_greedy"] = (
        gains["greedy_objective"] - gains["objective"]
    ) / gains["greedy_objective"].replace(0, pd.NA)

    plt.figure(figsize=(7, 5))
    for solver, frame in gains.groupby("solver"):
        plt.scatter(frame["overlap_density"], frame["relative_gain_vs_greedy"], label=solver, alpha=0.8)
    plt.xlabel("Wave overlap density")
    plt.ylabel("Relative gain vs greedy")
    plt.title("Overlap density versus improvement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "extra_analysis.png", dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(input_dir / "runs.csv")
    convergence_path = input_dir / "convergence.csv"
    convergence = pd.read_csv(convergence_path) if convergence_path.exists() else pd.DataFrame()

    save_runtime_plot(runs, output_dir)
    save_quality_runtime_plot(runs, output_dir)
    save_convergence_plot(convergence, output_dir)
    save_wave_plot(runs, output_dir)
    save_extra_plot(runs, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
