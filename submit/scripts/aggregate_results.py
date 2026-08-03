import argparse
from pathlib import Path

import pandas as pd


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left, right], axis=1).dropna()
    if paired.empty or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return float("nan")
    return paired.iloc[:, 0].corr(paired.iloc[:, 1])


def compute_gain_table(runs: pd.DataFrame) -> pd.DataFrame:
    greedy = (
        runs[runs["solver"] == "greedy"][["wave_id", "objective"]]
        .groupby("wave_id", as_index=False)
        .mean()
        .rename(columns={"objective": "greedy_objective"})
    )
    merged = runs.merge(greedy, on="wave_id", how="left")
    merged["relative_gain_vs_greedy"] = (
        merged["greedy_objective"] - merged["objective"]
    ) / merged["greedy_objective"].replace(0, pd.NA)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(input_dir / "runs.csv")
    gains = compute_gain_table(runs)

    summary = (
        runs.groupby(["solver", "config_id"], as_index=False)
        .agg(
            mean_objective=("objective", "mean"),
            std_objective=("objective", "std"),
            best_objective=("objective", "min"),
            mean_runtime_ms=("runtime_ms", "mean"),
            runs=("objective", "count"),
        )
        .sort_values(["mean_objective", "mean_runtime_ms"])
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    extra = gains[gains["solver"].isin(["simulated_annealing", "genetic"])].copy()
    extra.to_csv(output_dir / "relative_gains.csv", index=False)

    correlations = []
    for solver, frame in extra.groupby("solver"):
        correlations.append(
            {
                "solver": solver,
                "corr_overlap_gain": safe_corr(frame["overlap_density"], frame["relative_gain_vs_greedy"]),
                "corr_span_gain": safe_corr(frame["span_dispersion"], frame["relative_gain_vs_greedy"]),
            }
        )
    pd.DataFrame(correlations).to_csv(output_dir / "correlations.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
