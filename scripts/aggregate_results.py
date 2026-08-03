"""Aggregate benchmark output without treating waves as independent runs.

``runs.csv`` contains one row per wave, although a statistical run consists of
all waves solved with one solver/configuration/seed.  This module therefore
keeps the wave and seed levels explicit and only computes configuration-level
statistics after summing the complete-instance result for every seed.
"""

import argparse
from pathlib import Path

import pandas as pd


RUN_COLUMNS = {
    "solver",
    "config_id",
    "wave_id",
    "seed",
    "objective",
    "runtime_ms",
    "overlap_density",
    "span_dispersion",
}
CONVERGENCE_COLUMNS = {
    "solver",
    "config_id",
    "wave_id",
    "seed",
    "step",
    "best_objective",
    "current_objective",
    "control_value",
}
CONTROL_NAMES = {
    "genetic": "population_diversity_mean_normalized_hamming",
    "simulated_annealing": "temperature",
}

# Two-sided 95% Student-t critical values.  Experiments normally use ten
# seeds, so using t rather than 1.96 avoids overly narrow small-sample CIs.
_T_975 = {
    1: 12.706204736432095,
    2: 4.302652729696142,
    3: 3.182446305284263,
    4: 2.776445105197799,
    5: 2.570581835636314,
    6: 2.446911851144969,
    7: 2.364624251592784,
    8: 2.306004135204166,
    9: 2.262157162854099,
    10: 2.228138851964939,
    11: 2.200985160082949,
    12: 2.178812829663418,
    13: 2.160368656461013,
    14: 2.144786687916927,
    15: 2.131449545559323,
    16: 2.119905299221011,
    17: 2.109815577833181,
    18: 2.100922040240960,
    19: 2.093024054408263,
    20: 2.085963447265836,
    21: 2.079613844727662,
    22: 2.073873067904015,
    23: 2.068657610419041,
    24: 2.063898561628021,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.051830516480283,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.042272456301237,
}


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _ci95_half_width(std: pd.Series, count: pd.Series) -> pd.Series:
    """Return a two-sided 95% CI half-width, or NA for a single observation."""

    critical = count.map(
        lambda value: _T_975.get(int(value) - 1, 1.96) if int(value) >= 2 else float("nan")
    )
    return critical * std / count.pow(0.5)


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return float("nan")
    return paired.iloc[:, 0].corr(paired.iloc[:, 1])


def build_seed_totals(runs: pd.DataFrame) -> pd.DataFrame:
    """Sum all waves to one complete-instance observation per seed."""

    _require_columns(runs, RUN_COLUMNS, "runs.csv")
    totals = (
        runs.groupby(["solver", "config_id", "seed"], as_index=False, dropna=False)
        .agg(
            total_objective=("objective", "sum"),
            total_runtime_ms=("runtime_ms", "sum"),
            wave_count=("wave_id", "nunique"),
        )
        .sort_values(["solver", "config_id", "seed"], kind="stable")
        .reset_index(drop=True)
    )
    expected_waves = runs.groupby(
        ["solver", "config_id"], dropna=False
    )["wave_id"].nunique()
    observed_index = pd.MultiIndex.from_frame(totals[["solver", "config_id"]])
    expected_count = expected_waves.reindex(observed_index).to_numpy()
    incomplete = totals["wave_count"].to_numpy() != expected_count
    if incomplete.any():
        examples = totals.loc[
            incomplete, ["solver", "config_id", "seed", "wave_count"]
        ].head(5)
        raise ValueError(
            "runs.csv does not contain every wave for each seed; "
            f"examples: {examples.to_dict(orient='records')}"
        )
    return totals


def build_config_summary(seed_totals: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, sample SD, best value and CI over independent seeds."""

    required = {
        "solver",
        "config_id",
        "seed",
        "total_objective",
        "total_runtime_ms",
        "wave_count",
    }
    _require_columns(seed_totals, required, "seed totals")
    summary = (
        seed_totals.groupby(["solver", "config_id"], as_index=False, dropna=False)
        .agg(
            mean_objective=("total_objective", "mean"),
            std_objective=("total_objective", "std"),
            best_objective=("total_objective", "min"),
            mean_runtime_ms=("total_runtime_ms", "mean"),
            std_runtime_ms=("total_runtime_ms", "std"),
            best_runtime_ms=("total_runtime_ms", "min"),
            seed_count=("seed", "nunique"),
            min_wave_count=("wave_count", "min"),
            max_wave_count=("wave_count", "max"),
        )
    )
    summary["objective_ci95"] = _ci95_half_width(
        summary["std_objective"], summary["seed_count"]
    )
    summary["runtime_ci95_ms"] = _ci95_half_width(
        summary["std_runtime_ms"], summary["seed_count"]
    )
    # Keep the legacy column name, but it now correctly counts independent
    # complete-instance seed runs instead of wave rows.
    summary["runs"] = summary["seed_count"]
    return summary.sort_values(
        ["mean_objective", "mean_runtime_ms", "solver", "config_id"], kind="stable"
    ).reset_index(drop=True)


def _select_greedy_config(runs: pd.DataFrame) -> str | None:
    greedy = runs[runs["solver"] == "greedy"]
    if greedy.empty:
        return None
    greedy_summary = build_config_summary(build_seed_totals(greedy))
    return str(greedy_summary.iloc[0]["config_id"])


def build_wave_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seeds within each wave/config and attach one greedy baseline."""

    _require_columns(runs, RUN_COLUMNS, "runs.csv")
    # Repeated rows with the same seed are not independent replicates.  Collapse
    # them before estimating variation over seeds.
    wave_seed = (
        runs.groupby(
            ["solver", "config_id", "wave_id", "seed"],
            as_index=False,
            dropna=False,
        )
        .agg(
            objective=("objective", "mean"),
            runtime_ms=("runtime_ms", "mean"),
            overlap_density=("overlap_density", "mean"),
            span_dispersion=("span_dispersion", "mean"),
        )
    )
    summary = (
        wave_seed.groupby(
            ["solver", "config_id", "wave_id"], as_index=False, dropna=False
        )
        .agg(
            mean_objective=("objective", "mean"),
            std_objective=("objective", "std"),
            best_objective=("objective", "min"),
            mean_runtime_ms=("runtime_ms", "mean"),
            std_runtime_ms=("runtime_ms", "std"),
            best_runtime_ms=("runtime_ms", "min"),
            seed_count=("seed", "nunique"),
            overlap_density=("overlap_density", "mean"),
            span_dispersion=("span_dispersion", "mean"),
        )
    )
    summary["objective_ci95"] = _ci95_half_width(
        summary["std_objective"], summary["seed_count"]
    )
    summary["runtime_ci95_ms"] = _ci95_half_width(
        summary["std_runtime_ms"], summary["seed_count"]
    )

    baseline_config = _select_greedy_config(runs)
    summary["greedy_config_id"] = baseline_config
    if baseline_config is None:
        summary["greedy_objective"] = float("nan")
    else:
        baseline = summary[
            (summary["solver"] == "greedy")
            & (summary["config_id"].astype(str) == baseline_config)
        ][["wave_id", "mean_objective"]].rename(
            columns={"mean_objective": "greedy_objective"}
        )
        summary = summary.drop(columns=["greedy_objective"], errors="ignore").merge(
            baseline, on="wave_id", how="left"
        )

    denominator = summary["greedy_objective"].replace(0, pd.NA)
    summary["relative_gain_vs_greedy"] = (
        summary["greedy_objective"] - summary["mean_objective"]
    ) / denominator
    summary["relative_gain_std"] = summary["std_objective"] / denominator.abs()
    summary["relative_gain_ci95"] = summary["objective_ci95"] / denominator.abs()
    return summary.sort_values(
        ["wave_id", "solver", "config_id"], kind="stable"
    ).reset_index(drop=True)


def compute_gain_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Return one gain observation per wave/config, after averaging seeds."""

    wave_summary = build_wave_summary(runs)
    return wave_summary.rename(
        columns={
            "mean_objective": "objective",
            "mean_runtime_ms": "runtime_ms",
        }
    )


def build_convergence_seed_totals(convergence: pd.DataFrame) -> pd.DataFrame:
    """Sum wave objectives per seed/step; average the non-additive control trace."""

    _require_columns(convergence, CONVERGENCE_COLUMNS, "convergence.csv")
    if convergence.empty:
        return pd.DataFrame(
            columns=[
                "solver",
                "config_id",
                "seed",
                "step",
                "total_best_objective",
                "total_current_objective",
                "mean_control_value",
                "wave_count",
            ]
        )

    # A repeated seed/wave/step is a duplicate measurement, not a new wave.
    wave_step = (
        convergence.groupby(
            ["solver", "config_id", "seed", "wave_id", "step"],
            as_index=False,
            dropna=False,
        )
        .agg(
            best_objective=("best_objective", "min"),
            current_objective=("current_objective", "mean"),
            control_value=("control_value", "mean"),
        )
    )
    totals = (
        wave_step.groupby(
            ["solver", "config_id", "seed", "step"],
            as_index=False,
            dropna=False,
        )
        .agg(
            total_best_objective=("best_objective", "sum"),
            total_current_objective=("current_objective", "sum"),
            mean_control_value=("control_value", "mean"),
            wave_count=("wave_id", "nunique"),
        )
    )

    # Every seed/step must cover the same full wave set for its configuration;
    # otherwise a missing wave would make a total look spuriously better.
    expected_waves = wave_step.groupby(
        ["solver", "config_id"], dropna=False
    )["wave_id"].nunique()
    observed_index = pd.MultiIndex.from_frame(
        totals[["solver", "config_id"]]
    )
    totals["expected_wave_count"] = expected_waves.reindex(observed_index).to_numpy()
    incomplete = totals["wave_count"] != totals["expected_wave_count"]
    if incomplete.any():
        examples = totals.loc[
            incomplete, ["solver", "config_id", "seed", "step"]
        ].head(5)
        raise ValueError(
            "convergence.csv has incomplete wave coverage for seed/step rows; "
            f"examples: {examples.to_dict(orient='records')}"
        )

    steps_per_seed = totals.groupby(
        ["solver", "config_id", "seed"], dropna=False
    )["step"].nunique()
    expected_steps = totals.groupby(
        ["solver", "config_id"], dropna=False
    )["step"].nunique()
    seed_index = steps_per_seed.index.droplevel("seed")
    incomplete_seed = steps_per_seed.to_numpy() != expected_steps.reindex(
        seed_index
    ).to_numpy()
    if incomplete_seed.any():
        examples = [
            {
                "solver": index[0],
                "config_id": index[1],
                "seed": index[2],
            }
            for index in steps_per_seed.index[incomplete_seed][:5]
        ]
        raise ValueError(
            "convergence.csv has incomplete step coverage for seeds; "
            f"examples: {examples}"
        )
    return totals.drop(columns="expected_wave_count").sort_values(
        ["solver", "config_id", "seed", "step"], kind="stable"
    ).reset_index(drop=True)


def build_convergence_summary(seed_totals: pd.DataFrame) -> pd.DataFrame:
    """Compute full-instance convergence mean, SD and 95% CI over seeds."""

    required = {
        "solver",
        "config_id",
        "seed",
        "step",
        "total_best_objective",
        "total_current_objective",
        "mean_control_value",
        "wave_count",
    }
    _require_columns(seed_totals, required, "convergence seed totals")
    if seed_totals.empty:
        return pd.DataFrame(
            columns=[
                "solver",
                "config_id",
                "step",
                "mean_best_objective",
                "std_best_objective",
                "ci95_best_objective",
                "mean_current_objective",
                "std_current_objective",
                "ci95_current_objective",
                "mean_control_value",
                "std_control_value",
                "ci95_control_value",
                "seed_count",
                "wave_count",
            ]
        )

    summary = (
        seed_totals.groupby(["solver", "config_id", "step"], as_index=False, dropna=False)
        .agg(
            mean_best_objective=("total_best_objective", "mean"),
            std_best_objective=("total_best_objective", "std"),
            mean_current_objective=("total_current_objective", "mean"),
            std_current_objective=("total_current_objective", "std"),
            mean_control_value=("mean_control_value", "mean"),
            std_control_value=("mean_control_value", "std"),
            seed_count=("seed", "nunique"),
            wave_count=("wave_count", "min"),
        )
    )
    summary["ci95_best_objective"] = _ci95_half_width(
        summary["std_best_objective"], summary["seed_count"]
    )
    summary["ci95_current_objective"] = _ci95_half_width(
        summary["std_current_objective"], summary["seed_count"]
    )
    summary["ci95_control_value"] = _ci95_half_width(
        summary["std_control_value"], summary["seed_count"]
    )
    return summary.sort_values(
        ["solver", "config_id", "step"], kind="stable"
    ).reset_index(drop=True)


def build_control_summary(convergence_summary: pd.DataFrame) -> pd.DataFrame:
    """Expose temperature/diversity independently from additive objectives."""

    columns = [
        "solver",
        "config_id",
        "step",
        "control_name",
        "mean_control_value",
        "std_control_value",
        "ci95_control_value",
        "seed_count",
        "wave_count",
    ]
    if convergence_summary.empty:
        return pd.DataFrame(columns=columns)
    control = convergence_summary[
        [
            "solver",
            "config_id",
            "step",
            "mean_control_value",
            "std_control_value",
            "ci95_control_value",
            "seed_count",
            "wave_count",
        ]
    ].copy()
    control["control_name"] = control["solver"].map(CONTROL_NAMES).fillna(
        "control_value"
    )
    return control[columns]


def build_wave_control_summary(convergence: pd.DataFrame) -> pd.DataFrame:
    """Retain wave-specific control traces and estimate variation over seeds."""

    _require_columns(convergence, CONVERGENCE_COLUMNS, "convergence.csv")
    columns = [
        "solver",
        "config_id",
        "wave_id",
        "step",
        "control_name",
        "mean_control_value",
        "std_control_value",
        "ci95_control_value",
        "seed_count",
    ]
    if convergence.empty:
        return pd.DataFrame(columns=columns)
    wave_seed = (
        convergence.groupby(
            ["solver", "config_id", "wave_id", "seed", "step"],
            as_index=False,
            dropna=False,
        )
        .agg(control_value=("control_value", "mean"))
    )
    summary = (
        wave_seed.groupby(
            ["solver", "config_id", "wave_id", "step"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_control_value=("control_value", "mean"),
            std_control_value=("control_value", "std"),
            seed_count=("seed", "nunique"),
        )
    )
    summary["ci95_control_value"] = _ci95_half_width(
        summary["std_control_value"], summary["seed_count"]
    )
    summary["control_name"] = summary["solver"].map(CONTROL_NAMES).fillna(
        "control_value"
    )
    return summary[columns].sort_values(
        ["solver", "config_id", "wave_id", "step"], kind="stable"
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(input_dir / "runs.csv")
    seed_totals = build_seed_totals(runs)
    summary = build_config_summary(seed_totals)
    wave_summary = build_wave_summary(runs)
    gains = compute_gain_table(runs)

    seed_totals.to_csv(output_dir / "seed_totals.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    wave_summary.to_csv(output_dir / "wave_summary.csv", index=False)

    extra = gains[gains["solver"].isin(["simulated_annealing", "genetic"])].copy()
    extra.to_csv(output_dir / "relative_gains.csv", index=False)

    correlations = []
    for solver, frame in extra.groupby("solver"):
        correlations.append(
            {
                "solver": solver,
                "wave_config_points": len(frame),
                "corr_overlap_gain": safe_corr(
                    frame["overlap_density"], frame["relative_gain_vs_greedy"]
                ),
                "corr_span_gain": safe_corr(
                    frame["span_dispersion"], frame["relative_gain_vs_greedy"]
                ),
            }
        )
    pd.DataFrame(
        correlations,
        columns=[
            "solver",
            "wave_config_points",
            "corr_overlap_gain",
            "corr_span_gain",
        ],
    ).to_csv(output_dir / "correlations.csv", index=False)

    convergence_path = input_dir / "convergence.csv"
    if convergence_path.exists():
        convergence = pd.read_csv(convergence_path)
        convergence_seed_totals = build_convergence_seed_totals(convergence)
        convergence_summary = build_convergence_summary(convergence_seed_totals)
    else:
        convergence = pd.DataFrame(columns=sorted(CONVERGENCE_COLUMNS))
        convergence_seed_totals = build_convergence_seed_totals(convergence)
        convergence_summary = build_convergence_summary(convergence_seed_totals)
    control_summary = build_control_summary(convergence_summary)
    wave_control_summary = build_wave_control_summary(convergence)
    convergence_seed_totals.to_csv(
        output_dir / "convergence_seed_totals.csv", index=False
    )
    convergence_summary.to_csv(output_dir / "convergence_summary.csv", index=False)
    control_summary.to_csv(output_dir / "control_summary.csv", index=False)
    wave_control_summary.to_csv(
        output_dir / "control_wave_summary.csv", index=False
    )
    wave_control_summary[wave_control_summary["solver"] == "genetic"].to_csv(
        output_dir / "ga_diversity_summary.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
