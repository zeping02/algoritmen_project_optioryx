"""Reproducible per-wave analysis for the Release budget study.

The statistical sampling unit is a solver seed within one wave.  Candidate
objectives are paired with the deterministic greedy run carrying the same seed
label.  Positive gains therefore always mean a shorter route than greedy.

This script reads existing result files only.  It never launches a solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, t as student_t


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent
DEFAULT_STUDY_DIR = DEFAULT_OUTPUT_DIR.parent
DEFAULT_REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_FEATURE_SOURCE = (
    DEFAULT_REPOSITORY_ROOT
    / "results"
    / "comparison_fast_matched"
    / "runs.csv"
)

METHODS = {
    "greedy": {"solver": "greedy", "display_name": "greedy"},
    "sa_i10000": {
        "solver": "simulated_annealing",
        "display_name": "SA-10000",
    },
    "ga_g40": {"solver": "genetic", "display_name": "GA-40"},
}
CANDIDATES = ("sa_i10000", "ga_g40")
FEATURES = ("order_count", "overlap_density", "span_dispersion")
EXPECTED_WAVES = 12
EXPECTED_SEEDS = tuple(range(10))
TOLERANCE = 1e-9

OUTPUT_FILENAMES = {
    "method_summary": "release_wave_method_summary.csv",
    "paired_seed_differences": "release_wave_paired_seed_differences.csv",
    "paired_comparisons": "release_wave_paired_comparisons.csv",
    "features_and_gains": "release_wave_features_and_mean_gains.csv",
    "feature_associations": "release_wave_feature_associations.csv",
    "feature_leave_one_out": "release_wave_feature_leave_one_out.csv",
    "report": "release_wave_analysis_report.md",
    "protocol": "release_wave_analysis_protocol.json",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=DEFAULT_STUDY_DIR,
        help="Study directory containing W*/runs.csv.",
    )
    parser.add_argument(
        "--feature-source",
        type=Path,
        default=DEFAULT_FEATURE_SOURCE,
        help=(
            "Existing benchmark runs.csv containing the deterministic wave "
            "features overlap_density and span_dispersion."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the analysis outputs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level for Holm correction.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} misses required columns: {', '.join(missing)}")


def wave_sort_key(wave_id: str) -> tuple[int, str]:
    digits = "".join(character for character in str(wave_id) if character.isdigit())
    return (int(digits) if digits else sys.maxsize, str(wave_id))


def sample_summary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("A finite, non-empty one-dimensional sample is required.")
    count = int(len(array))
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1)) if count >= 2 else float("nan")
    if count >= 2:
        critical = float(student_t.ppf(0.975, count - 1))
        half_width = critical * sample_sd / math.sqrt(count)
    else:
        half_width = float("nan")
    return {
        "n": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_half_width": half_width,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
    }


def two_sided_one_sample_t_pvalue(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return float("nan")
    mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    if sample_sd <= TOLERANCE:
        return 1.0 if abs(mean) <= TOLERANCE else 0.0
    statistic = mean / (sample_sd / math.sqrt(len(array)))
    return float(2.0 * student_t.sf(abs(statistic), len(array) - 1))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm-adjusted p-values while preserving the original index."""

    if p_values.isna().any():
        raise ValueError("Holm correction requires finite p-values.")
    ordered = p_values.sort_values(kind="stable")
    family_size = len(ordered)
    adjusted_sorted: list[float] = []
    running_maximum = 0.0
    for rank, raw_value in enumerate(ordered.to_numpy(dtype=float)):
        scaled = min(1.0, (family_size - rank) * raw_value)
        running_maximum = max(running_maximum, scaled)
        adjusted_sorted.append(running_maximum)
    adjusted = pd.Series(adjusted_sorted, index=ordered.index, dtype=float)
    return adjusted.reindex(p_values.index)


def load_and_validate_runs(study_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    run_files = sorted(study_dir.glob("W*/runs.csv"), key=lambda path: wave_sort_key(path.parent.name))
    if len(run_files) != EXPECTED_WAVES:
        raise ValueError(
            f"Expected {EXPECTED_WAVES} wave run files, found {len(run_files)} in {study_dir}."
        )

    required = {
        "wave_id",
        "total_orders_in_wave",
        "experiment_label",
        "solver",
        "solver_seed",
        "status",
        "timed_out",
        "feasible",
        "objective",
        "solver_runtime_ms",
        "wall_process_ms",
        "config_sha256",
    }
    frames: list[pd.DataFrame] = []
    for path in run_files:
        frame = pd.read_csv(path)
        require_columns(frame, required, path)
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)

    selected = runs[runs["experiment_label"].isin(METHODS)].copy()
    expected_rows = EXPECTED_WAVES * len(METHODS) * len(EXPECTED_SEEDS)
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} selected runs, found {len(selected)}.")
    if selected.duplicated(["wave_id", "experiment_label", "solver_seed"]).any():
        raise ValueError("Duplicate wave/configuration/seed rows found.")
    if not (selected["status"] == "success").all():
        raise ValueError("At least one selected run did not finish successfully.")
    if selected["timed_out"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("At least one selected run timed out.")
    feasible = selected["feasible"].astype(str).str.lower().isin({"true", "1"})
    if not feasible.all():
        raise ValueError("At least one selected run is infeasible.")
    for column in ("objective", "solver_runtime_ms", "wall_process_ms"):
        numeric = pd.to_numeric(selected[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Selected runs contain invalid {column} values.")
        selected[column] = numeric

    waves = sorted(selected["wave_id"].astype(str).unique(), key=wave_sort_key)
    if len(waves) != EXPECTED_WAVES:
        raise ValueError(f"Expected {EXPECTED_WAVES} waves, found {len(waves)}.")
    for wave_id in waves:
        wave = selected[selected["wave_id"].astype(str) == wave_id]
        if wave["total_orders_in_wave"].nunique() != 1:
            raise ValueError(f"Inconsistent order count for {wave_id}.")
        for label, metadata in METHODS.items():
            group = wave[wave["experiment_label"] == label]
            seeds = tuple(sorted(int(value) for value in group["solver_seed"].unique()))
            if seeds != EXPECTED_SEEDS:
                raise ValueError(f"Unexpected seeds for {wave_id}/{label}: {seeds}")
            if group["solver"].nunique() != 1 or group["solver"].iloc[0] != metadata["solver"]:
                raise ValueError(f"Unexpected solver for {wave_id}/{label}.")
            if group["config_sha256"].nunique() != 1:
                raise ValueError(f"Multiple config hashes for {wave_id}/{label}.")
        greedy = wave[wave["experiment_label"] == "greedy"]
        if float(greedy["objective"].max() - greedy["objective"].min()) > 1e-6:
            raise ValueError(f"Greedy objective unexpectedly depends on seed in {wave_id}.")

    return selected.sort_values(
        ["wave_id", "experiment_label", "solver_seed"], kind="stable"
    ).reset_index(drop=True), run_files


def build_method_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = runs.groupby(["wave_id", "experiment_label"], sort=False)
    for (wave_id, label), group in grouped:
        objective = sample_summary(group["objective"])
        solver_time = sample_summary(group["solver_runtime_ms"])
        wall_time = sample_summary(group["wall_process_ms"])
        rows.append(
            {
                "wave_id": wave_id,
                "order_count": int(group["total_orders_in_wave"].iloc[0]),
                "experiment_label": label,
                "method": METHODS[label]["display_name"],
                "solver": METHODS[label]["solver"],
                "config_sha256": str(group["config_sha256"].iloc[0]),
                "seed_count": objective["n"],
                "objective_mean": objective["mean"],
                "objective_sample_sd": objective["sample_sd"],
                "objective_ci95_low": objective["ci95_low"],
                "objective_ci95_high": objective["ci95_high"],
                "objective_best": objective["minimum"],
                "objective_worst": objective["maximum"],
                "solver_runtime_ms_mean": solver_time["mean"],
                "solver_runtime_ms_sample_sd": solver_time["sample_sd"],
                "solver_runtime_ms_ci95_low": solver_time["ci95_low"],
                "solver_runtime_ms_ci95_high": solver_time["ci95_high"],
                "wall_process_ms_mean": wall_time["mean"],
                "wall_process_ms_sample_sd": wall_time["sample_sd"],
                "wall_process_ms_ci95_low": wall_time["ci95_low"],
                "wall_process_ms_ci95_high": wall_time["ci95_high"],
            }
        )
    result = pd.DataFrame(rows)
    label_order = {label: index for index, label in enumerate(METHODS)}
    result["_wave"] = result["wave_id"].map(lambda value: wave_sort_key(str(value))[0])
    result["_method"] = result["experiment_label"].map(label_order)
    return result.sort_values(["_wave", "_method"], kind="stable").drop(
        columns=["_wave", "_method"]
    ).reset_index(drop=True)


def build_paired_tables(runs: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    objectives = runs.pivot(
        index=["wave_id", "solver_seed"], columns="experiment_label", values="objective"
    )
    solver_times = runs.pivot(
        index=["wave_id", "solver_seed"],
        columns="experiment_label",
        values="solver_runtime_ms",
    )
    wall_times = runs.pivot(
        index=["wave_id", "solver_seed"], columns="experiment_label", values="wall_process_ms"
    )
    order_counts = runs.groupby("wave_id")["total_orders_in_wave"].first().astype(int)

    raw_rows: list[dict[str, object]] = []
    for wave_id in sorted(order_counts.index.astype(str), key=wave_sort_key):
        for candidate in CANDIDATES:
            for seed in EXPECTED_SEEDS:
                greedy_objective = float(objectives.loc[(wave_id, seed), "greedy"])
                candidate_objective = float(objectives.loc[(wave_id, seed), candidate])
                absolute_gain = greedy_objective - candidate_objective
                raw_rows.append(
                    {
                        "wave_id": wave_id,
                        "order_count": int(order_counts.loc[wave_id]),
                        "candidate_label": candidate,
                        "candidate_method": METHODS[candidate]["display_name"],
                        "solver_seed": seed,
                        "greedy_objective": greedy_objective,
                        "candidate_objective": candidate_objective,
                        "paired_gain_abs": absolute_gain,
                        "paired_gain_pct": 100.0 * absolute_gain / greedy_objective,
                        "greedy_solver_runtime_ms": float(
                            solver_times.loc[(wave_id, seed), "greedy"]
                        ),
                        "candidate_solver_runtime_ms": float(
                            solver_times.loc[(wave_id, seed), candidate]
                        ),
                        "paired_solver_runtime_delta_ms": float(
                            solver_times.loc[(wave_id, seed), candidate]
                            - solver_times.loc[(wave_id, seed), "greedy"]
                        ),
                        "paired_solver_runtime_ratio": float(
                            solver_times.loc[(wave_id, seed), candidate]
                            / solver_times.loc[(wave_id, seed), "greedy"]
                        ),
                        "greedy_wall_process_ms": float(
                            wall_times.loc[(wave_id, seed), "greedy"]
                        ),
                        "candidate_wall_process_ms": float(
                            wall_times.loc[(wave_id, seed), candidate]
                        ),
                        "paired_wall_process_delta_ms": float(
                            wall_times.loc[(wave_id, seed), candidate]
                            - wall_times.loc[(wave_id, seed), "greedy"]
                        ),
                    }
                )
    paired = pd.DataFrame(raw_rows)

    summary_rows: list[dict[str, object]] = []
    for (wave_id, candidate), group in paired.groupby(
        ["wave_id", "candidate_label"], sort=False
    ):
        absolute = sample_summary(group["paired_gain_abs"])
        percentage = sample_summary(group["paired_gain_pct"])
        runtime_delta = sample_summary(group["paired_solver_runtime_delta_ms"])
        runtime_ratio = sample_summary(group["paired_solver_runtime_ratio"])
        wall_delta = sample_summary(group["paired_wall_process_delta_ms"])
        values = group["paired_gain_abs"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "wave_id": wave_id,
                "order_count": int(group["order_count"].iloc[0]),
                "candidate_label": candidate,
                "candidate_method": METHODS[candidate]["display_name"],
                "paired_seed_count": absolute["n"],
                "positive_gain_seed_count": int(np.sum(values > TOLERANCE)),
                "tied_seed_count": int(np.sum(np.abs(values) <= TOLERANCE)),
                "negative_gain_seed_count": int(np.sum(values < -TOLERANCE)),
                "gain_abs_mean": absolute["mean"],
                "gain_abs_sample_sd": absolute["sample_sd"],
                "gain_abs_ci95_low": absolute["ci95_low"],
                "gain_abs_ci95_high": absolute["ci95_high"],
                "gain_abs_best": absolute["maximum"],
                "gain_pct_mean": percentage["mean"],
                "gain_pct_sample_sd": percentage["sample_sd"],
                "gain_pct_ci95_low": percentage["ci95_low"],
                "gain_pct_ci95_high": percentage["ci95_high"],
                "raw_two_sided_t_p": two_sided_one_sample_t_pvalue(values),
                "raw_ci95_excludes_zero_positive": bool(absolute["ci95_low"] > 0.0),
                "solver_runtime_delta_ms_mean": runtime_delta["mean"],
                "solver_runtime_delta_ms_sample_sd": runtime_delta["sample_sd"],
                "solver_runtime_delta_ms_ci95_low": runtime_delta["ci95_low"],
                "solver_runtime_delta_ms_ci95_high": runtime_delta["ci95_high"],
                "solver_runtime_ratio_mean": runtime_ratio["mean"],
                "solver_runtime_ratio_sample_sd": runtime_ratio["sample_sd"],
                "solver_runtime_ratio_ci95_low": runtime_ratio["ci95_low"],
                "solver_runtime_ratio_ci95_high": runtime_ratio["ci95_high"],
                "wall_process_delta_ms_mean": wall_delta["mean"],
                "wall_process_delta_ms_sample_sd": wall_delta["sample_sd"],
                "wall_process_delta_ms_ci95_low": wall_delta["ci95_low"],
                "wall_process_delta_ms_ci95_high": wall_delta["ci95_high"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["holm_adjusted_p_12"] = np.nan
    summary["holm_reject_0_05"] = False
    for candidate in CANDIDATES:
        mask = summary["candidate_label"] == candidate
        adjusted = holm_adjust(summary.loc[mask, "raw_two_sided_t_p"])
        summary.loc[mask, "holm_adjusted_p_12"] = adjusted
        summary.loc[mask, "holm_reject_0_05"] = adjusted <= alpha

    candidate_order = {label: index for index, label in enumerate(CANDIDATES)}
    paired["_wave"] = paired["wave_id"].map(lambda value: wave_sort_key(str(value))[0])
    paired["_candidate"] = paired["candidate_label"].map(candidate_order)
    paired = paired.sort_values(
        ["_candidate", "_wave", "solver_seed"], kind="stable"
    ).drop(columns=["_wave", "_candidate"])
    summary["_wave"] = summary["wave_id"].map(lambda value: wave_sort_key(str(value))[0])
    summary["_candidate"] = summary["candidate_label"].map(candidate_order)
    summary = summary.sort_values(["_candidate", "_wave"], kind="stable").drop(
        columns=["_wave", "_candidate"]
    )
    return paired.reset_index(drop=True), summary.reset_index(drop=True)


def load_wave_features(feature_source: Path, paired_summary: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(feature_source)
    required = {"wave_id", "order_count", "overlap_density", "span_dispersion"}
    require_columns(frame, required, feature_source)

    for column in required.difference({"wave_id"}):
        uniqueness = frame.groupby("wave_id")[column].nunique(dropna=False)
        if (uniqueness != 1).any():
            bad = uniqueness[uniqueness != 1].index.tolist()
            raise ValueError(f"Feature {column} is inconsistent for waves: {bad}")
    features = frame.groupby("wave_id", as_index=False).agg(
        order_count=("order_count", "first"),
        overlap_density=("overlap_density", "first"),
        span_dispersion=("span_dispersion", "first"),
    )

    gains = paired_summary.pivot(
        index="wave_id", columns="candidate_label", values="gain_pct_mean"
    ).reset_index()
    gains = gains.rename(
        columns={
            "sa_i10000": "sa_i10000_gain_pct_mean",
            "ga_g40": "ga_g40_gain_pct_mean",
        }
    )
    merged = features.merge(gains, on="wave_id", how="inner", validate="one_to_one")
    expected_wave_set = set(paired_summary["wave_id"].astype(str).unique())
    if set(merged["wave_id"].astype(str)) != expected_wave_set:
        raise ValueError("Feature waves do not exactly match the Release study waves.")

    expected_orders = paired_summary.groupby("wave_id")["order_count"].first()
    actual_orders = merged.set_index("wave_id")["order_count"].astype(int)
    if not expected_orders.astype(int).sort_index().equals(actual_orders.sort_index()):
        raise ValueError("Feature-source order counts differ from the Release study.")
    merged["_wave"] = merged["wave_id"].map(lambda value: wave_sort_key(str(value))[0])
    return merged.sort_values("_wave", kind="stable").drop(columns="_wave").reset_index(
        drop=True
    )


def build_feature_associations(
    features_and_gains: pd.DataFrame, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    association_rows: list[dict[str, object]] = []
    leave_one_out_rows: list[dict[str, object]] = []

    for candidate in CANDIDATES:
        response = f"{candidate}_gain_pct_mean"
        for feature in FEATURES:
            rho, raw_p = spearmanr(features_and_gains[feature], features_and_gains[response])
            loo_values: list[tuple[str, float, float]] = []
            for omitted_wave in features_and_gains["wave_id"]:
                subset = features_and_gains[features_and_gains["wave_id"] != omitted_wave]
                loo_rho, loo_p = spearmanr(subset[feature], subset[response])
                loo_values.append((str(omitted_wave), float(loo_rho), float(loo_p)))
                leave_one_out_rows.append(
                    {
                        "candidate_label": candidate,
                        "candidate_method": METHODS[candidate]["display_name"],
                        "response": "per-wave mean paired percentage gain vs greedy",
                        "feature": feature,
                        "omitted_wave": omitted_wave,
                        "remaining_wave_count": len(subset),
                        "spearman_rho": float(loo_rho),
                        "raw_two_sided_p": float(loo_p),
                    }
                )
            influential = max(loo_values, key=lambda item: abs(item[1] - float(rho)))
            same_sign = [
                np.sign(value) == np.sign(float(rho)) and abs(value) > TOLERANCE
                for _, value, _ in loo_values
            ]
            association_rows.append(
                {
                    "candidate_label": candidate,
                    "candidate_method": METHODS[candidate]["display_name"],
                    "response": "per-wave mean paired percentage gain vs greedy",
                    "feature": feature,
                    "wave_count": len(features_and_gains),
                    "spearman_rho": float(rho),
                    "raw_two_sided_p": float(raw_p),
                    "loo_rho_min": min(value for _, value, _ in loo_values),
                    "loo_rho_median": float(np.median([value for _, value, _ in loo_values])),
                    "loo_rho_max": max(value for _, value, _ in loo_values),
                    "loo_same_sign_fraction": float(np.mean(same_sign)),
                    "most_influential_omitted_wave": influential[0],
                    "most_influential_loo_rho": influential[1],
                    "max_abs_loo_rho_change": abs(influential[1] - float(rho)),
                }
            )

    associations = pd.DataFrame(association_rows)
    associations["holm_adjusted_p_3"] = np.nan
    associations["holm_reject_0_05"] = False
    for candidate in CANDIDATES:
        mask = associations["candidate_label"] == candidate
        adjusted = holm_adjust(associations.loc[mask, "raw_two_sided_p"])
        associations.loc[mask, "holm_adjusted_p_3"] = adjusted
        associations.loc[mask, "holm_reject_0_05"] = adjusted <= alpha

    candidate_order = {label: index for index, label in enumerate(CANDIDATES)}
    feature_order = {feature: index for index, feature in enumerate(FEATURES)}
    associations["_candidate"] = associations["candidate_label"].map(candidate_order)
    associations["_feature"] = associations["feature"].map(feature_order)
    associations = associations.sort_values(
        ["_candidate", "_feature"], kind="stable"
    ).drop(columns=["_candidate", "_feature"])
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    leave_one_out["_candidate"] = leave_one_out["candidate_label"].map(candidate_order)
    leave_one_out["_feature"] = leave_one_out["feature"].map(feature_order)
    leave_one_out["_wave"] = leave_one_out["omitted_wave"].map(
        lambda value: wave_sort_key(str(value))[0]
    )
    leave_one_out = leave_one_out.sort_values(
        ["_candidate", "_feature", "_wave"], kind="stable"
    ).drop(columns=["_candidate", "_feature", "_wave"])
    return associations.reset_index(drop=True), leave_one_out.reset_index(drop=True)


def format_decimal(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_markdown_report(
    paired_summary: pd.DataFrame,
    associations: pd.DataFrame,
    selected_run_count: int,
    alpha: float,
) -> str:
    lines = [
        "# Per-wave-analyse van de Release-budgetstudie",
        "",
        "Deze analyse gebruikt uitsluitend bestaande solverresultaten. Er zijn geen nieuwe",
        "optimalisatieruns uitgevoerd.",
        "",
        "## Validatie en protocol",
        "",
        f"- Gevalideerde records: {selected_run_count} = 12 waves × 3 methoden × 10 seeds.",
        "- Alle geselecteerde runs zijn succesvol, niet getime-out en haalbaar.",
        "- Iedere SA-10000- en GA-40-run is gekoppeld aan greedy met dezelfde wave en seed.",
        "- Absolute winst is `D_greedy - D_kandidaat`; een positieve waarde is beter.",
        "- Procentuele winst wordt per gekoppelde seed berekend als",
        "  `100 * (D_greedy - D_kandidaat) / D_greedy`.",
        "- Gemiddelde, steekproef-SD en tweezijdig Student-t-95%-BI gebruiken de tien",
        "  gekoppelde seeds binnen één wave.",
        "- De tweezijdige t-toets gebruikt H0: gemiddelde gekoppelde winst = 0.",
        f"- Holm corrigeert per kandidaat over de twaalf wave-toetsen bij alpha={alpha:.2f}.",
        "- Waves zijn alleen de twaalf eenheden van de exploratieve Spearman-analyse en",
        "  worden niet als extra seedreplicaten behandeld.",
        "",
        "## Kwaliteitsbevindingen",
        "",
    ]

    for candidate in CANDIDATES:
        group = paired_summary[paired_summary["candidate_label"] == candidate]
        display = METHODS[candidate]["display_name"]
        nonzero = int((group["positive_gain_seed_count"] > 0).sum())
        raw_significant = int(group["raw_ci95_excludes_zero_positive"].sum())
        holm_significant = int(group["holm_reject_0_05"].sum())
        best_relative = group.loc[group["gain_pct_mean"].idxmax()]
        best_absolute = group.loc[group["gain_abs_mean"].idxmax()]
        lines.extend(
            [
                f"### {display}",
                "",
                f"- Minstens één seed verbetert in {nonzero}/12 waves.",
                f"- Een ruw tweezijdig 95%-BI sluit nul positief uit in {raw_significant}/12 waves.",
                f"- Na Holm-correctie blijven {holm_significant}/12 waves significant.",
                f"- Grootste gemiddelde relatieve winst: {best_relative['wave_id']} met "
                f"{format_decimal(float(best_relative['gain_pct_mean']), 6)}%.",
                f"- Grootste gemiddelde absolute winst: {best_absolute['wave_id']} met "
                f"{format_decimal(float(best_absolute['gain_abs_mean']), 3)} afstandseenheden.",
                "",
            ]
        )

    significant = paired_summary[paired_summary["holm_reject_0_05"]].copy()
    lines.extend(
        [
            "### Holm-significante waves",
            "",
            "| methode | wave | gem. winst | 95%-BI | gem. winst (%) | Holm-p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if significant.empty:
        lines.append("| geen | — | — | — | — | — |")
    else:
        for row in significant.itertuples(index=False):
            lines.append(
                f"| {row.candidate_method} | {row.wave_id} | "
                f"{format_decimal(row.gain_abs_mean, 3)} | "
                f"[{format_decimal(row.gain_abs_ci95_low, 3)}, "
                f"{format_decimal(row.gain_abs_ci95_high, 3)}] | "
                f"{format_decimal(row.gain_pct_mean, 6)} | "
                f"{format_decimal(row.holm_adjusted_p_12, 8)} |"
            )

    lines.extend(
        [
            "",
            "## Exploratieve wavekenmerken",
            "",
            "Spearman gebruikt de gemiddelde gekoppelde procentuele winst per wave. De drie",
            "vooraf bestaande kenmerken zijn orderaantal, gemiddelde paarsgewijze",
            "Jaccard-overlap en SD van de orderspans. Holm corrigeert per kandidaat over deze",
            "drie associaties. Leave-one-wave-out is een gevoeligheidsanalyse, geen BI.",
            "",
            "| methode | kenmerk | rho | ruwe p | Holm-p | LOO-bereik |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in associations.itertuples(index=False):
        lines.append(
            f"| {row.candidate_method} | {row.feature} | "
            f"{format_decimal(row.spearman_rho, 6)} | "
            f"{format_decimal(row.raw_two_sided_p, 8)} | "
            f"{format_decimal(row.holm_adjusted_p_3, 8)} | "
            f"[{format_decimal(row.loo_rho_min, 6)}, {format_decimal(row.loo_rho_max, 6)}] |"
        )

    significant_associations = associations[associations["holm_reject_0_05"]]
    lines.extend(["", "### Voorzichtige interpretatie", ""])
    if significant_associations.empty:
        lines.append(
            "Geen feature-associatie blijft significant na de vooraf bepaalde Holm-correctie."
        )
    else:
        for row in significant_associations.itertuples(index=False):
            lines.append(
                f"- {row.candidate_method}: `{row.feature}` heeft rho="
                f"{format_decimal(row.spearman_rho, 3)} (Holm-p="
                f"{format_decimal(row.holm_adjusted_p_3, 6)}); de LOO-rho blijft tussen "
                f"{format_decimal(row.loo_rho_min, 3)} en "
                f"{format_decimal(row.loo_rho_max, 3)}."
            )
    lines.extend(
        [
            "",
            "Met slechts twaalf waves, gebonden winstwaarden en meerdere nulwaarden zijn deze",
            "associaties beschrijvend. Ze bewijzen geen causaal wave-effect.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.15g", lineterminator="\n")


def main() -> None:
    args = parse_arguments()
    study_dir = args.study_dir.resolve()
    output_dir = args.output_dir.resolve()
    feature_source = args.feature_source.resolve()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie strictly between zero and one.")
    if not study_dir.is_dir():
        raise FileNotFoundError(study_dir)
    if not feature_source.is_file():
        raise FileNotFoundError(feature_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, run_files = load_and_validate_runs(study_dir)
    method_summary = build_method_summary(runs)
    paired_seed_differences, paired_comparisons = build_paired_tables(runs, args.alpha)
    features_and_gains = load_wave_features(feature_source, paired_comparisons)
    feature_associations, feature_leave_one_out = build_feature_associations(
        features_and_gains, args.alpha
    )

    frames = {
        "method_summary": method_summary,
        "paired_seed_differences": paired_seed_differences,
        "paired_comparisons": paired_comparisons,
        "features_and_gains": features_and_gains,
        "feature_associations": feature_associations,
        "feature_leave_one_out": feature_leave_one_out,
    }
    output_paths: dict[str, Path] = {}
    for key, frame in frames.items():
        path = output_dir / OUTPUT_FILENAMES[key]
        write_csv(frame, path)
        output_paths[key] = path

    report_path = output_dir / OUTPUT_FILENAMES["report"]
    report_path.write_text(
        build_markdown_report(
            paired_comparisons,
            feature_associations,
            selected_run_count=len(runs),
            alpha=args.alpha,
        ),
        encoding="utf-8",
        newline="\n",
    )
    output_paths["report"] = report_path

    protocol = {
        "schema_version": 1,
        "state": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "analysis_scope": "per-wave greedy, SA-10000 and GA-40 Release performance",
        "solver_execution": "none; existing runs.csv files only",
        "script": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            "study_dir": str(study_dir),
            "wave_run_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in run_files
            ],
            "feature_source": {
                "path": str(feature_source),
                "sha256": sha256_file(feature_source),
                "use": (
                    "deterministic overlap_density and span_dispersion values only; "
                    "order counts are cross-validated against the Release runs"
                ),
            },
        },
        "validation": {
            "selected_run_count": len(runs),
            "wave_count": int(runs["wave_id"].nunique()),
            "method_count": int(runs["experiment_label"].nunique()),
            "seeds_per_wave_method": len(EXPECTED_SEEDS),
            "expected_seed_labels": list(EXPECTED_SEEDS),
            "all_successful": bool((runs["status"] == "success").all()),
            "all_feasible": bool(
                runs["feasible"].astype(str).str.lower().isin({"true", "1"}).all()
            ),
            "all_not_timed_out": bool(
                ~runs["timed_out"].astype(str).str.lower().isin({"true", "1"}).any()
            ),
        },
        "quality_statistics": {
            "pairing": "same wave_id and solver_seed",
            "absolute_gain": "D_greedy - D_candidate; positive is better",
            "percentage_gain": "100 * (D_greedy - D_candidate) / D_greedy, per paired seed",
            "mean": "arithmetic mean over the ten paired solver seeds within a wave",
            "dispersion": "sample standard deviation over paired solver seeds (ddof=1)",
            "confidence_interval": (
                "two-sided Student-t 95% CI: mean +/- t_(0.975,n-1) * sample_sd / sqrt(n)"
            ),
            "hypothesis_test": (
                "two-sided one-sample Student-t test of paired absolute gains against mean zero"
            ),
            "holm": (
                "Holm step-down adjusted p-values, separately for each candidate's "
                "family of twelve wave tests"
            ),
            "familywise_alpha": args.alpha,
        },
        "feature_statistics": {
            "response": "per-wave arithmetic mean of paired percentage gains",
            "features": list(FEATURES),
            "overlap_density_definition": (
                "mean pairwise Jaccard similarity of unique pick-location sets across orders"
            ),
            "span_dispersion_definition": (
                "population SD across orders of each order's width between extreme snake ranks"
            ),
            "association": "Spearman rank correlation over twelve waves, two-sided p-value",
            "holm": (
                "Holm step-down adjusted p-values, separately for each candidate's "
                "family of three pre-specified features"
            ),
            "sensitivity": (
                "leave-one-wave-out Spearman rho; descriptive stability check, not a CI"
            ),
            "caveat": (
                "n=12 waves, tied/zero gains and observational features; associations are "
                "exploratory and do not establish causality"
            ),
        },
        "outputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in output_paths.items()
        },
    }
    protocol_path = output_dir / OUTPUT_FILENAMES["protocol"]
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Validated selected runs: {len(runs)}")
    print(f"Wrote {len(output_paths) + 1} outputs to {output_dir}")


if __name__ == "__main__":
    main()
