"""Plots based on complete-instance seed replicates.

The raw benchmark files contain one row per wave.  All uncertainty bands in
this module are therefore computed after reconstructing one full-instance
observation per seed; waves are never used as statistical replicates.
"""

import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Works both as ``python scripts/plot_results.py`` and as a module import.
    from aggregate_results import (
        _ci95_half_width,
        build_config_summary,
        build_convergence_seed_totals,
        build_convergence_summary,
        build_seed_totals,
        build_wave_summary,
        compute_gain_table,
    )
except ImportError:  # pragma: no cover - depends on how the script is invoked
    from scripts.aggregate_results import (
        _ci95_half_width,
        build_config_summary,
        build_convergence_seed_totals,
        build_convergence_summary,
        build_seed_totals,
        build_wave_summary,
        compute_gain_table,
    )


def _errors(frame: pd.DataFrame, column: str) -> pd.Series:
    """Matplotlib wants finite errors; a one-seed CI is intentionally absent."""

    return frame[column].fillna(0.0).clip(lower=0.0)


def _nonnegative_errors(
    frame: pd.DataFrame, mean_column: str, error_column: str
) -> list[list[float]]:
    """Use an asymmetric bar when a symmetric CI would imply negatives."""

    upper = _errors(frame, error_column)
    lower = upper.clip(upper=frame[mean_column].clip(lower=0.0))
    return [lower.tolist(), upper.tolist()]


def _short_config_label(solver: str, config_id: str) -> str:
    """Keep explicit config IDs in CSV while making dense figures readable."""

    if solver == "greedy" and config_id.startswith("greedy_"):
        return "baseline"
    if solver == "simulated_annealing":
        match = re.search(
            r"(?:cool-(.*?)_move-|c-(.*?)_n-)"
            r"(.*?)_start-(.*?)_(?:iter-|i-)(.*?)(?:_t0-|$)",
            config_id,
        )
        if match:
            long_cooling, short_cooling, move, start, iterations = match.groups()
            cooling = long_cooling or short_cooling
            cooling = {
                "geometric": "geometrisch",
                "lundy_mees": "Lundy-Mees",
            }.get(cooling, cooling)
            move = {"relocate": "verplaats", "swap": "wissel"}.get(move, move)
            if start == "randomized":
                start = "willekeurig"
            elif start.startswith("greedy-ls-"):
                start = start.replace("greedy-ls-", "greedy LS").replace("-win-", "/w")
            return f"{cooling} / {move} / {start} ({iterations} it.)"
    if solver == "genetic":
        match = re.search(
            r"pop-(.*?)_gen-(.*?)_tour-.*?_mut-(.*?)_insert-.*?_elite-(.*)$",
            config_id,
        )
        if match:
            population, generations, mutation, elitism = match.groups()
            return f"pop {population}, gen {generations}, mut {mutation}, elite {elitism}"
    if solver == "seed_selection":
        match = re.search(r"seed_seed-(.*?)_select-(.*)$", config_id)
        if match:
            seed_rule, selection_rule = match.groups()
            replacements = {
                "highest_standalone_cost": "hoogste losse kost",
                "smallest_snake_span": "kleinste span",
                "largest_snake_span": "grootste span",
                "minimum_incremental_cost": "min. meerkost",
                "minimum_span_expansion": "min. spanuitbreiding",
                "maximum_location_overlap": "max. locatie-overlap",
            }
            seed_rule = replacements.get(seed_rule, seed_rule)
            selection_rule = replacements.get(selection_rule, selection_rule)
            if selection_rule.startswith("hybrid_score"):
                selection_rule = "hybride kost/span"
            elif selection_rule.startswith("bounded_two_step_lookahead"):
                selection_rule = "begrensde tweestaps-lookahead"
            return f"{seed_rule} / {selection_rule}"
    if len(config_id) <= 42:
        return config_id
    return f"{config_id[:19]}...{config_id[-19:]}"


def _solver_label(solver: str) -> str:
    return {
        "genetic": "genetisch algoritme",
        "simulated_annealing": "simulated annealing",
        "greedy": "greedy",
        "seed_selection": "seedselectie",
    }.get(solver, solver)


def _series_label(solver: str, config_id: str) -> str:
    return f"{_solver_label(solver)}: {_short_config_label(solver, config_id)}"


def _save_figure(figure: plt.Figure, png_path: Path) -> None:
    """Write a preview PNG and a vector PDF suitable for the report."""

    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")


def save_runtime_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    summary = build_config_summary(build_seed_totals(runs)).sort_values(
        "mean_runtime_ms", kind="stable"
    )
    labels = [
        _series_label(str(row.solver), str(row.config_id))
        for row in summary.itertuples()
    ]
    colors = {
        solver: plt.get_cmap("tab10")(index)
        for index, solver in enumerate(summary["solver"].drop_duplicates())
    }

    figure, axis = plt.subplots(figsize=(max(10, 0.55 * len(summary)), 5.5))
    positions = list(range(len(summary)))
    axis.bar(
        positions,
        summary["mean_runtime_ms"],
        yerr=_nonnegative_errors(summary, "mean_runtime_ms", "runtime_ci95_ms"),
        capsize=3,
        color=[colors[solver] for solver in summary["solver"]],
        alpha=0.85,
    )
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.set_ylabel("Gemiddelde uitvoeringstijd volledige instantie (ms)")
    axis.set_title("Uitvoeringstijd per configuratie (95%-BI over seeds)")
    figure.tight_layout()
    _save_figure(figure, output_dir / "runtime.png")
    plt.close(figure)


def save_quality_runtime_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    summary = build_config_summary(build_seed_totals(runs))
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for solver, frame in summary.groupby("solver", sort=False):
        axis.errorbar(
            frame["mean_runtime_ms"],
            frame["mean_objective"],
            xerr=_nonnegative_errors(frame, "mean_runtime_ms", "runtime_ci95_ms"),
            yerr=_errors(frame, "objective_ci95"),
            fmt="o",
            capsize=3,
            markersize=7,
            alpha=0.85,
            label=_solver_label(str(solver)),
        )
        for row in frame.itertuples():
            annotation_offset = {
                "greedy": (4, -14),
                "simulated_annealing": (4, 8),
            }.get(str(solver), (4, 3))
            axis.annotate(
                _short_config_label(str(solver), str(row.config_id)),
                (row.mean_runtime_ms, row.mean_objective),
                fontsize=7,
                xytext=annotation_offset,
                textcoords="offset points",
            )
    axis.set_xlabel("Gemiddelde uitvoeringstijd volledige instantie (ms)")
    axis.set_ylabel("Gemiddelde doelwaarde (lager is beter)")
    axis.set_title("Rekentijd versus kwaliteit (95%-BI over seeds)")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_dir / "quality_vs_runtime.png")
    plt.close(figure)


def _full_convergence_summary(convergence: pd.DataFrame) -> pd.DataFrame:
    if convergence.empty:
        return pd.DataFrame()
    return build_convergence_summary(build_convergence_seed_totals(convergence))


def save_convergence_plot(convergence: pd.DataFrame, output_dir: Path) -> None:
    summary = _full_convergence_summary(convergence)
    if summary.empty:
        return

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for (solver, config_id), frame in summary.groupby(
        ["solver", "config_id"], sort=False
    ):
        frame = frame.sort_values("step")
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean_best_objective"].to_numpy(dtype=float)
        ci = _errors(frame, "ci95_best_objective").to_numpy(dtype=float)
        line = axis.plot(
            steps,
            mean,
            label=_series_label(str(solver), str(config_id)),
        )[0]
        if (ci > 0).any():
            axis.fill_between(
                steps,
                mean - ci,
                mean + ci,
                color=line.get_color(),
                alpha=0.16,
                linewidth=0,
            )
    axis.set_xlabel("Iteratie / generatie")
    axis.set_ylabel("Gemiddelde beste doelwaarde over de volledige instantie")
    axis.set_title("Convergentie over alle waves (95%-BI over seeds)")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.legend(fontsize=7)
    figure.tight_layout()
    _save_figure(figure, output_dir / "convergence.png")
    plt.close(figure)

    seed_totals = build_convergence_seed_totals(convergence)
    start_values = (
        seed_totals.sort_values("step", kind="stable")
        .groupby(["solver", "config_id", "seed"], as_index=False, dropna=False)
        .first()[["solver", "config_id", "seed", "total_best_objective"]]
        .rename(columns={"total_best_objective": "start_objective"})
    )
    relative = seed_totals.merge(
        start_values,
        on=["solver", "config_id", "seed"],
        validate="many_to_one",
    )
    relative["improvement_percent"] = 100.0 * (
        relative["start_objective"] - relative["total_best_objective"]
    ) / relative["start_objective"]
    relative_summary = (
        relative.groupby(["solver", "config_id", "step"], as_index=False, dropna=False)
        .agg(
            mean_improvement_percent=("improvement_percent", "mean"),
            std_improvement_percent=("improvement_percent", "std"),
            seed_count=("seed", "nunique"),
        )
    )
    relative_summary["ci95_improvement_percent"] = _ci95_half_width(
        relative_summary["std_improvement_percent"],
        relative_summary["seed_count"],
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for (solver, config_id), frame in relative_summary.groupby(
        ["solver", "config_id"], sort=False
    ):
        frame = frame.sort_values("step")
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean_improvement_percent"].to_numpy(dtype=float)
        ci = frame["ci95_improvement_percent"].fillna(0.0).to_numpy(dtype=float)
        line = axis.plot(
            steps,
            mean,
            label=_series_label(str(solver), str(config_id)),
        )[0]
        if (ci > 0).any():
            axis.fill_between(
                steps,
                mean - ci,
                mean + ci,
                color=line.get_color(),
                alpha=0.16,
                linewidth=0,
            )
    axis.set_xlabel("Iteratie / generatie")
    axis.set_ylabel("Verbetering tegenover de eigen start (%)")
    axis.set_title("Relatieve convergentie over alle waves (95%-BI over seeds)")
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.legend(fontsize=7)
    figure.tight_layout()
    _save_figure(figure, output_dir / "convergence_relative.png")
    plt.close(figure)


def _save_control_plot(
    summary: pd.DataFrame,
    output_path: Path,
    solver: str,
    ylabel: str,
    title: str,
) -> None:
    selected = summary[summary["solver"] == solver]
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for config_id, frame in selected.groupby("config_id", sort=False):
        frame = frame.sort_values("step")
        steps = frame["step"].to_numpy(dtype=float)
        mean = frame["mean_control_value"].to_numpy(dtype=float)
        ci = _errors(frame, "ci95_control_value").to_numpy(dtype=float)
        line = axis.plot(
            steps, mean, label=_short_config_label(solver, str(config_id))
        )[0]
        if (ci > 0).any():
            lower = mean - ci
            upper = mean + ci
            if solver == "genetic":
                # Normalized Hamming diversity is bounded by construction.
                lower = lower.clip(min=0.0)
                upper = upper.clip(max=1.0)
            axis.fill_between(
                steps,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.16,
                linewidth=0,
            )
    axis.set_xlabel("Iteratie / generatie")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if solver == "genetic":
        axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)


def save_control_plots(convergence: pd.DataFrame, output_dir: Path) -> None:
    """Keep non-additive SA temperature and GA diversity off the score plot."""

    summary = _full_convergence_summary(convergence)
    if summary.empty:
        return
    _save_control_plot(
        summary,
        output_dir / "ga_diversity.png",
        "genetic",
        "Gemiddelde populatiediversiteit over waves",
        "Diversiteit van het genetisch algoritme (95%-BI over seeds)",
    )
    _save_control_plot(
        summary,
        output_dir / "sa_temperature.png",
        "simulated_annealing",
        "Gemiddelde temperatuur over waves",
        "Temperatuurschema van simulated annealing",
    )


def _best_config_per_solver(runs: pd.DataFrame) -> dict[str, str]:
    config_summary = build_config_summary(build_seed_totals(runs))
    best = (
        config_summary.sort_values(["mean_objective", "mean_runtime_ms"], kind="stable")
        .groupby("solver", sort=False, as_index=False)
        .first()
    )
    return {
        str(row.solver): str(row.config_id)
        for row in best.itertuples()
    }


def save_wave_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    """Plot normalized gains for the best full-instance config of each solver."""

    summary = build_wave_summary(runs)
    chosen = _best_config_per_solver(runs)
    selected = summary[
        summary.apply(
            lambda row: str(row["config_id"]) == chosen.get(str(row["solver"])),
            axis=1,
        )
    ].copy()
    if selected.empty or selected["greedy_objective"].isna().all():
        return

    # Preserve the instance order from runs.csv (e.g. wave_2 before wave_10),
    # instead of relying on lexicographic sorting of identifiers.
    all_wave_order = list(dict.fromkeys(runs["wave_id"].astype(str)))
    present_waves = set(selected["wave_id"].astype(str))
    wave_order = [wave_id for wave_id in all_wave_order if wave_id in present_waves]
    positions = {wave_id: index for index, wave_id in enumerate(wave_order)}
    figure, axis = plt.subplots(figsize=(max(9, 0.5 * len(wave_order)), 5.5))
    plotted = False
    for (solver, config_id), frame in selected.groupby(
        ["solver", "config_id"], sort=False
    ):
        if solver == "greedy":
            continue
        frame = frame.assign(
            wave_position=frame["wave_id"].astype(str).map(positions)
        ).sort_values("wave_position")
        axis.errorbar(
            frame["wave_position"],
            100.0 * frame["relative_gain_vs_greedy"],
            yerr=100.0 * _errors(frame, "relative_gain_ci95"),
            marker="o",
            capsize=3,
            linewidth=1.4,
            label=_series_label(str(solver), str(config_id)),
        )
        plotted = True
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1, label="greedy-baseline")
    axis.set_xticks(range(len(wave_order)), wave_order, rotation=45, ha="right")
    axis.set_ylabel("Verbetering van de doelwaarde t.o.v. greedy (%)")
    axis.set_title("Prestatie per wave, genormaliseerd t.o.v. greedy")
    if plotted:
        axis.legend(fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output_dir / "wave_by_wave.png")
    plt.close(figure)


def save_extra_plot(runs: pd.DataFrame, output_dir: Path) -> None:
    # One marker is one wave/config after averaging its seeds.  This prevents
    # the ten seeds of the same wave being shown as ten independent data points.
    gains = compute_gain_table(runs)
    gains = gains[gains["solver"].isin(["simulated_annealing", "genetic"])].copy()
    gains = gains.dropna(subset=["overlap_density", "relative_gain_vs_greedy"])
    if gains.empty:
        return

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for solver, frame in gains.groupby("solver", sort=False):
        axis.errorbar(
            frame["overlap_density"],
            frame["relative_gain_vs_greedy"],
            yerr=_errors(frame, "relative_gain_ci95"),
            fmt="o",
            capsize=2,
            alpha=0.75,
            label=solver,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Overlapdichtheid van de wave")
    axis.set_ylabel("Relatieve verbetering t.o.v. greedy")
    axis.set_title("Overlapdichtheid versus verbetering (gemiddelden per wave/configuratie)")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_dir / "extra_analysis.png")
    plt.close(figure)


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
    save_control_plots(convergence, output_dir)
    save_wave_plot(runs, output_dir)
    save_extra_plot(runs, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
