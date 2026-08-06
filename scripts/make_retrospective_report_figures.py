"""Maak compacte figuren voor het retrospectieve Optioryx-verslag.

De invoer bestaat uitsluitend uit de canonieke resultatenmappen.  Alle
onzekerheid voor stochastische methoden is reeds berekend op het niveau van een
volledige instantie per seed. Waves worden dus niet als onafhankelijke
replicaten behandeld.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

COLORS = {
    "greedy": "#2ca02c",
    "simulated_annealing": "#ff7f0e",
    "genetic": "#1f77b4",
    "seed_selection": "#d62728",
}


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "figure.titlesize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sa_ga_figure() -> None:
    sa = read_csv(
        "results/sa_factorial/factorial_analysis/sa_config_summary.csv"
    )
    ga = read_csv(
        "results/ga_sensitivity_fast/factorial_analysis/ga_sensitivity_summary.csv"
    )
    diversity = read_csv(
        "results/ga_sensitivity_fast/analysis/control_summary.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)

    order = [
        ("geometric", "relocate", "Geo/verplaats"),
        ("lundy_mees", "relocate", "LM/verplaats"),
        ("geometric", "swap", "Geo/wissel"),
        ("lundy_mees", "swap", "LM/wissel"),
    ]
    for ax, start, title in zip(
        axes[0],
        ["greedy", "randomized"],
        ["(a) SA met sterke greedy-start", "(b) SA met willekeurige start"],
    ):
        rows = []
        labels = []
        for cooling, neighborhood, label in order:
            row = sa[
                (sa["start_strategy"] == start)
                & (sa["cooling"] == cooling)
                & (sa["neighborhood"] == neighborhood)
            ].iloc[0]
            rows.append(row)
            labels.append(label)
        frame = pd.DataFrame(rows).reset_index(drop=True)
        x = np.arange(len(frame))
        ax.errorbar(
            x,
            frame["objective_mean"] / 1e6,
            yerr=frame["objective_ci95"] / 1e6,
            fmt="o",
            color=COLORS["simulated_annealing"],
            capsize=3,
            label=r"gemiddelde $\pm$ 95%-BI",
        )
        ax.scatter(
            x,
            frame["objective_best"] / 1e6,
            marker="*",
            s=42,
            color="#8c2d04",
            label="beste run",
            zorder=3,
        )
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_ylabel("doelfunctie ($10^6$, lager is beter)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        note_y = 0.03 if start == "greedy" else 0.97
        ax.text(
            0.02,
            note_y,
            "10 seeds/config., 400 iteraties",
            transform=ax.transAxes,
            fontsize=6.7,
            va="bottom" if start == "greedy" else "top",
        )
    axes[0, 0].legend(loc="upper right", frameon=False)

    variants = ga[ga["changed_factor"] != "baseline"].copy()
    factor_order = [
        ("elitism_fraction", 0.0, "elitisme 0"),
        ("elitism_fraction", 0.20, "elitisme 0,20"),
        ("generations", 10.0, "10 generaties"),
        ("generations", 40.0, "40 generaties"),
        ("mutation_rate", 0.05, "mutatie 0,05"),
        ("mutation_rate", 0.40, "mutatie 0,40"),
        ("population_size", 24.0, "populatie 24"),
        ("population_size", 6.0, "populatie 6"),
    ]
    selected = []
    labels = []
    for factor, value, label in factor_order:
        factor_rows = variants[variants["changed_factor"] == factor]
        numeric_values = pd.to_numeric(factor_rows["variant_value"])
        row = factor_rows[np.isclose(numeric_values, value)].iloc[0]
        selected.append(row)
        labels.append(label)
    frame = pd.DataFrame(selected).reset_index(drop=True)
    y = np.arange(len(frame))
    axes[1, 0].errorbar(
        frame["objective_delta_mean"],
        y,
        xerr=frame["objective_delta_ci95"],
        fmt="o",
        color=COLORS["genetic"],
        capsize=2,
    )
    axes[1, 0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_yticks(y, labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("variant $-$ baseline (afstand, lager is beter)")
    axes[1, 0].set_title("(c) GA: gekoppelde parameter-effecten")
    axes[1, 0].grid(axis="x", alpha=0.25)

    baseline_id = (
        "ga_pop-12_gen-20_tour-3_mut-0p20_insert-0p50_elite-0p08"
    )
    diversity_configs = {
        baseline_id: "basis",
        "ga_pop-12_gen-20_tour-3_mut-0p05_insert-0p50_elite-0p08": "mut. 0,05",
        "ga_pop-12_gen-20_tour-3_mut-0p40_insert-0p50_elite-0p08": "mut. 0,40",
        "ga_pop-12_gen-20_tour-3_mut-0p20_insert-0p50_elite-0": "geen elitisme",
        "ga_pop-24_gen-20_tour-3_mut-0p20_insert-0p50_elite-0p08": "pop. 24",
    }
    for config_id, label in diversity_configs.items():
        part = diversity[diversity["config_id"] == config_id].sort_values("step")
        axes[1, 1].plot(
            part["step"],
            part["mean_control_value"],
            linewidth=1.4,
            label=label,
        )
    axes[1, 1].set_xlabel("generatie")
    axes[1, 1].set_ylabel("gem. genormaliseerde Hammingdiversiteit")
    axes[1, 1].set_ylim(0, 1.03)
    axes[1, 1].set_title("(d) GA: populatie wordt snel homogeen")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(ncol=2, frameon=False, loc="upper right")

    save(fig, "retrospective_sa_ga")


def comparison_figure() -> None:
    scaling = read_csv("results/scaling_controlled_matched/summary.csv")
    fits = read_csv("results/scaling_controlled_matched/fits.csv")
    convergence = read_csv(
        "results/comparison_fast_matched/analysis/convergence_summary.csv"
    )

    # Behoud dezelfde totale canvasmaat als de vroegere driepanelenfiguur. Zo
    # krijgen schaalfit en convergentie elk meer horizontale ruimte zonder dat
    # de ingevoegde figuur in het verslag hoger wordt.
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.35), constrained_layout=True)

    names = {
        "simulated_annealing": "SA",
        "genetic": "GA",
    }

    label_map = {
        "greedy_matched": ("greedy", "greedy"),
        "sa_matched": ("SA", "simulated_annealing"),
        "ga_matched": ("GA", "genetic"),
    }
    for experiment, (label, solver) in label_map.items():
        part = scaling[scaling["experiment_label"] == experiment].sort_values("n_orders")
        fit = fits[fits["experiment_label"] == experiment].iloc[0]
        x = part["n_orders"].to_numpy(dtype=float)
        med = part["solver_runtime_ms_median"].to_numpy(dtype=float)
        axes[0].loglog(
            x,
            med,
            "o-",
            color=COLORS[solver],
            linewidth=1.3,
            markersize=3.5,
            label=f"{label}: b={fit['empirical_exponent_b']:.2f}",
        )
        axes[0].fill_between(
            x,
            part["solver_runtime_ms_q1"],
            part["solver_runtime_ms_q3"],
            color=COLORS[solver],
            alpha=0.12,
            linewidth=0,
        )
    axes[0].set_xlabel("orders $n$ (geneste subsets)")
    axes[0].set_ylabel("mediane solvertijd (ms)")
    axes[0].set_title("(a) Empirische schaalfit")
    axes[0].grid(which="both", alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")

    for solver in ["genetic", "simulated_annealing"]:
        part = convergence[convergence["solver"] == solver].sort_values("step")
        start = float(part.iloc[0]["mean_best_objective"])
        gain = 100.0 * (start - part["mean_best_objective"]) / start
        ci = 100.0 * part["ci95_best_objective"] / start
        axes[1].plot(
            part["step"],
            gain,
            color=COLORS[solver],
            linewidth=1.4,
            label=names[solver],
        )
        axes[1].fill_between(
            part["step"],
            gain - ci,
            gain + ci,
            color=COLORS[solver],
            alpha=0.14,
            linewidth=0,
        )
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("iteratie / generatie")
    axes[1].set_ylabel("verbetering t.o.v. eigen start (%)")
    axes[1].set_title("(b) Convergentie, 10 seeds")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")

    save(fig, "retrospective_comparison")


def waves_figure() -> None:
    gains = read_csv(
        "results/comparison_fast_matched/analysis/relative_gains.csv"
    )
    correlations = read_csv(
        "results/comparison_fast_matched/analysis/correlations.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.25), constrained_layout=True)
    for solver, label in [("genetic", "GA"), ("simulated_annealing", "SA")]:
        part = gains[gains["solver"] == solver].sort_values("wave_id")
        x = np.arange(1, len(part) + 1)
        axes[0].errorbar(
            x,
            100.0 * part["relative_gain_vs_greedy"],
            yerr=100.0 * part["relative_gain_ci95"].fillna(0.0),
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            capsize=2,
            color=COLORS[solver],
            label=label,
        )
    axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_xticks(np.arange(1, 13))
    axes[0].set_xlabel("wave")
    axes[0].set_ylabel("verbetering t.o.v. greedy (%)")
    axes[0].set_title("(a) Verschillen tussen twaalf waves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    ga = gains[gains["solver"] == "genetic"].copy()
    x = ga["span_dispersion"].to_numpy(dtype=float)
    y = 100.0 * ga["relative_gain_vs_greedy"].to_numpy(dtype=float)
    colors = ga["overlap_density"].to_numpy(dtype=float)
    scatter = axes[1].scatter(
        x,
        y,
        c=colors,
        cmap="viridis_r",
        s=34,
        edgecolor="white",
        linewidth=0.5,
    )
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        axes[1].plot(xx, slope * xx + intercept, color="#555555", linewidth=1.0)
    # Bij de gematchte start vallen veel GA-winsten exact samen op nul. Vaste
    # verticale offsets houden de wave-ID's in die dichte cluster leesbaar.
    label_offsets = {
        "01": (0, 4),
        "02": (3, -28),
        "03": (3, 12),
        "04": (0, 14),
        "05": (-3, -28),
        "06": (2, 14),
        "07": (2, -15),
        "08": (-5, 4),
        "09": (3, 4),
        "10": (3, 4),
        "11": (-10, -15),
        "12": (3, 4),
    }
    for _, row in ga.iterrows():
        wave_key = str(row["wave_id"])[-2:]
        wave_label = str(int(wave_key))
        axes[1].annotate(
            wave_label,
            (row["span_dispersion"], 100.0 * row["relative_gain_vs_greedy"]),
            xytext=label_offsets[wave_key],
            textcoords="offset points",
            fontsize=5.8,
        )
    corr = correlations[correlations["solver"] == "genetic"].iloc[0]
    axes[1].set_xlabel("spreiding van snake-spans")
    axes[1].set_ylabel("GA-winst t.o.v. greedy (%)")
    axes[1].set_ylim(bottom=-0.020)
    axes[1].set_title(
        f"(b) Eigen analyse: $r_{{span}}$={corr['corr_span_gain']:.2f}, "
        f"$r_{{overlap}}$={corr['corr_overlap_gain']:.2f}"
    )
    axes[1].grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=axes[1], pad=0.02)
    cbar.set_label("overlapdichtheid")

    save(fig, "retrospective_waves")


def seed_figure() -> None:
    paper = read_csv(
        "results/seed_factorial/factorial_analysis/seed_paper_2x2.csv"
    )
    custom = read_csv(
        "results/seed_factorial/factorial_analysis/seed_custom_ablation.csv"
    )
    summary = read_csv("results/seed_factorial/analysis/summary.csv")
    matched = read_csv(
        "results/comparison_fast_matched/analysis/summary.csv"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.35),
        gridspec_kw={"width_ratios": [1.05, 0.75, 1.2]},
        constrained_layout=True,
    )

    seeds = ["largest_snake_span", "smallest_snake_span"]
    selections = ["maximum_location_overlap", "minimum_span_expansion"]
    matrix = np.empty((2, 2))
    for i, seed in enumerate(seeds):
        for j, selection in enumerate(selections):
            matrix[i, j] = paper[
                (paper["seed_rule"] == seed)
                & (paper["selection_rule"] == selection)
            ].iloc[0]["objective_mean"] / 1e6
    im = axes[0].imshow(matrix, cmap="viridis_r", aspect="auto")
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 7.2 else "black",
                fontsize=8,
            )
    axes[0].set_xticks([0, 1], ["max. overlap", "min. span-\nuitbreiding"])
    axes[0].set_yticks([0, 1], ["grootste span", "kleinste span"])
    axes[0].set_title("(a) Paperregels ($10^6$)")
    fig.colorbar(im, ax=axes[0], pad=0.02, shrink=0.86)

    custom_order = ["largest_snake_span", "smallest_snake_span"]
    rows = [custom[custom["seed_rule"] == seed].iloc[0] for seed in custom_order]
    frame = pd.DataFrame(rows)
    improvement = -frame["objective_delta_percent_mean"].to_numpy(dtype=float)
    runtime_ratio = (
        frame["custom_runtime_ms_mean"].to_numpy(dtype=float)
        / frame["immediate_runtime_ms_mean"].to_numpy(dtype=float)
    )
    bars = axes[1].bar(
        [0, 1],
        improvement,
        color=["#8c6d31", "#e7ba52"],
        width=0.62,
    )
    axes[1].set_xticks([0, 1], ["grootste\nspan", "kleinste\nspan"])
    axes[1].set_ylabel("winst door lookahead (%)")
    axes[1].set_title("(b) Ablatie eigen regel")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, ratio in zip(bars, runtime_ratio):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            rf"tijd $\times${ratio:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    def selection_label(config_id: str) -> str:
        if "maximum_location_overlap" in config_id:
            return "max. overlap"
        if "minimum_span_expansion" in config_id:
            return "min. spanuitbreiding"
        if "minimum_incremental_cost" in config_id:
            return "directe kost"
        return "tweestaps-lookahead"

    marker_by_seed = {
        "largest_snake_span": "o",
        "smallest_snake_span": "s",
    }
    color_by_rule = {
        "max. overlap": "#9467bd",
        "min. spanuitbreiding": "#17becf",
        "directe kost": "#7f7f7f",
        "tweestaps-lookahead": "#d62728",
    }
    used = set()
    for _, row in summary.iterrows():
        config_id = str(row["config_id"])
        seed = "smallest_snake_span" if "smallest_snake_span" in config_id else "largest_snake_span"
        rule = selection_label(config_id)
        label = rule if rule not in used else None
        used.add(rule)
        axes[2].scatter(
            row["mean_runtime_ms"] / 1000.0,
            row["mean_objective"] / 1e6,
            marker=marker_by_seed[seed],
            color=color_by_rule[rule],
            s=30,
            label=label,
        )
    greedy_matched = matched[matched["solver"] == "greedy"].iloc[0]
    axes[2].scatter(
        greedy_matched["mean_runtime_ms"] / 1000.0,
        greedy_matched["mean_objective"] / 1e6,
        marker="*",
        color=COLORS["greedy"],
        s=55,
        label="greedy (gematcht)",
        zorder=4,
    )
    axes[2].set_xlabel("uitvoeringstijd (s)")
    axes[2].set_ylabel("doelfunctie ($10^6$)")
    axes[2].set_title("(c) Kwaliteit--tijd")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=5.8, loc="upper right")
    axes[2].text(
        0.02,
        0.02,
        "○ grootste span, □ kleinste span",
        transform=axes[2].transAxes,
        fontsize=5.8,
    )

    save(fig, "retrospective_seed")


def main() -> None:
    setup_style()
    sa_ga_figure()
    comparison_figure()
    waves_figure()
    seed_figure()


if __name__ == "__main__":
    main()
