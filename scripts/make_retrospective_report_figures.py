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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


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


def decimal_comma(value: float, decimals: int) -> str:
    """Formatteer een vaste precisie met de Nederlandse decimale komma."""

    return f"{value:.{decimals}f}".replace(".", ",")


def apply_decimal_comma(
    ax: plt.Axes,
    *,
    x_axis: bool = False,
    y_axis: bool = False,
) -> None:
    """Gebruik een decimale komma op gewone lineaire assen.

    Logaritmische assen behouden hun gebruikelijke macht-van-tiennotatie.
    """

    formatter = lambda value, _position: f"{value:g}".replace(".", ",")
    if x_axis:
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    if y_axis:
        ax.yaxis.set_major_formatter(FuncFormatter(formatter))


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
        "results/sa_factorial_capacity_matched_release_20260807/"
        "factorial_analysis/sa_config_summary.csv"
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
        ["(a) SA met sterke greedy-start", "(b) SA met capaciteitsgelijke randomstart"],
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
            label="beste uitvoering",
            zorder=3,
        )
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_ylabel("doelfunctie ($10^6$, lager is beter)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        apply_decimal_comma(ax, y_axis=True)
        note_y = 0.03 if start == "greedy" else 0.97
        ax.text(
            0.02,
            note_y,
            (
                "10 seeds, 400 iteraties"
                if start == "greedy"
                else "10 seeds, 400 iteraties, 517 startroutes"
            ),
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
    axes[1, 0].set_xlabel(
        "variant $-$ basisconfiguratie (afstand, lager is beter)"
    )
    axes[1, 0].set_title("(c) GA: gekoppelde parameter-effecten")
    axes[1, 0].grid(axis="x", alpha=0.25)
    apply_decimal_comma(axes[1, 0], x_axis=True)

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
    apply_decimal_comma(axes[1, 1], x_axis=True, y_axis=True)
    axes[1, 1].legend(ncol=2, frameon=False, loc="upper right")

    save(fig, "retrospective_sa_ga")


def comparison_figure() -> None:
    scaling = read_csv("results/scaling_controlled_matched/summary.csv")
    fits = read_csv(
        "results/retrospective_budget_scaling_inference_20260807/"
        "scaling_exponents.csv"
    )
    budget_runtime = read_csv(
        "results/retrospective_budget_release_full_20260806/analysis/summary.csv"
    )[["experiment_label", "solver_runtime_ms_mean"]]
    budget_inference = read_csv(
        "results/retrospective_budget_scaling_inference_20260807/"
        "budget_paired_comparisons.csv"
    )
    budget = budget_inference.merge(
        budget_runtime,
        on="experiment_label",
        how="left",
        validate="one_to_one",
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 2.55),
        gridspec_kw={"width_ratios": [1.02, 1.18]},
        constrained_layout=True,
    )

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
            label=(
                f"{label}: $b={decimal_comma(fit['empirical_exponent_b'], 2)}$ "
                f"[{decimal_comma(fit['slope_t95_low'], 2)}; "
                f"{decimal_comma(fit['slope_t95_high'], 2)}]"
            ),
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
    axes[0].set_title("(a) Schaalprotocol")
    axes[0].grid(which="both", alpha=0.22)
    axes[0].legend(
        frameon=False,
        loc="upper left",
        fontsize=6.25,
        title="Student-t-95%-BI voor $b$",
        title_fontsize=6.35,
        handlelength=1.8,
        handletextpad=0.5,
    )
    axes[0].text(
        0.98,
        0.04,
        "lijn = mediaan, band = IQR",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
    )

    budget_styles = {
        "SA": ("simulated_annealing", "o"),
        "GA": ("genetic", "s"),
    }
    budget_annotations = {
        "sa_i120": (12, -9),
        "sa_i400": (-16, 8),
        "sa_i2000": (4, 8),
        "sa_i10000": (5, 0),
        "ga_g10": (-15, -9),
        "ga_g20": (-2, 8),
        "ga_g40": (-12, 8),
    }
    short_budget_labels = {
        "sa_i120": "120 i",
        "sa_i400": "400 i",
        "sa_i2000": "2k i",
        "sa_i10000": "10k i",
        "ga_g10": "10 g",
        "ga_g20": "20 g",
        "ga_g40": "40 g",
    }
    all_low = budget["t95_low_gain_pct"].to_numpy(dtype=float)
    all_high = budget["t95_high_gain_pct"].to_numpy(dtype=float)
    star_offset = max(0.002, 0.035 * float(np.max(all_high) - np.min(all_low)))
    for family, (solver, marker) in budget_styles.items():
        part = budget[budget["family"] == family].sort_values("budget")
        x = part["solver_runtime_ms_mean"].to_numpy(dtype=float) / 1000.0
        gain = part["mean_gain_pct"].to_numpy(dtype=float)
        low = part["t95_low_gain_pct"].to_numpy(dtype=float)
        high = part["t95_high_gain_pct"].to_numpy(dtype=float)
        axes[1].errorbar(
            x,
            gain,
            yerr=np.vstack((gain - low, high - gain)),
            marker=marker,
            markersize=3.8,
            linewidth=1.15,
            capsize=2,
            color=COLORS[solver],
            label=family,
        )
        holm_significant = (
            part["holm_reject_at_alpha"].astype(str).str.lower() == "true"
        ).to_numpy()
        if np.any(holm_significant):
            axes[1].scatter(
                x[holm_significant],
                high[holm_significant] + star_offset,
                marker="*",
                s=27,
                color="#111111",
                linewidth=0.3,
                zorder=4,
            )
        for (_, row), x_value, gain_value in zip(part.iterrows(), x, gain):
            experiment = str(row["experiment_label"])
            axes[1].annotate(
                short_budget_labels[experiment],
                (x_value, gain_value),
                xytext=budget_annotations[experiment],
                textcoords="offset points",
                fontsize=6.25,
            )
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("gemiddelde solvertijd (s)")
    axes[1].set_ylabel("gekoppelde winst t.o.v. greedy (%)")
    axes[1].set_title("(b) Release-budgetprotocol")
    axes[1].grid(alpha=0.25)
    axes[1].margins(x=0.10)
    axes[1].set_ylim(
        min(-0.008, float(np.min(all_low)) - 2.0 * star_offset),
        float(np.max(all_high)) + 4.0 * star_offset,
    )
    apply_decimal_comma(axes[1], x_axis=True, y_axis=True)
    budget_handles, budget_labels = axes[1].get_legend_handles_labels()
    budget_handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            color="#111111",
            markersize=6,
        )
    )
    budget_labels.append(r"Holm: $p_{\mathrm{adj}}<0{,}05$")
    axes[1].legend(
        budget_handles,
        budget_labels,
        frameon=False,
        loc="upper right",
        fontsize=6.25,
        handletextpad=0.4,
    )
    axes[1].text(
        0.98,
        0.72,
        "Student-t-95%-BI, $n=10$",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
    )

    save(fig, "retrospective_comparison")


def waves_figure() -> None:
    wave_gains = read_csv(
        "results/retrospective_budget_release_full_20260806/analysis/"
        "release_wave_paired_comparisons.csv"
    )
    greedy_ablation = read_csv(
        "results/retrospective_greedy_development_20260807/greedy_ablation.csv"
    ).sort_values("stage_index")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 2.35),
        gridspec_kw={"width_ratios": [1.32, 1.0]},
        constrained_layout=True,
    )

    method_styles = {
        "sa_i10000": ("SA-10.000", "simulated_annealing", "o", -0.10),
        "ga_g40": ("GA-40", "genetic", "s", 0.10),
    }
    for candidate, (label, solver, marker, x_offset) in method_styles.items():
        part = wave_gains[
            wave_gains["candidate_label"] == candidate
        ].sort_values("wave_id")
        wave_number = (
            part["wave_id"].str.extract(r"(\d+)$", expand=False).astype(int)
        ).to_numpy(dtype=float)
        gain = part["gain_pct_mean"].to_numpy(dtype=float)
        low = part["gain_pct_ci95_low"].to_numpy(dtype=float)
        high = part["gain_pct_ci95_high"].to_numpy(dtype=float)
        x = wave_number + x_offset
        axes[0].errorbar(
            x,
            gain,
            yerr=np.vstack((gain - low, high - gain)),
            fmt=marker,
            linestyle="none",
            markersize=3.5,
            capsize=2,
            elinewidth=0.9,
            color=COLORS[solver],
            label=label,
        )

        holm_significant = (
            part["holm_reject_0_05"].astype(str).str.lower() == "true"
        ).to_numpy()
        if np.any(holm_significant):
            interval_span = float(np.max(high) - np.min(low))
            star_offset = max(0.025, 0.025 * interval_span)
            axes[0].scatter(
                x[holm_significant],
                high[holm_significant] + star_offset,
                marker="*",
                s=31,
                color="#111111",
                linewidth=0.3,
                zorder=4,
            )

    axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[0].set_xticks(np.arange(1, 13))
    axes[0].set_xlabel("wave")
    axes[0].set_ylabel("gekoppelde winst t.o.v. greedy (%)")
    axes[0].set_title("(a) Release-winst per wave")
    axes[0].grid(axis="y", alpha=0.25)
    apply_decimal_comma(axes[0], x_axis=True, y_axis=True)
    method_handles, method_labels = axes[0].get_legend_handles_labels()
    method_handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            color="#111111",
            markersize=6,
        )
    )
    method_labels.append(r"Holm: $p_{\mathrm{adj}}<0{,}05$")
    axes[0].legend(
        method_handles,
        method_labels,
        frameon=False,
        loc="upper left",
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    axes[0].text(
        0.99,
        0.03,
        "gemiddelde en Student-t-95%-BI, $n=10$",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
    )

    stage_labels = {
        "savings": "basis",
        "local_search": "lokaal",
        "elimination_post_ls": "elimin. +\npost-LS",
        "pair_final_ls": "paar +\nfinal-LS",
        "triple_rebuild": "triple-\nherbouw",
        "triple_cycle_bundle": "triplecyclus\n+ venster",
        "tail_pair_rebuild": "slot-\npaar",
    }
    x = np.arange(len(greedy_ablation))
    labels = [stage_labels[str(value)] for value in greedy_ablation["mechanism"]]
    cumulative = greedy_ablation["cumulative_gain_pct_baseline"].to_numpy(dtype=float)
    baseline_objective = float(greedy_ablation.iloc[0]["objective"])
    marginal = (
        100.0
        * greedy_ablation["marginal_gain_abs"].to_numpy(dtype=float)
        / baseline_objective
    )
    bars = axes[1].bar(
        x,
        marginal,
        width=0.62,
        color="#9ecae1",
        edgecolor="#4c78a8",
        linewidth=0.5,
        label="marginale bijdrage",
        zorder=2,
    )
    line = axes[1].plot(
        x,
        cumulative,
        "o-",
        color=COLORS["greedy"],
        linewidth=1.4,
        markersize=3.8,
        label="cumulatieve winst",
        zorder=3,
    )[0]
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("afstandsreductie t.o.v. basis (%)")
    axes[1].set_title("(b) Greedy-tuning per fase")
    axes[1].grid(axis="y", alpha=0.25, zorder=0)
    apply_decimal_comma(axes[1], y_axis=True)
    axes[1].legend(
        [line, bars],
        ["cumulatief", "marginaal"],
        frameon=False,
        loc="upper left",
    )
    axes[1].set_ylim(0, max(cumulative) * 1.18)
    axes[1].annotate(
        f"{decimal_comma(greedy_ablation.iloc[1]['share_of_final_gain_pct'], 0)}% "
        "van eindwinst",
        (x[1], marginal[1]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=5.8,
    )
    axes[1].annotate(
        f"{int(greedy_ablation.iloc[2]['routes'])} routes",
        (x[2], cumulative[2]),
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        fontsize=5.8,
    )
    axes[1].annotate(
        f"{decimal_comma(cumulative[-1], 2)}% totaal",
        (x[-1], cumulative[-1]),
        xytext=(-3, 7),
        textcoords="offset points",
        ha="right",
        fontsize=5.8,
    )

    save(fig, "retrospective_waves")


def seed_figure() -> None:
    analysis_dir = "results/retrospective_seed_k_timing_release_20260807/analysis"
    summary = read_csv(f"{analysis_dir}/summary.csv")
    k_sensitivity = read_csv(f"{analysis_dir}/k_sensitivity.csv")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.35),
        gridspec_kw={"width_ratios": [1.10, 1.0, 1.0]},
        constrained_layout=True,
    )

    seed_styles = {
        "largest": ("grootste span", COLORS["seed_selection"], "o"),
        "smallest": ("kleinste span", "#8c6d31", "s"),
    }
    seed_order = list(seed_styles)
    selection_order = ["overlap", "span"]
    selection_labels = ["max. locatie-\noverlap", "min. span-\nuitbreiding"]

    def unique_row(
        frame: pd.DataFrame,
        *,
        seed: str,
        selection: str,
    ) -> pd.Series:
        selected = frame[
            (frame["seed_rule"] == seed)
            & (frame["selection_rule"] == selection)
        ]
        if len(selected) != 1:
            raise ValueError(
                "verwacht precies een resultaat voor "
                f"seed={seed!r}, selectie={selection!r}; gevonden: {len(selected)}"
            )
        return selected.iloc[0]

    matrix = np.empty((2, 2))
    for i, seed in enumerate(seed_order):
        for j, selection in enumerate(selection_order):
            matrix[i, j] = (
                float(unique_row(summary, seed=seed, selection=selection)["objective"])
                / 1e6
            )
    im = axes[0].imshow(matrix, cmap="viridis_r", aspect="auto")
    value_range = float(np.ptp(matrix))
    for i in range(2):
        for j in range(2):
            normalized = (
                (matrix[i, j] - float(np.min(matrix))) / value_range
                if value_range > 0
                else 0.5
            )
            axes[0].text(
                j,
                i,
                decimal_comma(matrix[i, j], 3),
                ha="center",
                va="center",
                color="white" if normalized > 0.58 else "black",
                fontsize=7.5,
            )
    axes[0].set_xticks([0, 1], selection_labels)
    axes[0].set_yticks(
        [0, 1],
        [seed_styles[seed][0] for seed in seed_order],
    )
    axes[0].set_xlabel("selectieregel")
    axes[0].set_ylabel("startregel")
    axes[0].set_title(r"(a) Oorspronkelijke $2\times2$-regels")
    cbar = fig.colorbar(im, ax=axes[0], pad=0.02, shrink=0.82)
    cbar.set_label(r"doelfunctie ($10^6$)", fontsize=7)
    apply_decimal_comma(cbar.ax, y_axis=True)

    expected_k = [2, 4, 8, 16]
    for seed in seed_order:
        label, color, marker = seed_styles[seed]
        part = k_sensitivity[k_sensitivity["seed_rule"] == seed].copy()
        part["lookahead_width"] = pd.to_numeric(part["lookahead_width"])
        part = part.sort_values("lookahead_width")
        observed_k = part["lookahead_width"].astype(int).tolist()
        if observed_k != expected_k:
            raise ValueError(
                f"onvolledige K-reeks voor seed={seed!r}: {observed_k}"
            )

        axes[1].plot(
            part["lookahead_width"],
            part["gain_vs_immediate_pct"],
            color=color,
            marker=marker,
            linewidth=1.3,
            markersize=4,
            label=label,
        )

        median_seconds = part["solver_runtime_ms_median"].to_numpy(dtype=float) / 1000.0
        q1_seconds = part["solver_runtime_ms_q1"].to_numpy(dtype=float) / 1000.0
        q3_seconds = part["solver_runtime_ms_q3"].to_numpy(dtype=float) / 1000.0
        axes[2].errorbar(
            part["lookahead_width"],
            median_seconds,
            yerr=np.vstack(
                (median_seconds - q1_seconds, q3_seconds - median_seconds)
            ),
            color=color,
            marker=marker,
            linewidth=1.2,
            markersize=4,
            capsize=2.5,
            label=label,
        )

    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel(r"voorselectiebreedte $K$")
    axes[1].set_ylabel("winst t.o.v. directe kost (%)")
    axes[1].set_title("(b) Kwaliteitswinst vooruitblik")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="best", fontsize=6.2)

    axes[2].set_xlabel(r"voorselectiebreedte $K$")
    axes[2].set_ylabel("interne tijd, mediaan [IQR] (s)")
    axes[2].set_title("(c) Tijdskost vooruitblik")
    axes[2].grid(alpha=0.25)

    for ax in axes[1:]:
        ax.set_xticks(expected_k)
        apply_decimal_comma(ax, x_axis=True, y_axis=True)

    save(fig, "retrospective_seed")


def main() -> None:
    setup_style()
    sa_ga_figure()
    comparison_figure()
    waves_figure()
    seed_figure()


if __name__ == "__main__":
    main()
