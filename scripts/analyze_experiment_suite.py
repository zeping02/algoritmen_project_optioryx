"""Create report-ready statistics for the preregistered experiment suites.

The statistical replicate is always one complete instance solved with one
configuration and seed.  Raw wave rows are summed before any means, standard
deviations, confidence intervals, contrasts, or paired differences are made.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_COLUMNS = {
    "solver",
    "config_id",
    "wave_id",
    "seed",
    "objective",
    "runtime_ms",
}
MANIFEST_COLUMNS = {"suite", "solver", "config_id", "parameters", "seeds"}
CONVERGENCE_COLUMNS = {
    "solver",
    "config_id",
    "wave_id",
    "seed",
    "step",
    "control_value",
}
T_975 = {
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


def require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def canonical_value(value: object) -> str:
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def parse_parameters(value: object) -> dict[str, str]:
    parameters: dict[str, str] = {}
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"invalid manifest parameter token: {token!r}")
        key, parameter_value = token.split("=", 1)
        if not key or key in parameters:
            raise ValueError(f"duplicate or empty manifest parameter: {key!r}")
        parameters[key] = parameter_value
    return parameters


def expand_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    require_columns(manifest, MANIFEST_COLUMNS, "suite_manifest.csv")
    if manifest["config_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["config_id"].duplicated(keep=False), "config_id"
        ].unique()
        raise ValueError(f"duplicate config_id values in manifest: {duplicates.tolist()}")
    declared_config_ids = manifest["config_id"].astype(str).reset_index(drop=True)
    parsed = [parse_parameters(value) for value in manifest["parameters"]]
    keys = sorted({key for parameters in parsed for key in parameters})
    expanded = manifest.copy()
    for key in keys:
        expanded[key] = [parameters.get(key, pd.NA) for parameters in parsed]
    parameter_config_ids = pd.Series(
        [parameters.get("config_id", "") for parameters in parsed], dtype=str
    )
    if not parameter_config_ids.equals(declared_config_ids):
        raise ValueError("manifest config_id does not match parameters config_id")
    return expanded


def build_seed_totals(runs: pd.DataFrame) -> pd.DataFrame:
    require_columns(runs, RUN_COLUMNS, "runs.csv")
    numeric = runs[["objective", "runtime_ms"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.map(math.isfinite).all().all():
        raise ValueError("runs.csv contains non-finite objective/runtime values")
    if (numeric["runtime_ms"] < 0).any():
        raise ValueError("runs.csv contains a negative runtime")
    duplicate_keys = ["solver", "config_id", "wave_id", "seed"]
    if runs.duplicated(duplicate_keys).any():
        example = runs.loc[runs.duplicated(duplicate_keys, keep=False), duplicate_keys].head(5)
        raise ValueError(
            "runs.csv contains duplicate config/seed/wave rows: "
            f"{example.to_dict(orient='records')}"
        )
    totals = (
        runs.assign(
            objective=numeric["objective"],
            runtime_ms=numeric["runtime_ms"],
            seed=runs["seed"].map(canonical_value),
        )
        .groupby(["solver", "config_id", "seed"], as_index=False, dropna=False)
        .agg(
            total_objective=("objective", "sum"),
            total_runtime_ms=("runtime_ms", "sum"),
            wave_count=("wave_id", "nunique"),
        )
    )
    expected_wave_count = runs["wave_id"].nunique()
    incomplete = totals["wave_count"].to_numpy() != expected_wave_count
    if incomplete.any():
        example = totals.loc[incomplete, ["solver", "config_id", "seed"]].head(5)
        raise ValueError(
            "not every seed covers the complete wave set: "
            f"{example.to_dict(orient='records')}"
        )
    return totals.sort_values(["solver", "config_id", "seed"], kind="stable")


def ci95(std: float, count: int) -> float:
    if count < 2 or not math.isfinite(std):
        return float("nan")
    critical = T_975.get(count - 1, 1.96)
    return critical * std / math.sqrt(count)


def series_statistics(values: Iterable[float], prefix: str) -> dict[str, float | int]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    count = int(series.size)
    standard_deviation = float(series.std(ddof=1)) if count >= 2 else float("nan")
    return {
        f"{prefix}_n": count,
        f"{prefix}_mean": float(series.mean()) if count else float("nan"),
        f"{prefix}_std": standard_deviation,
        f"{prefix}_ci95": ci95(standard_deviation, count),
        f"{prefix}_best": float(series.min()) if count else float("nan"),
    }


def attach_manifest(
    seed_totals: pd.DataFrame, manifest: pd.DataFrame, solver: str
) -> pd.DataFrame:
    selected_manifest = manifest[manifest["solver"] == solver].copy()
    selected_totals = seed_totals[seed_totals["solver"] == solver].copy()
    if selected_manifest.empty or selected_totals.empty:
        raise ValueError(f"no {solver} rows are available")
    missing_runs = set(selected_manifest["config_id"]) - set(selected_totals["config_id"])
    extra_runs = set(selected_totals["config_id"]) - set(selected_manifest["config_id"])
    if missing_runs or extra_runs:
        raise ValueError(
            f"manifest/runs config mismatch; missing={sorted(missing_runs)}, extra={sorted(extra_runs)}"
        )
    for row in selected_manifest.itertuples():
        expected_seeds = {
            canonical_value(seed) for seed in str(row.seeds).split(";") if str(seed).strip()
        }
        observed_seeds = set(
            selected_totals.loc[
                selected_totals["config_id"] == row.config_id, "seed"
            ].map(canonical_value)
        )
        if expected_seeds != observed_seeds:
            raise ValueError(
                f"seed set mismatch for {row.config_id}: "
                f"manifest={sorted(expected_seeds)}, runs={sorted(observed_seeds)}"
            )
        if {"wave_runs", "instance_runs"}.issubset(selected_manifest.columns):
            instance_runs = int(row.instance_runs)
            wave_runs = int(row.wave_runs)
            if instance_runs <= 0 or wave_runs % instance_runs != 0:
                raise ValueError(f"invalid manifest workload counts for {row.config_id}")
            expected_wave_count = wave_runs // instance_runs
            observed_wave_counts = set(
                selected_totals.loc[
                    selected_totals["config_id"] == row.config_id, "wave_count"
                ]
            )
            if observed_wave_counts != {expected_wave_count}:
                raise ValueError(
                    f"wave-count mismatch for {row.config_id}: "
                    f"manifest={expected_wave_count}, runs={sorted(observed_wave_counts)}"
                )
    return selected_totals.merge(
        selected_manifest, on=["solver", "config_id"], how="inner", validate="many_to_one"
    )


def config_statistics(data: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for config_id, frame in data.groupby("config_id", sort=False):
        record: dict[str, object] = {"config_id": config_id}
        for factor in factors:
            record[factor] = frame.iloc[0][factor]
        record.update(series_statistics(frame["total_objective"], "objective"))
        record.update(series_statistics(frame["total_runtime_ms"], "runtime_ms"))
        records.append(record)
    return pd.DataFrame(records)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    figure.tight_layout()
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=180)
    figure.savefig(paths[1])
    plt.close(figure)
    return [path.name for path in paths]


def preferred_reference(factor: str, levels: list[str]) -> str:
    preferred = {
        "cooling": "geometric",
        "neighborhood": "relocate",
        "start_strategy": "greedy",
    }.get(factor)
    return preferred if preferred in levels else sorted(levels)[0]


def contrast_statistics(values: pd.Series, metric: str) -> dict[str, object]:
    return series_statistics(values, metric)


def analyze_sa(data: pd.DataFrame, output_dir: Path) -> tuple[list[str], dict[str, object]]:
    factors = ["cooling", "neighborhood", "start_strategy"]
    for factor in factors:
        if factor not in data or data[factor].isna().any():
            raise ValueError(f"SA manifest lacks factor {factor}")
    levels = {factor: sorted(data[factor].astype(str).unique()) for factor in factors}
    if any(len(values) != 2 for values in levels.values()):
        raise ValueError(f"SA requires two levels per factor, got {levels}")
    cells = data[factors].drop_duplicates()
    if len(cells) != 8:
        raise ValueError("SA manifest is not a complete 2x2x2 factorial")
    seed_sets = {
        config_id: frozenset(frame["seed"].astype(str))
        for config_id, frame in data.groupby("config_id")
    }
    if len(set(seed_sets.values())) != 1:
        raise ValueError("SA factorial cells do not use the same seed set")

    config_table = config_statistics(data, factors).sort_values(factors, kind="stable")
    config_table.to_csv(output_dir / "sa_config_summary.csv", index=False)

    references = {factor: preferred_reference(factor, levels[factor]) for factor in factors}
    main_records: list[dict[str, object]] = []
    main_contrasts: list[dict[str, object]] = []
    for factor in factors:
        per_seed_level = (
            data.groupby(["seed", factor], as_index=False)
            .agg(
                total_objective=("total_objective", "mean"),
                total_runtime_ms=("total_runtime_ms", "mean"),
            )
        )
        reference = references[factor]
        reference_frame = per_seed_level[per_seed_level[factor].astype(str) == reference]
        for level in levels[factor]:
            selected = per_seed_level[per_seed_level[factor].astype(str) == level]
            paired = selected.merge(
                reference_frame,
                on="seed",
                suffixes=("_level", "_reference"),
                validate="one_to_one",
            )
            record: dict[str, object] = {
                "factor": factor,
                "level": level,
                "reference_level": reference,
            }
            record.update(series_statistics(selected["total_objective"], "objective"))
            record.update(series_statistics(selected["total_runtime_ms"], "runtime_ms"))
            record.update(
                series_statistics(
                    paired["total_objective_level"] - paired["total_objective_reference"],
                    "objective_delta_vs_reference",
                )
            )
            record.update(
                series_statistics(
                    paired["total_runtime_ms_level"] - paired["total_runtime_ms_reference"],
                    "runtime_ms_delta_vs_reference",
                )
            )
            main_records.append(record)
            if level != reference:
                main_contrasts.append(record)
    main_table = pd.DataFrame(main_records)
    main_table.to_csv(output_dir / "sa_main_effects.csv", index=False)

    interaction_records: list[dict[str, object]] = []
    for first_index, first in enumerate(factors):
        for second in factors[first_index + 1 :]:
            remaining = [factor for factor in factors if factor not in {first, second}][0]
            cells_by_seed = (
                data.groupby(["seed", first, second], as_index=False)
                .agg(
                    total_objective=("total_objective", "mean"),
                    total_runtime_ms=("total_runtime_ms", "mean"),
                )
            )
            first_reference = references[first]
            second_reference = references[second]
            first_alternative = next(level for level in levels[first] if level != first_reference)
            second_alternative = next(level for level in levels[second] if level != second_reference)
            objective = cells_by_seed.pivot(
                index="seed", columns=[first, second], values="total_objective"
            )
            runtime = cells_by_seed.pivot(
                index="seed", columns=[first, second], values="total_runtime_ms"
            )
            objective_contrast = (
                objective[(first_alternative, second_alternative)]
                - objective[(first_alternative, second_reference)]
                - objective[(first_reference, second_alternative)]
                + objective[(first_reference, second_reference)]
            )
            runtime_contrast = (
                runtime[(first_alternative, second_alternative)]
                - runtime[(first_alternative, second_reference)]
                - runtime[(first_reference, second_alternative)]
                + runtime[(first_reference, second_reference)]
            )
            record = {
                "interaction_order": 2,
                "factors": f"{first}:{second}",
                "averaged_over": remaining,
                "contrast_definition": "alt_alt-alt_ref-ref_alt+ref_ref",
            }
            record.update(contrast_statistics(objective_contrast, "objective_contrast"))
            record.update(contrast_statistics(runtime_contrast, "runtime_ms_contrast"))
            interaction_records.append(record)

    per_cell = data.groupby(["seed", *factors], as_index=False).agg(
        total_objective=("total_objective", "mean"),
        total_runtime_ms=("total_runtime_ms", "mean"),
    )
    objective_three = pd.Series(0.0, index=sorted(per_cell["seed"].unique()))
    runtime_three = pd.Series(0.0, index=objective_three.index)
    for _, row in per_cell.iterrows():
        alternatives = sum(
            str(row[factor]) != references[factor] for factor in factors
        )
        coefficient = 1.0 if alternatives in {1, 3} else -1.0
        objective_three.loc[row["seed"]] += coefficient * row["total_objective"]
        runtime_three.loc[row["seed"]] += coefficient * row["total_runtime_ms"]
    three_record: dict[str, object] = {
        "interaction_order": 3,
        "factors": ":".join(factors),
        "averaged_over": "",
        "contrast_definition": "three_factor_difference_of_differences",
    }
    three_record.update(contrast_statistics(objective_three, "objective_contrast"))
    three_record.update(contrast_statistics(runtime_three, "runtime_ms_contrast"))
    interaction_records.append(three_record)
    interaction_table = pd.DataFrame(interaction_records)
    interaction_table.to_csv(output_dir / "sa_interactions.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    main_frame = pd.DataFrame(main_contrasts)
    factor_labels = {
        "cooling": "afkoeling",
        "neighborhood": "buurt",
        "start_strategy": "start",
    }
    level_labels = {
        "geometric": "geometrisch",
        "lundy_mees": "Lundy-Mees",
        "relocate": "verplaats",
        "swap": "wissel",
        "greedy": "greedy",
        "randomized": "willekeurig",
    }
    main_labels = [
        f"{factor_labels.get(row.factor, row.factor)}\n"
        f"{level_labels.get(row.level, row.level)} - "
        f"{level_labels.get(row.reference_level, row.reference_level)}"
        for row in main_frame.itertuples()
    ]
    interaction_labels = [
        " x ".join(factor_labels.get(factor, factor) for factor in factors.split(":"))
        for factors in interaction_table["factors"]
    ]
    for axis, value, error, title, labels in [
        (
            axes[0, 0],
            main_frame["objective_delta_vs_reference_mean"],
            main_frame["objective_delta_vs_reference_ci95"],
            "SA-hoofdeffecten: doelfunctie",
            main_labels,
        ),
        (
            axes[0, 1],
            main_frame["runtime_ms_delta_vs_reference_mean"],
            main_frame["runtime_ms_delta_vs_reference_ci95"],
            "SA-hoofdeffecten: uitvoeringstijd",
            main_labels,
        ),
        (
            axes[1, 0],
            interaction_table["objective_contrast_mean"],
            interaction_table["objective_contrast_ci95"],
            "SA-interactiecontrasten: doelfunctie",
            interaction_labels,
        ),
        (
            axes[1, 1],
            interaction_table["runtime_ms_contrast_mean"],
            interaction_table["runtime_ms_contrast_ci95"],
            "SA-interactiecontrasten: uitvoeringstijd",
            interaction_labels,
        ),
    ]:
        positions = range(len(labels))
        axis.errorbar(
            list(positions), value, yerr=error.fillna(0.0), fmt="o", capsize=4
        )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xticks(list(positions), labels, rotation=20, ha="right")
        axis.set_title(title)
    artifacts = [
        "sa_config_summary.csv",
        "sa_main_effects.csv",
        "sa_interactions.csv",
        *save_figure(figure, output_dir, "sa_effects"),
    ]
    return artifacts, {
        "factor_levels": levels,
        "reference_levels": references,
        "effect_estimation": "paired_seed_marginal_contrasts",
        "interaction_sign": "alternative_minus_reference_difference_of_differences",
    }


GA_FACTORS = ["population_size", "generations", "mutation_rate", "elitism_fraction"]


def detect_ga_baseline(manifest: pd.DataFrame, explicit: str | None) -> str:
    for factor in GA_FACTORS:
        if factor not in manifest or manifest[factor].isna().any():
            raise ValueError(f"GA manifest lacks factor {factor}")
    normalized = manifest.copy()
    for factor in GA_FACTORS:
        normalized[factor] = normalized[factor].map(canonical_value)
    if explicit:
        if explicit not in set(normalized["config_id"]):
            raise ValueError(f"baseline config {explicit!r} is not in the manifest")
        baseline_id = explicit
    else:
        candidates: list[str] = []
        for _, candidate in normalized.iterrows():
            difference_factors = []
            for _, other in normalized.iterrows():
                if other["config_id"] == candidate["config_id"]:
                    continue
                changed = [
                    factor
                    for factor in GA_FACTORS
                    if other[factor] != candidate[factor]
                ]
                difference_factors.append(changed)
            counts = {
                factor: sum(changed == [factor] for changed in difference_factors)
                for factor in GA_FACTORS
            }
            if all(len(changed) == 1 for changed in difference_factors) and all(
                counts[factor] == 2 for factor in GA_FACTORS
            ):
                candidates.append(str(candidate["config_id"]))
        if len(candidates) != 1:
            raise ValueError(
                "GA baseline is ambiguous; pass --baseline-config explicitly "
                f"(structural candidates: {candidates})"
            )
        baseline_id = candidates[0]

    constant_factors = ["tournament_size", "insertion_rate"]
    for factor in constant_factors:
        if factor in normalized and normalized[factor].dropna().map(canonical_value).nunique() > 1:
            raise ValueError(f"GA OFAT suite changes non-target factor {factor}")
    baseline = normalized[normalized["config_id"] == baseline_id].iloc[0]
    changed_sets = []
    for _, row in normalized[normalized["config_id"] != baseline_id].iterrows():
        changed_sets.append(
            [factor for factor in GA_FACTORS if row[factor] != baseline[factor]]
        )
    factor_counts = {
        factor: sum(changed == [factor] for changed in changed_sets)
        for factor in GA_FACTORS
    }
    if not all(len(changed) == 1 for changed in changed_sets) or any(
        factor_counts[factor] != 2 for factor in GA_FACTORS
    ):
        raise ValueError(
            f"selected GA baseline is not the centre of a 2-per-factor OFAT design: {factor_counts}"
        )
    return baseline_id


def final_ga_diversity(
    convergence: pd.DataFrame | None, manifest: pd.DataFrame
) -> pd.DataFrame:
    columns = ["config_id", "seed", "final_diversity"]
    if convergence is None or convergence.empty:
        return pd.DataFrame(columns=columns)
    require_columns(convergence, CONVERGENCE_COLUMNS, "convergence.csv")
    ga = convergence[convergence["solver"] == "genetic"].copy()
    if ga.empty:
        return pd.DataFrame(columns=columns)
    ga["step"] = pd.to_numeric(ga["step"], errors="coerce")
    ga["control_value"] = pd.to_numeric(ga["control_value"], errors="coerce")
    ga["seed"] = ga["seed"].map(canonical_value)
    if ga[["step", "control_value"]].isna().any().any():
        raise ValueError("convergence.csv has invalid GA step/control values")
    expected_steps = {
        str(row.config_id): int(Decimal(str(row.generations)))
        for row in manifest.itertuples()
    }
    missing_configs = set(expected_steps) - set(ga["config_id"].astype(str))
    if missing_configs:
        raise ValueError(
            f"convergence.csv is missing GA configs: {sorted(missing_configs)}"
        )
    final_rows: list[pd.DataFrame] = []
    for config_id, frame in ga.groupby("config_id"):
        expected = expected_steps.get(str(config_id))
        if expected is None:
            continue
        selected = frame[frame["step"] == expected]
        keys = ["config_id", "seed", "wave_id"]
        if selected.duplicated(keys).any():
            raise ValueError(f"duplicate GA final-diversity rows for {config_id}")
        expected_groups = frame[keys].drop_duplicates()
        if len(selected) != len(expected_groups):
            raise ValueError(f"GA convergence does not reach generation {expected} for {config_id}")
        manifest_row = manifest[manifest["config_id"].astype(str) == str(config_id)].iloc[0]
        expected_seeds = {
            canonical_value(seed)
            for seed in str(manifest_row["seeds"]).split(";")
            if str(seed).strip()
        }
        if set(selected["seed"]) != expected_seeds:
            raise ValueError(f"GA final diversity has an incomplete seed set for {config_id}")
        if {"wave_runs", "instance_runs"}.issubset(manifest.columns):
            expected_wave_count = int(manifest_row["wave_runs"]) // int(
                manifest_row["instance_runs"]
            )
            observed_wave_counts = selected.groupby("seed")["wave_id"].nunique()
            if not (observed_wave_counts == expected_wave_count).all():
                raise ValueError(f"GA final diversity has incomplete waves for {config_id}")
        final_rows.append(selected)
    if not final_rows:
        return pd.DataFrame(columns=columns)
    final = pd.concat(final_rows, ignore_index=True)
    if ((final["control_value"] < 0) | (final["control_value"] > 1)).any():
        raise ValueError("GA final diversity is outside [0, 1]")
    return (
        final.groupby(["config_id", "seed"], as_index=False)
        .agg(final_diversity=("control_value", "mean"))
        .sort_values(["config_id", "seed"], kind="stable")
    )


def analyze_ga(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    convergence: pd.DataFrame | None,
    output_dir: Path,
    baseline_argument: str | None,
) -> tuple[list[str], dict[str, object]]:
    baseline_id = detect_ga_baseline(manifest, baseline_argument)
    normalized_manifest = manifest.copy()
    for factor in GA_FACTORS:
        normalized_manifest[f"_{factor}"] = normalized_manifest[factor].map(canonical_value)
    baseline_parameters = normalized_manifest[
        normalized_manifest["config_id"] == baseline_id
    ].iloc[0]
    baseline_runs = data[data["config_id"] == baseline_id][
        ["seed", "total_objective", "total_runtime_ms"]
    ].rename(
        columns={
            "total_objective": "baseline_objective",
            "total_runtime_ms": "baseline_runtime_ms",
        }
    )
    baseline_seeds = frozenset(baseline_runs["seed"].astype(str))
    diversity = final_ga_diversity(convergence, manifest)
    diversity_lookup = diversity.set_index(["config_id", "seed"])["final_diversity"]

    raw_records: list[dict[str, object]] = []
    for _, config in normalized_manifest.iterrows():
        config_id = str(config["config_id"])
        config_runs = data[data["config_id"] == config_id][
            ["seed", "total_objective", "total_runtime_ms"]
        ]
        if frozenset(config_runs["seed"].astype(str)) != baseline_seeds:
            raise ValueError(f"GA config {config_id} does not share the baseline seed set")
        changed = [
            factor
            for factor in GA_FACTORS
            if config[f"_{factor}"] != baseline_parameters[f"_{factor}"]
        ]
        if config_id == baseline_id:
            changed_factor = "baseline"
        elif len(changed) == 1:
            changed_factor = changed[0]
        else:
            raise ValueError(f"GA config {config_id} changes {len(changed)} OFAT factors")
        paired = config_runs.merge(
            baseline_runs, on="seed", validate="one_to_one", how="inner"
        )
        for row in paired.itertuples():
            final_diversity = diversity_lookup.get((config_id, row.seed), float("nan"))
            baseline_diversity = diversity_lookup.get((baseline_id, row.seed), float("nan"))
            raw_records.append(
                {
                    "config_id": config_id,
                    "baseline_config_id": baseline_id,
                    "changed_factor": changed_factor,
                    "baseline_value": (
                        baseline_parameters[changed_factor]
                        if changed_factor != "baseline"
                        else pd.NA
                    ),
                    "variant_value": config[changed_factor] if changed_factor != "baseline" else pd.NA,
                    "seed": row.seed,
                    "objective": row.total_objective,
                    "baseline_objective": row.baseline_objective,
                    "objective_delta": row.total_objective - row.baseline_objective,
                    "objective_delta_percent": (
                        100.0 * (row.total_objective - row.baseline_objective) / row.baseline_objective
                        if row.baseline_objective != 0
                        else float("nan")
                    ),
                    "runtime_ms": row.total_runtime_ms,
                    "baseline_runtime_ms": row.baseline_runtime_ms,
                    "runtime_ms_delta": row.total_runtime_ms - row.baseline_runtime_ms,
                    "runtime_delta_percent": (
                        100.0 * (row.total_runtime_ms - row.baseline_runtime_ms) / row.baseline_runtime_ms
                        if row.baseline_runtime_ms != 0
                        else float("nan")
                    ),
                    "final_diversity": final_diversity,
                    "baseline_final_diversity": baseline_diversity,
                    "final_diversity_delta": final_diversity - baseline_diversity,
                }
            )
    raw = pd.DataFrame(raw_records).sort_values(
        ["changed_factor", "config_id", "seed"], kind="stable"
    )
    raw.to_csv(output_dir / "ga_paired_seed_deltas.csv", index=False)

    summary_records: list[dict[str, object]] = []
    for config_id, frame in raw.groupby("config_id", sort=False):
        first = frame.iloc[0]
        record: dict[str, object] = {
            "config_id": config_id,
            "baseline_config_id": baseline_id,
            "changed_factor": first["changed_factor"],
            "baseline_value": first["baseline_value"],
            "variant_value": first["variant_value"],
            "paired_seed_count": frame["seed"].nunique(),
        }
        for source, prefix in [
            ("objective", "objective"),
            ("objective_delta", "objective_delta"),
            ("objective_delta_percent", "objective_delta_percent"),
            ("runtime_ms", "runtime_ms"),
            ("runtime_ms_delta", "runtime_ms_delta"),
            ("runtime_delta_percent", "runtime_delta_percent"),
            ("final_diversity", "final_diversity"),
            ("final_diversity_delta", "final_diversity_delta"),
        ]:
            record.update(series_statistics(frame[source], prefix))
        summary_records.append(record)
    summary = pd.DataFrame(summary_records).sort_values(
        ["changed_factor", "variant_value"], kind="stable"
    )
    summary.to_csv(output_dir / "ga_sensitivity_summary.csv", index=False)

    variants = summary[summary["changed_factor"] != "baseline"].copy()
    ga_factor_labels = {
        "elitism_fraction": "elitisme",
        "generations": "generaties",
        "mutation_rate": "mutatie",
        "population_size": "populatie",
    }
    labels = [
        f"{ga_factor_labels.get(row.changed_factor, row.changed_factor)}\n"
        f"{row.variant_value} (basis {row.baseline_value})"
        for row in variants.itertuples()
    ]
    plot_specs = [
        ("objective_delta_mean", "objective_delta_ci95", "Gekoppeld verschil in doelfunctie"),
        ("runtime_delta_percent_mean", "runtime_delta_percent_ci95", "Gekoppeld tijdsverschil (%)"),
    ]
    if variants["final_diversity_n"].max() > 0:
        plot_specs.append(
            (
                "final_diversity_delta_mean",
                "final_diversity_delta_ci95",
                "Gekoppeld verschil in einddiversiteit",
            )
        )
    figure, axes = plt.subplots(
        1, len(plot_specs), figsize=(5.2 * len(plot_specs), 5), squeeze=False
    )
    for axis, (mean_column, ci_column, title) in zip(axes[0], plot_specs):
        positions = list(range(len(variants)))
        axis.errorbar(
            positions,
            variants[mean_column],
            yerr=variants[ci_column].fillna(0.0),
            fmt="o",
            capsize=4,
        )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_title(title)
        if mean_column == "objective_delta_mean":
            axis.set_ylabel("Variant minus baseline (lager is beter)")
        elif mean_column == "runtime_delta_percent_mean":
            axis.set_ylabel("Variant minus baseline (%)")
        else:
            axis.set_ylabel("Variant minus baseline")
    artifacts = [
        "ga_paired_seed_deltas.csv",
        "ga_sensitivity_summary.csv",
        *save_figure(figure, output_dir, "ga_sensitivity"),
    ]
    return artifacts, {
        "baseline_config_id": baseline_id,
        "delta_sign": "variant_minus_baseline",
        "pairing_key": "seed",
        "final_diversity_aggregation": "mean_of_final_wave_diversities_per_seed",
    }


def analyze_seed(data: pd.DataFrame, output_dir: Path) -> tuple[list[str], dict[str, object]]:
    factors = ["seed_rule", "selection_rule"]
    for factor in factors:
        if factor not in data or data[factor].isna().any():
            raise ValueError(f"seed-selection manifest lacks factor {factor}")
    seed_rules = sorted(data["seed_rule"].astype(str).unique())
    if len(seed_rules) != 2:
        raise ValueError("seed factorial requires exactly two seed rules")
    selections = set(data["selection_rule"].astype(str))
    preferred_paper = {"maximum_location_overlap", "minimum_span_expansion"}
    if preferred_paper.issubset(selections):
        paper_selections = sorted(preferred_paper)
    else:
        excluded = {"bounded_two_step_lookahead", "minimum_incremental_cost", "hybrid_score"}
        paper_selections = sorted(selections - excluded)
    if len(paper_selections) != 2:
        raise ValueError(f"cannot identify two paper selection rules from {sorted(selections)}")

    config_table = config_statistics(data, factors)
    paper = config_table[config_table["selection_rule"].isin(paper_selections)].copy()
    expected_cells = {(seed, selection) for seed in seed_rules for selection in paper_selections}
    actual_cells = set(zip(paper["seed_rule"].astype(str), paper["selection_rule"].astype(str)))
    if actual_cells != expected_cells:
        raise ValueError("seed paper design is not a complete 2x2")
    paper = paper.sort_values(factors, kind="stable")
    paper.to_csv(output_dir / "seed_paper_2x2.csv", index=False)

    custom_rule = "bounded_two_step_lookahead"
    immediate_rule = "minimum_incremental_cost"
    if not {custom_rule, immediate_rule}.issubset(selections):
        raise ValueError("seed ablation requires bounded lookahead and immediate cost")
    ablation_records: list[dict[str, object]] = []
    for seed_rule in seed_rules:
        custom = data[
            (data["seed_rule"].astype(str) == seed_rule)
            & (data["selection_rule"].astype(str) == custom_rule)
        ]
        immediate = data[
            (data["seed_rule"].astype(str) == seed_rule)
            & (data["selection_rule"].astype(str) == immediate_rule)
        ]
        if custom.empty or immediate.empty:
            raise ValueError(f"seed ablation is incomplete for {seed_rule}")
        if "lookahead_width" in custom and not all(
            canonical_value(value) == "8" for value in custom["lookahead_width"]
        ):
            raise ValueError("bounded lookahead ablation must use lookahead_width=8")
        paired = custom[["seed", "config_id", "total_objective", "total_runtime_ms"]].merge(
            immediate[["seed", "config_id", "total_objective", "total_runtime_ms"]],
            on="seed",
            suffixes=("_custom", "_immediate"),
            validate="one_to_one",
        )
        objective_delta = paired["total_objective_custom"] - paired["total_objective_immediate"]
        runtime_delta = paired["total_runtime_ms_custom"] - paired["total_runtime_ms_immediate"]
        record: dict[str, object] = {
            "seed_rule": seed_rule,
            "custom_selection_rule": custom_rule,
            "immediate_selection_rule": immediate_rule,
            "custom_config_id": custom.iloc[0]["config_id"],
            "immediate_config_id": immediate.iloc[0]["config_id"],
            "custom_objective_mean": custom["total_objective"].mean(),
            "immediate_objective_mean": immediate["total_objective"].mean(),
            "custom_runtime_ms_mean": custom["total_runtime_ms"].mean(),
            "immediate_runtime_ms_mean": immediate["total_runtime_ms"].mean(),
        }
        record.update(series_statistics(objective_delta, "objective_delta_custom_minus_immediate"))
        record.update(series_statistics(runtime_delta, "runtime_ms_delta_custom_minus_immediate"))
        percent = 100.0 * objective_delta / paired["total_objective_immediate"].replace(0, pd.NA)
        record.update(series_statistics(percent, "objective_delta_percent"))
        ablation_records.append(record)
    ablation = pd.DataFrame(ablation_records).sort_values("seed_rule", kind="stable")
    ablation.to_csv(output_dir / "seed_custom_ablation.csv", index=False)

    objective_matrix = paper.pivot(
        index="seed_rule", columns="selection_rule", values="objective_mean"
    ).reindex(index=seed_rules, columns=paper_selections)
    seed_labels = {
        "largest_snake_span": "grootste snake-span",
        "smallest_snake_span": "kleinste snake-span",
    }
    selection_labels = {
        "maximum_location_overlap": "maximale locatie-overlap",
        "minimum_span_expansion": "minimale spanuitbreiding",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    image = axes[0].imshow(objective_matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    axes[0].set_xticks(
        range(len(paper_selections)),
        [
            selection_labels.get(selection, selection.replace("_", " "))
            for selection in paper_selections
        ],
        rotation=25,
        ha="right",
    )
    axes[0].set_yticks(
        range(len(seed_rules)),
        [
            seed_labels.get(seed_rule, seed_rule.replace("_", " "))
            for seed_rule in seed_rules
        ],
    )
    axes[0].set_title("Seedselectie uit paper (2x2): doelfunctie")
    for row_index in range(len(seed_rules)):
        for column_index in range(len(paper_selections)):
            value = objective_matrix.iloc[row_index, column_index]
            axes[0].text(
                column_index,
                row_index,
                f"{value:.3g}",
                ha="center",
                va="center",
                color="white" if image.norm(value) > 0.55 else "black",
            )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    positions = list(range(len(ablation)))
    axes[1].bar(
        positions,
        ablation["objective_delta_custom_minus_immediate_mean"],
        yerr=ablation["objective_delta_custom_minus_immediate_ci95"].fillna(0.0),
        capsize=4,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(
        positions,
        [
            seed_labels.get(seed_rule, seed_rule.replace("_", " "))
            for seed_rule in ablation["seed_rule"]
        ],
        rotation=25,
        ha="right",
    )
    axes[1].set_title("Tweestaps-lookahead minus directe keuze")
    axes[1].set_ylabel("Verschil in doelfunctie (lager is beter)")
    artifacts = [
        "seed_paper_2x2.csv",
        "seed_custom_ablation.csv",
        *save_figure(figure, output_dir, "seed_factorial"),
    ]
    return artifacts, {
        "paper_selection_rules": paper_selections,
        "custom_rule": custom_rule,
        "immediate_ablation_rule": immediate_rule,
        "delta_sign": "custom_minus_immediate",
        "uncertainty_for_single_seed": "NA",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Optioryx factorial and sensitivity benchmark suites."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing runs.csv and suite_manifest.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV, PNG and PDF artifacts")
    parser.add_argument("--suite-type", required=True, choices=["sa", "ga", "seed"])
    parser.add_argument(
        "--baseline-config",
        help="GA baseline config_id; otherwise the unique OFAT centre is detected",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.baseline_config and args.suite_type != "ga":
        parser.error("--baseline-config is only valid with --suite-type ga")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = input_dir / "runs.csv"
    manifest_path = input_dir / "suite_manifest.csv"
    try:
        runs = pd.read_csv(runs_path)
        manifest = expand_manifest(pd.read_csv(manifest_path))
        seed_totals = build_seed_totals(runs)
        if args.suite_type == "sa":
            data = attach_manifest(seed_totals, manifest, "simulated_annealing")
            artifacts, details = analyze_sa(data, output_dir)
        elif args.suite_type == "ga":
            ga_manifest = manifest[manifest["solver"] == "genetic"].copy()
            data = attach_manifest(seed_totals, manifest, "genetic")
            convergence_path = input_dir / "convergence.csv"
            convergence = pd.read_csv(convergence_path) if convergence_path.exists() else None
            artifacts, details = analyze_ga(
                data, ga_manifest, convergence, output_dir, args.baseline_config
            )
        else:
            data = attach_manifest(seed_totals, manifest, "seed_selection")
            artifacts, details = analyze_seed(data, output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))
        return 2

    protocol = {
        "suite_type": args.suite_type,
        "statistical_unit": "complete_instance_per_solver_config_seed",
        "wave_aggregation": "sum_objective_and_runtime_before_seed_statistics",
        "dispersion": "sample_standard_deviation_ddof_1",
        "confidence_interval": "two_sided_95_percent_student_t; NA_when_n_less_than_2",
        "inputs": {
            "runs.csv": sha256_file(runs_path),
            "suite_manifest.csv": sha256_file(manifest_path),
        },
        "artifacts": [*artifacts, "analysis_protocol.json"],
        "details": details,
    }
    convergence_path = input_dir / "convergence.csv"
    if convergence_path.exists():
        protocol["inputs"]["convergence.csv"] = sha256_file(convergence_path)
    (output_dir / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
