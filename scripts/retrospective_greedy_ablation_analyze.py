"""Build a cumulative greedy development path from recorded tuning milestones.

The selected configurations add phases cumulatively, but some phases bundle
multiple mechanisms and are therefore not isolated causal ablations. Objectives
are rounded values recorded during full-instance tuning. Runtime values are
historical single-run order-of-magnitude measurements and are never treated as
replicated timing estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence


STAGES: tuple[tuple[str, str, str], ...] = (
    ("greedy_fast", "besparingsbasis", "savings"),
    ("greedy_best_plus", "+ lokale paarzoektocht", "local_search"),
    ("beam_eliminate", "+ eliminatie en post-LS", "elimination_post_ls"),
    ("final_plus2_regret", "+ pair- en final-LS", "pair_final_ls"),
    ("final_triple3", "+ triple-rebuild", "triple_rebuild"),
    ("final_combo_triple_ultra11", "+ triplecyclus en combinatievenster", "triple_cycle_bundle"),
    ("final_tail_pair3", "+ afsluitende pair-rebuild", "tail_pair_rebuild"),
)


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path, root: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing benchmark table: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"version", "config", "score", "routes", "runtime_seconds", "valid", "notes"}
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames), "benchmark schema differs")
        indexed = {str(row["version"]): row for row in reader}
    output: list[dict[str, Any]] = []
    baseline_score: float | None = None
    previous_score: float | None = None
    for index, (version, label, mechanism) in enumerate(STAGES, start=1):
        require(version in indexed, f"missing required milestone: {version}")
        source = indexed[version]
        require(str(source["valid"]).casefold() == "yes", f"invalid milestone: {version}")
        config_path = (root / str(source["config"])).resolve()
        require(config_path.is_file(), f"missing config for {version}: {config_path}")
        score = float(source["score"])
        runtime = float(source["runtime_seconds"])
        routes = int(source["routes"])
        require(math.isfinite(score) and math.isfinite(runtime), f"non-finite milestone: {version}")
        require(runtime > 0.0, f"non-positive runtime: {version}")
        if baseline_score is None:
            baseline_score = score
        marginal = 0.0 if previous_score is None else previous_score - score
        cumulative = baseline_score - score
        output.append(
            {
                "stage_index": index,
                "version": version,
                "label": label,
                "mechanism": mechanism,
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "objective": score,
                "routes": routes,
                "runtime_seconds_single": runtime,
                "marginal_gain_abs": marginal,
                "marginal_gain_pct_previous": 0.0 if previous_score is None else 100.0 * marginal / previous_score,
                "cumulative_gain_abs": cumulative,
                "cumulative_gain_pct_baseline": 100.0 * cumulative / baseline_score,
                "share_of_final_gain_pct": 0.0,
                "notes": source["notes"],
            }
        )
        previous_score = score
    final_gain = float(output[-1]["cumulative_gain_abs"])
    require(final_gain > 0.0, "final milestone does not improve the baseline")
    for row in output:
        row["share_of_final_gain_pct"] = 100.0 * float(row["marginal_gain_abs"]) / final_gain
    return output


def make_plot(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    x = list(range(1, len(rows) + 1))
    short = [
        "basis", "lokaal", "eliminatie", "pair", "triple", "combi", "slot-pair"
    ]
    cumulative = [float(row["cumulative_gain_pct_baseline"]) for row in rows]
    marginal = [float(row["share_of_final_gain_pct"]) for row in rows]
    axes[0].plot(x, cumulative, marker="o", color="#1f77b4")
    axes[0].set_ylabel("cumulatieve afstandswinst (%)")
    axes[0].set_title("(a) Cumulatieve kwaliteitswinst")
    axes[1].bar(x[1:], marginal[1:], color="#4c78a8")
    axes[1].set_ylabel("aandeel in totale winst (%)")
    axes[1].set_title("(b) Marginale bijdrage per fase")
    for axis in axes:
        axis.set_xticks(x, short, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}".replace(".", ",")))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return matplotlib.__version__


def run(args: argparse.Namespace, root: Path) -> int:
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if os.path.lexists(out_dir):
        raise ValueError(f"output directory already exists: {out_dir}")
    rows = load_rows(input_path, root)
    out_dir.mkdir(parents=True, exist_ok=False)
    summary_path = out_dir / "greedy_ablation.csv"
    plot_path = out_dir / "greedy_ablation.png"
    write_csv(summary_path, rows)
    matplotlib_version = make_plot(plot_path, rows)
    protocol = {
        "schema_version": 1,
        "state": "complete",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "selection": "seven cumulative, valid full-instance development milestones",
        "quality": "rounded objective recorded for each full-instance tuning milestone",
        "runtime_limitation": "one historical measurement per configuration; descriptive order of magnitude only",
        "causal_scope": "marginal changes describe this cumulative development path; bundled phases are not isolated causal ablations",
        "adaptive_tuning_limitation": "all mechanisms were developed and selected on the same instance; no held-out validation",
        "matplotlib_version": matplotlib_version,
        "outputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in (summary_path, plot_path, plot_path.with_suffix(".pdf"))
        },
    }
    (out_dir / "analysis_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Ablation summary: {summary_path}")
    print(f"Ablation figure:  {plot_path.with_suffix('.pdf')}")
    return 0


def parser(root: Path) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--input", default=str(root / "results" / "benchmark_results.csv"))
    result.add_argument("--out-dir", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    root = root_dir()
    args = parser(root).parse_args(argv)
    try:
        return run(args, root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
