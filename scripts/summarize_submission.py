import argparse
import csv
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def run_validation(cli: Path, input_path: Path, submission_path: Path) -> tuple[bool, str, float | None]:
    command = [
        str(cli),
        "validate-submission",
        "--input",
        str(input_path),
        "--submission",
        str(submission_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    match = re.search(r"Local score:\s*([0-9.eE+-]+)", output)
    score = float(match.group(1)) if match else None
    return completed.returncode == 0, output, score


def summarize_rows(submission_path: Path) -> dict[str, object]:
    rows = []
    with submission_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    orders_by_list: dict[str, list[str]] = defaultdict(list)
    wave_by_list: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        orders_by_list[row["LIST_ID"]].append(row["ORDER_ID"])
        wave_by_list[row["LIST_ID"]].add(row["WAVE_ID"])

    sizes = [len(orders) for orders in orders_by_list.values()]
    distribution = Counter(sizes)
    mixed_wave_lists = sum(1 for waves in wave_by_list.values() if len(waves) > 1)

    return {
        "rows": len(rows),
        "lists": len(sizes),
        "min_batch_size": min(sizes) if sizes else 0,
        "max_batch_size": max(sizes) if sizes else 0,
        "avg_batch_size": sum(sizes) / len(sizes) if sizes else 0.0,
        "mixed_wave_lists": mixed_wave_lists,
        "size_distribution": distribution,
    }


def write_summary_csv(path: Path, summary: dict[str, object], valid: bool, score: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    distribution: Counter[int] = summary["size_distribution"]  # type: ignore[assignment]
    with path.open("w", newline="") as handle:
        fieldnames = [
            "valid",
            "score",
            "rows",
            "lists",
            "min_batch_size",
            "max_batch_size",
            "avg_batch_size",
            "mixed_wave_lists",
            "batch_size",
            "batch_size_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if distribution:
            for batch_size in sorted(distribution):
                writer.writerow(
                    {
                        "valid": valid,
                        "score": "" if score is None else score,
                        "rows": summary["rows"],
                        "lists": summary["lists"],
                        "min_batch_size": summary["min_batch_size"],
                        "max_batch_size": summary["max_batch_size"],
                        "avg_batch_size": f"{summary['avg_batch_size']:.6f}",
                        "mixed_wave_lists": summary["mixed_wave_lists"],
                        "batch_size": batch_size,
                        "batch_size_count": distribution[batch_size],
                    }
                )
        else:
            writer.writerow({"valid": valid, "score": "" if score is None else score})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cli = Path(args.cli)
    input_path = Path(args.input)
    submission_path = Path(args.submission)
    output_path = Path(args.output)

    valid, validation_output, score = run_validation(cli, input_path, submission_path)
    sys.stdout.write(validation_output)
    summary = summarize_rows(submission_path)
    write_summary_csv(output_path, summary, valid, score)
    print(f"Summary written to: {output_path}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
