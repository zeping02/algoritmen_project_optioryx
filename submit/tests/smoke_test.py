import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    workdir = Path(args.workdir)
    benchmark_dir = workdir / "benchmark"

    if workdir.exists():
        shutil.rmtree(workdir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            args.cli,
            "benchmark",
            "--input",
            args.instance,
            "--out-dir",
            str(benchmark_dir),
            "--suite",
            "smoke",
        ],
        cwd=repo_root,
    )

    run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_results.py"),
            "--input-dir",
            str(benchmark_dir),
            "--output-dir",
            str(benchmark_dir / "analysis"),
        ],
        cwd=repo_root,
    )

    run(
        [
            sys.executable,
            str(repo_root / "scripts" / "plot_results.py"),
            "--input-dir",
            str(benchmark_dir),
            "--output-dir",
            str(benchmark_dir / "plots"),
        ],
        cwd=repo_root,
    )

    expected_files = [
        benchmark_dir / "runs.csv",
        benchmark_dir / "convergence.csv",
        benchmark_dir / "analysis" / "summary.csv",
        benchmark_dir / "plots" / "runtime.png",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        print("Missing smoke-test outputs:")
        for path in missing:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
