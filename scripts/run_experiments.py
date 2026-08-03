import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="benchmark suite to run; repeat for multiple suites (default: assignment)",
    )
    parser.add_argument("--instance", action="append", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    suites = args.suite or ["assignment"]
    for instance in args.instance:
        instance_path = Path(instance)
        for suite in suites:
            out_dir = out_root / instance_path.stem / suite
            command = [
                args.cli,
                "benchmark",
                "--input",
                str(instance_path),
                "--out-dir",
                str(out_dir),
                "--suite",
                suite,
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
