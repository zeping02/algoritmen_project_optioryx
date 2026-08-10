"""Resume an interrupted retrospective budget study without overwriting runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from retrospective_budget_common import WAVES, repo_root, sha256_file
from retrospective_budget_run import atomic_write_json, utc_now, validate_completed_wave


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate completed waves and run only missing waves from a recorded study plan."
    )
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()

    root = repo_root()
    study_dir = Path(args.study_dir).expanduser().resolve()
    manifest_path = study_dir / "study_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing study manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") not in {"failed", "running"}:
        raise SystemExit(f"study state is not resumable: {manifest.get('state')!r}")

    commands = manifest.get("commands")
    if not isinstance(commands, list) or len(commands) != len(WAVES):
        raise SystemExit("manifest does not contain the twelve original commands")

    previous_failure = manifest.pop("failure", None)
    manifest["state"] = "running"
    manifest["resumption"] = {
        "started_utc": utc_now(),
        "reason": "The controlling process was interrupted after W000009 had completed.",
        "previous_failure": previous_failure,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_write_json(manifest_path, manifest)

    completed_waves: list[str] = []
    try:
        for index, ((wave_id, _), raw_command) in enumerate(zip(WAVES, commands), start=1):
            wave_dir = study_dir / wave_id
            protocol_path = wave_dir / "protocol.json"
            if protocol_path.is_file():
                validate_completed_wave(wave_dir, wave_id)
                print(f"[{index}/{len(WAVES)}] Reusing validated {wave_id}", flush=True)
            else:
                if wave_dir.exists():
                    raise RuntimeError(f"refusing incomplete existing directory: {wave_dir}")
                if not isinstance(raw_command, list) or not all(
                    isinstance(part, str) for part in raw_command
                ):
                    raise RuntimeError(f"malformed recorded command for {wave_id}")
                print(f"[{index}/{len(WAVES)}] Running missing wave {wave_id}", flush=True)
                completed = subprocess.run(raw_command, cwd=root, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"delegated runner failed for {wave_id} with exit code "
                        f"{completed.returncode}"
                    )
                validate_completed_wave(wave_dir, wave_id)
            completed_waves.append(wave_id)
            manifest["completed_waves"] = completed_waves.copy()
            atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["state"] = "failed"
        manifest["finished_utc"] = utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(manifest_path, manifest)
        raise

    manifest["state"] = "complete"
    manifest["finished_utc"] = utc_now()
    manifest["resumption"]["finished_utc"] = manifest["finished_utc"]
    atomic_write_json(manifest_path, manifest)
    print(f"Study complete: {study_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
