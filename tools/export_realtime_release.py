#!/usr/bin/env python3
"""Export the original five-map benchmark without website/map-transfer content.

Only tracked files from explicit roots are copied. The export has no Git
history, runtime binaries, generated results, or local environment files.
Maintainers can make a parentless release commit from this directory so that
GitHub's automatic source archives have the same scope as the attachment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    "SimWorld", "base", "benchmark", "data", "evaluation", "llm", "manager",
    "online_rl", "sample", "tests", "tools", "utils", "examples",
}
FILES = {
    ".env.example", ".gitattributes", ".gitignore", "config.yaml", "pytest.ini",
    "requirements.txt", "requirements-dev.txt", "requirements.lock.txt",
    "requirements-openai-rollout.txt", "requirements-serving.txt", "run_scenario.py",
    "sample_tasks.py", "visualize_scenarios.py", "visualize_waypoints_obstacles.py",
    "docs/PROTOCOL.md", "docs/REALTIME_ADDON.md", "docs/source-manifest.json",
    "docs/release-edits.json", "scripts/check-release.py",
}


def included(name: str) -> bool:
    path = Path(name)
    # Never publish the transfer adapter, its tests, or any NYC-named artifact.
    if "nyc" in name.lower() or "map_transfer" in name.lower():
        return False
    return (path.parts[0] in DIRECTORIES or name in FILES
            or name.startswith("validation/realtime-")
            or name.startswith("release/realtime/"))


def export(destination: Path) -> dict:
    if destination.exists():
        raise ValueError("Output must be a new directory")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    sources = {name: ROOT / name for name in tracked if name and included(name)}
    for name in tracked:
        if name.startswith("release/realtime/"):
            sources[name.removeprefix("release/realtime/")] = ROOT / name
    if "README.md" not in sources:
        raise ValueError("Missing release documentation templates")
    destination.mkdir(parents=True)
    inventory = {}
    for name, source in sorted(sources.items()):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Not a regular source file: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        target.chmod(source.stat().st_mode & 0o777)
        inventory[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    record = {
        "scope": "Original RT10/RT12/RT15/RT18/RT20 benchmark only",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_has_local_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "files": inventory,
    }
    (destination / "SOURCE_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = export(args.output.resolve())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Exported {len(record['files'])} files to {args.output}")


if __name__ == "__main__":
    main()
