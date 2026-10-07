#!/usr/bin/env python3
"""Check tracked release content without printing possible secret values."""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(rb"\bsk-(?:proj-|or-v1-|ant-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]
FORBIDDEN_PARTS = {".secrets", ".venv", "node_modules", "__pycache__", "runtime", "models"}


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    names = [s.decode() for s in result.stdout.split(b"\0") if s]
    if not names:
        raise SystemExit("No tracked files; stage the intended release before checking it.")
    errors = []
    total = 0
    for name in names:
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"{name}: symlinks are not allowed in the release")
            continue
        if not path.is_file():
            errors.append(f"{name}: tracked file is missing")
            continue
        if FORBIDDEN_PARTS.intersection(path.relative_to(ROOT).parts):
            errors.append(f"{name}: private/generated directory is tracked")
        if path.name.startswith(".env") and path.name != ".env.example":
            errors.append(f"{name}: environment file is tracked")
        if path.name.endswith("_api.txt") or path.name in {"auth.json", "credentials.json"}:
            errors.append(f"{name}: credential filename is tracked")
        data = path.read_bytes()
        total += len(data)
        if len(data) >= 95 * 1024 * 1024:
            errors.append(f"{name}: exceeds the release's 95 MiB file limit")
        if b"\0" in data[:8192]:
            continue
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            errors.append(f"{name}: possible credential pattern (value withheld)")
        if re.search(rb"^<{7} |^>{7} ", data, re.M):
            errors.append(f"{name}: unresolved merge marker")
        if re.search(rb"/(?:data|home)/(?:rose|murray|zhaoxu)/", data):
            errors.append(f"{name}: machine-specific workspace path")
        if path.suffix == ".json":
            try:
                json.loads(data)
            except (ValueError, UnicodeDecodeError):
                errors.append(f"{name}: invalid JSON")
    # These are the portable release's primary documentation links.
    for name in ["README.md", "CONTRIBUTING.md", *[str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md")]]:
        path = ROOT / name
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#")[0]
            if not (path.parent / target).exists():
                errors.append(f"{name}: missing linked file {target}")
    print(f"Checked {len(names)} tracked files ({total / 1024**2:.1f} MiB).")
    if errors:
        print("\n".join(errors))
        return 1
    print("Release-content checks passed. Pattern scanning is not a comprehensive security audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
