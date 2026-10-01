#!/usr/bin/env bash
# Creates only a private remote. Run from any directory after gh auth login.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
gh auth status >/dev/null
destination="${1:-}"
if [[ -z "$destination" ]]; then
  owner="$(gh api user --jq .login)"
  destination="$owner/RT-SAFE"
fi
if [[ ! "$destination" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo 'Use an explicit OWNER/REPOSITORY destination.' >&2
  exit 2
fi
if git remote get-url origin >/dev/null 2>&1; then
  echo 'An origin already exists. Inspect it before creating another repository.' >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Commit the reviewed release files before creating the remote.' >&2
  exit 2
fi
python3 scripts/check-release.py
gh repo create "$destination" --private --source=. --remote=origin --push \
  --description 'RT-SAFE: benchmarking agent safety in real-time embodied environments'
gh repo view "$destination" --json nameWithOwner,url,visibility
