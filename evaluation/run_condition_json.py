#!/usr/bin/env python3
"""Run one qwen3vl8b setting-sweep condition JSON against a live UE server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_qwen3vl8b_setting_sweep import REPO_ROOT, run_condition_with_unrealcv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition_json")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ue-ip", default="127.0.0.1")
    parser.add_argument("--ue-port", type=int, default=9006)
    parser.add_argument("--inter-run-sleep", type=float, default=0.0)
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    condition_path = Path(args.condition_json)
    if not condition_path.is_absolute():
        condition_path = REPO_ROOT / condition_path
    with condition_path.open("r", encoding="utf-8") as f:
        condition = json.load(f)

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path

    record = run_condition_with_unrealcv(condition, summary_path, args)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if record.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
