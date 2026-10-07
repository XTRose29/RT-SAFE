#!/usr/bin/env python3
"""Plan or run the protocol-matched GPT-5.6 Sol Codex CLI matrix.

The default remains plan-only. CLI results intentionally form a separate
access-surface arm from raw Responses API results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SIMWORLD_CAMPAIGN_MODEL", "gpt-5.6-sol")
os.environ.setdefault("SIMWORLD_CAMPAIGN_PROVIDER", "codex-cli")
os.environ.setdefault("SIMWORLD_CAMPAIGN_API_MODE", "codex_cli")
os.environ.setdefault("SIMWORLD_CAMPAIGN_DIFFICULTIES", "easy")
os.environ.setdefault("SIMWORLD_CAMPAIGN_ENV_MODES", "realtime,static")
os.environ.setdefault("SIMWORLD_CAMPAIGN_EFFORTS", "low,high")
os.environ.setdefault("SIMWORLD_CAMPAIGN_INPUT_PRICE", "0")
os.environ.setdefault("SIMWORLD_CAMPAIGN_OUTPUT_PRICE", "0")

from evaluation.run_openai_frontier_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
