#!/usr/bin/env python3
"""Fail when a raw benchmark demo is too dark to inspect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def measure(path: Path, overlay_height: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to decode middle frame: {path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scene = gray[max(0, min(overlay_height, gray.shape[0] - 1)) :]
    return {
        "path": str(path),
        "frame_count": frame_count,
        "scene_mean": round(float(scene.mean()), 3),
        "scene_p50": int(np.percentile(scene, 50)),
        "scene_p90": int(np.percentile(scene, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--overlay-height", type=int, default=120)
    parser.add_argument("--minimum-mean", type=float, default=55.0)
    args = parser.parse_args()
    reports = [measure(path, args.overlay_height) for path in args.videos]
    print(json.dumps(reports, indent=2))
    failures = [item for item in reports if item["scene_mean"] < args.minimum_mean]
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
