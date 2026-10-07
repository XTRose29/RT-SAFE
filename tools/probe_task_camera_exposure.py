#!/usr/bin/env python3
"""Capture exposure variants from an existing first-person task camera.

The script never moves the camera or any scene actor.  It is intended to be
run after pausing a rollout so that every exposure sample observes the exact
same full-scene task state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

# Avoid importing optional asset-retrieval dependencies from simworld.__init__;
# this probe only needs the UnrealCV communicator package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
simworld = types.ModuleType("simworld")
simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", simworld)

from base.rt_unrealcv import RTUnrealCV


def _camera_ids(value: object) -> list[int]:
    tokens = str(value).replace(",", " ").split()
    ids = []
    for token in tokens:
        try:
            ids.append(int(token))
        except ValueError:
            continue
    # SimWorld's UnrealCV build returns camera component names instead of
    # numeric ids.  Their positions in the response are the usable ids.
    return ids or list(range(len(tokens)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--bias", type=float, action="append")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unrealcv = RTUnrealCV(port=args.port, ip=args.host)
    ids = _camera_ids(unrealcv.get_cameras())
    if not ids:
        raise RuntimeError("No UnrealCV cameras are available")
    camera_id = args.camera_id if args.camera_id is not None else max(ids)
    if camera_id not in ids:
        raise RuntimeError(f"Camera {camera_id} is unavailable; found {ids}")

    unrealcv.set_camera_resolution(camera_id, (args.width, args.height))
    unrealcv.client.request("vrun r.DefaultFeature.AutoExposure 1")
    samples = []
    contact_frames = []
    for bias in args.bias or [0.0, 0.5, 1.0, 1.5, 2.0]:
        response = unrealcv.client.request(
            f"vrun r.DefaultFeature.AutoExposure.Bias {bias:g}"
        )
        time.sleep(max(0.0, args.settle_seconds))
        frame = unrealcv.get_image(camera_id, "lit", "direct")
        path = output_dir / f"bias_{bias:g}.png"
        cv2.imwrite(str(path), frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(
            {
                "bias": bias,
                "path": str(path),
                "mean": round(float(gray.mean()), 3),
                "p01": int(np.percentile(gray, 1)),
                "p50": int(np.percentile(gray, 50)),
                "p99": int(np.percentile(gray, 99)),
                "clipped_black_fraction": round(float((gray <= 3).mean()), 6),
                "clipped_white_fraction": round(float((gray >= 252).mean()), 6),
                "response": response,
            }
        )
        labelled = frame.copy()
        cv2.rectangle(labelled, (0, 0), (210, 38), (0, 0, 0), -1)
        cv2.putText(
            labelled,
            f"exposure bias {bias:g}",
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        contact_frames.append(labelled)

    cv2.imwrite(str(output_dir / "contact_sheet.png"), cv2.hconcat(contact_frames))
    summary = {
        "camera_id": camera_id,
        "available_camera_ids": ids,
        "camera_location": unrealcv.get_camera_location(camera_id),
        "camera_rotation": unrealcv.get_camera_rotation(camera_id),
        "resolution": [args.width, args.height],
        "samples": samples,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
