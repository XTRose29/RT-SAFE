#!/usr/bin/env python3
"""Measure raw UnrealCV brightness at a benchmark intersection.

This intentionally records pixel statistics before any video post-processing,
so a packaged runtime can be compared against another one at the same camera
pose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from base.rt_unrealcv import RTUnrealCV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--x", type=float, default=-10000.0)
    parser.add_argument("--y", type=float, default=-700.0)
    parser.add_argument("--z", type=float, default=4200.0)
    parser.add_argument("--bias", type=float, action="append", default=None)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--settle-sim-seconds", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    unrealcv = RTUnrealCV(port=args.port, ip=args.host)
    existing = str(unrealcv.get_cameras()).split()
    response = unrealcv.client.request("vset /cameras/spawn")
    if isinstance(response, str) and response.lower().startswith("error"):
        raise RuntimeError(response)
    camera_id = len(existing)
    unrealcv.set_camera_resolution(camera_id, (1280, 720))
    unrealcv.set_camera_location(camera_id, (args.x, args.y, args.z))
    unrealcv.set_camera_rotation(camera_id, (-90.0, 0.0, 0.0))
    unrealcv.set_camera_fov(camera_id, 70.0)

    samples = []
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for bias in args.bias or [0.0, 2.0, 4.0, 6.0]:
        responses = {
            "auto_exposure": unrealcv.client.request(
                "vrun r.DefaultFeature.AutoExposure 1"
            ),
            "bias": unrealcv.client.request(
                f"vrun r.DefaultFeature.AutoExposure.Bias {bias:g}"
            ),
        }
        if args.settle_sim_seconds > 0:
            unrealcv.advance_simulation_time(args.settle_sim_seconds)
        time.sleep(max(0.0, args.settle_seconds))
        frame = unrealcv.get_image(camera_id, "lit", "direct")
        image_path = None
        if output_dir is not None:
            image_path = output_dir / f"topdown_bias_{bias:g}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Could not write {image_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(
            {
                "bias": bias,
                "mean": round(float(gray.mean()), 3),
                "p50": int(np.percentile(gray, 50)),
                "p90": int(np.percentile(gray, 90)),
                "image_path": str(image_path) if image_path else None,
                "responses": responses,
            }
        )
    summary = {
        "camera_id": camera_id,
        "camera_location": [args.x, args.y, args.z],
        "samples": samples,
    }
    if output_dir is not None:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
