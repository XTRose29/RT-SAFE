#!/usr/bin/env python3
"""Small UnrealCV smoke probe for the realtime SimWorld UE server."""

import argparse
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The realtime benchmark does not use the optional city-generation modules
# imported by SimWorld's package initializer.
_simworld = types.ModuleType("simworld")
_simworld.__path__ = [str(REPO_ROOT / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", _simworld)

from base.rt_unrealcv import RTUnrealCV  # noqa: E402


def _image_stats(img: np.ndarray) -> str:
    return (
        f"shape={tuple(img.shape)} min={int(img.min())} max={int(img.max())} "
        f"mean={float(img.mean()):.2f} std={float(img.std()):.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9006)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=10)
    parser.add_argument("--location-query-rounds", type=int, default=5)
    parser.add_argument("--location-query-limit", type=int, default=32)
    parser.add_argument("--out", default="/tmp/ue_smoke_frame.png")
    args = parser.parse_args()

    print(f"[smoke] connecting to {args.ip}:{args.port}", flush=True)
    ue = RTUnrealCV(port=args.port, ip=args.ip)

    cameras = ue.get_cameras()
    print(f"[smoke] cameras: {cameras}", flush=True)

    objects = list(ue.get_objects())
    print(f"[smoke] objects: count={len(objects)} sample={objects[:10]}", flush=True)

    img = ue.get_image(args.camera_id, "lit", mode="direct")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"[smoke] image: {_image_stats(img)} saved={args.out}", flush=True)

    if img.std() < 1.0:
        raise RuntimeError("rendered image appears blank")

    query_names = [str(name) for name in objects[: args.location_query_limit]]
    for round_idx in range(args.location_query_rounds):
        if query_names:
            locs = ue.get_location_batch(query_names)
            print(
                f"[smoke] location batch {round_idx + 1}/{args.location_query_rounds}: "
                f"count={len(locs)} first={locs[0].round(2).tolist()}",
                flush=True,
            )
        time.sleep(0.05)

    ue.set_mode("sync", tick_interval=0.05)
    for tick_idx in range(args.ticks):
        ue.tick()
        if (tick_idx + 1) % 5 == 0 or tick_idx == args.ticks - 1:
            print(f"[smoke] ticked {tick_idx + 1}/{args.ticks}", flush=True)
        time.sleep(0.02)

    img2 = ue.get_image(args.camera_id, "lit", mode="direct")
    print(f"[smoke] image after ticks: {_image_stats(img2)}", flush=True)

    ue.disconnect()
    print("[smoke] ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
