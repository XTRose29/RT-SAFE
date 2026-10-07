#!/usr/bin/env python3
"""Create a display-friendly copy of an off-screen UE demo.

Only the rendered scene below the fixed overlay is exposure-adjusted.  The
simulation frames, timing, actor positions, and overlay text remain unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def brighten_scene(frame: np.ndarray, overlay_height: int, gamma: float) -> np.ndarray:
    output = frame.copy()
    scene = output[max(0, overlay_height) :]
    inverse_gamma = 1.0 / gamma
    table = np.array(
        [((value / 255.0) ** inverse_gamma) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )
    output[max(0, overlay_height) :] = cv2.LUT(scene, table)
    return output


def process_video(source: Path, destination: Path, overlay_height: int, gamma: float) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video: {destination}")
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(brighten_scene(frame, overlay_height, gamma))
        frames += 1
    capture.release()
    writer.release()
    if frames == 0:
        raise RuntimeError(f"No frames decoded from: {source}")
    print(f"wrote {destination} ({frames} frames at {fps:.2f} fps)")


def process_image(source: Path, destination: Path, overlay_height: int, gamma: float) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to open image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), brighten_scene(image, overlay_height, gamma)):
        raise RuntimeError(f"Unable to create image: {destination}")
    print(f"wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--overlay-height", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=2.2)
    args = parser.parse_args()
    if args.source.suffix.lower() == ".mp4":
        process_video(args.source, args.destination, args.overlay_height, args.gamma)
    else:
        process_image(args.source, args.destination, args.overlay_height, args.gamma)


if __name__ == "__main__":
    main()
