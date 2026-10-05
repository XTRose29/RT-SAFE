"""Deterministic waypoint annotations for native RGB policy observations."""

from __future__ import annotations
import io
import math
from typing import Sequence
from PIL import Image, ImageDraw


def project(
    point: Sequence[float],
    camera_position: Sequence[float],
    rotation: Sequence[float],
    width: int,
    height: int,
    fov_degrees: float,
) -> tuple[float, float] | None:
    pitch, yaw, _ = map(math.radians, rotation)
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    forward = (cp * cy, cp * sy, sp)
    right = (-sy, cy, 0)
    up = (-sp * cy, -sp * sy, cp)
    delta = [a - b for a, b in zip(point, camera_position)]
    depth = sum(a * b for a, b in zip(delta, forward))
    if depth <= 1:
        return None
    focal = width / (2 * math.tan(math.radians(fov_degrees) / 2))
    return (
        width / 2 + focal * sum(a * b for a, b in zip(delta, right)) / depth,
        height / 2 - focal * sum(a * b for a, b in zip(delta, up)) / depth,
    )


def annotate_waypoints(
    jpeg: bytes, request: dict, waypoints: dict[str, list[float]]
) -> bytes:
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    draw = ImageDraw.Draw(image)
    source = request["camera_sources"][0]
    width, height = image.size
    for label, point in waypoints.items():
        pixel = project(
            point,
            source["location_cm"],
            source["rotation_degrees"],
            width,
            height,
            request["fov_degrees"],
        )
        if pixel is None:
            continue
        x, y = pixel
        if not (14 <= x < width - 14 and 14 <= y < height - 14):
            continue
        draw.ellipse(
            (x - 12, y - 12, x + 12, y + 12), fill="#ed321c", outline="white", width=2
        )
        box = draw.textbbox((0, 0), label)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text((x - tw / 2, y - th / 2 - box[1]), label, fill="white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
