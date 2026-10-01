#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from base.rt_unrealcv import RTUnrealCV


client = RTUnrealCV(port=9001, ip="127.0.0.1")
output = Path("results/camera_probe")
output.mkdir(parents=True, exist_ok=True)
print("cameras", client.get_cameras())
print("spawn", client.client.request("vset /cameras/spawn"))
print("cameras_after_spawn", client.get_cameras())
for index, rotation in enumerate(
    [(-90, 0, 0), (0, -90, 0), (0, 90, 0), (90, 0, 0)]
):
    client.set_camera_location(1, (-10000, 0, 7500))
    client.set_camera_rotation(1, rotation)
    client.set_camera_resolution(1, (1280, 720))
    print(index, rotation, client.get_camera_location(1), client.get_camera_rotation(1))
    cv2.imwrite(str(output / f"camera_{index}.png"), client.get_image(1, "lit"))
client.disconnect()
