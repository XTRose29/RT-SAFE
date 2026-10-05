"""Inspect authored route samples against the actual owned map collision."""

import math
import unreal

from .manifest import validate_manifest
from .unreal_runtime import _world, _vec, _xyz, _ground


def audit_map(manifest, spacing_cm=100):
    validate_manifest(manifest)
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise RuntimeError("End PIE before auditing the map")
    world = _world(True)
    if world.get_path_name().split(".")[0] != manifest["map_path"]:
        raise RuntimeError("Load the manifest's owned map before auditing")
    reports = []
    radius = manifest["agent"]["capsule_radius_cm"]
    half_height = manifest["agent"]["capsule_half_height_cm"]
    for task in manifest["tasks"]:
        route = task.get("reference_route_cm") or [task["start_cm"], task["goal_cm"]]
        samples = []
        for a, b in zip(route, route[1:]):
            count = max(1, math.ceil(math.dist(a[:2], b[:2]) / spacing_cm))
            samples.extend(
                [
                    [a[j] + (b[j] - a[j]) * i / count for j in range(3)]
                    for i in range(count)
                ]
            )
        samples.append(route[-1])
        checked = []
        for point in samples:
            row = {"authored_position_cm": point}
            try:
                ground = _ground(world, point)
                center = _vec([point[0], point[1], ground + half_height + 4])
                hits = (
                    unreal.SystemLibrary.capsule_trace_multi(
                        world,
                        center,
                        center + unreal.Vector(0, 0, 0.1),
                        radius * 0.98,
                        half_height * 0.98,
                        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                        False,
                        [],
                        unreal.DrawDebugTrace.NONE,
                        True,
                    )
                    or []
                )
                blockers = [
                    hit.to_tuple()[9].get_path_name()
                    for hit in hits
                    if hit.to_tuple()[0] and hit.to_tuple()[1]
                ]
                row.update(
                    ground_z_cm=ground,
                    capsule_center_cm=_xyz(center),
                    blocking_actors=blockers,
                    clear=not blockers,
                )
            except Exception as error:
                row.update(clear=False, error=str(error))
            checked.append(row)
        reports.append(
            {
                "task_id": task["id"],
                "all_samples_clear": all(row["clear"] for row in checked),
                "samples": checked,
            }
        )
    return {
        "schema": "rtsafe-native-route-audit-v1",
        "map": world.get_path_name(),
        "sample_spacing_cm": spacing_cm,
        "sampling_only": True,
        "fixtures_included": False,
        "certified_shortest_paths": False,
        "tasks": reports,
    }
