import json
import os
import sys
import time
import math
import random
import re
import traceback
import copy
import numpy as np
from datetime import datetime
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication
from simworld.utils.vector import Vector
from simworld.agent.vehicle import VehicleState
from simworld.agent.pedestrian import PedestrianState
from simworld.traffic.base.traffic_signal import TrafficSignalState
from simworld.config import Config
from simworld.map.map import Map
from simworld.utils.logger import Logger
from simworld.utils.traffic_utils import (
    cal_waypoints,
    extend_control_point,
    get_bezier_points,
)
try:
    from base.human_control_interface import HumanControlInterface
except ImportError:  # Optional for headless benchmark execution.
    HumanControlInterface = None
from base.rt_agent import WEDGE_RECOVERY_ALL_TASKS, WEDGE_RECOVERY_CONSECUTIVE_MOVES, WEDGE_RECOVERY_TASK_IDS
from base.rt_agent import (
    ROADWAY_OCCUPANCY_GUARD_HALF_WIDTH_CM,
    RTAgent,
    TRAFFIC_POLICY_VISUAL_ONLY,
    normalize_traffic_policy,
)
from base.rt_pedestrian import RTPedestrian
from base.rt_communicator import RTCommunicator, RTSignalTrafficCommunicator
from base.rt_traffic_controller import RTTrafficController
from base.rt_traffic_system import (
    TrafficPhaseTiming,
    apply_timing_environment_overrides,
)
from llm.rt_llm import RTLLM
from utils.route_steps import route_length_to_max_steps
from utils.task_routes import reconstruct_route_points, route_length_cm
from utils.evaluation_metrics import episode_metrics
            
CONFIG_PATH = "config.yaml"
UE_ASSET_PATH = "data/ue_assets.json"
AVG_RESPONSE_TIME = 3

# ``BP_Road_Small`` only paints zebra stripes on a subset of intersection
# arms.  The Python road graph, however, exposes every connected sidewalk pair
# as a crosswalk.  In visual-only benchmarking that asset mismatch can make a
# geometrically correct route look like jaywalking to the VLM.  These markings
# are persistent UE scene actors: they have no collision and never change path
# selection, signal timing, traffic policy, or violation semantics.  Keeping
# them in UE (instead of drawing them only on decision images) also guarantees
# that decision, action, and terminal frames show the same physical scene.
ROUTE_CROSSWALK_MARKING_ASSET = (
    '/Game/RealTimeBench/Objects/RT_Box.RT_Box_C'
)
ROUTE_CROSSWALK_MARKING_STRIPE_COUNT = 7
ROUTE_CROSSWALK_MARKING_STRIPE_WIDTH_CM = 45.0
ROUTE_CROSSWALK_MARKING_LENGTH_CM = 420.0
ROUTE_CROSSWALK_MARKING_HEIGHT_CM = 2.0
ROUTE_CROSSWALK_MARKING_BASE_SIZE_CM = 120.0

TIME_ALPHA = 0.02   # per char
TIME_BETA = 1    # encoding time
SLOMO = 1
PED_NUM_PER_EDGE = 20

SCRIPTED_PEDESTRIAN_MOTION_POLICY = {
    "version": "forward_reload_bevel_clearance_detour_in_place_recovery_v14",
    "physical_collision_enabled": True,
    "right_hand_lane_offset_cm": 200.0,
    "lane_offset_jitter_cm": 0.0,
    "observed_capsule_blocking_distance_cm": 269.0,
    "minimum_beveled_corner_center_clearance_cm": 282.84,
    "non_loop_return_lane": "opposite_right_hand_lane",
    "minimum_spawn_separation_cm": 325.0,
    "minimum_static_obstacle_separation_cm": 250.0,
    "force_overlapping_spawn_fallback": False,
    "motion_sample_interval_s": 2.0,
    "motion_threshold_cm": 25.0,
    "controller_restart_after_s": 6.0,
    "endpoint_recycle_distance_cm": 125.0,
    "waypoint_reload_strategy": "nearest_forward_segment",
    "in_place_controller_recovery_after_stalled_restarts": 2,
    "destructive_same_name_actor_recreation": False,
    "in_place_recovery_agent_clearance_cm": 325.0,
    "in_place_recovery_actor_clearance_cm": 325.0,
    "local_detour_enabled": True,
    "local_detour_after_stalled_restarts": 1,
    "local_detour_trigger_distance_cm": 450.0,
    "local_detour_static_clearance_cm": 350.0,
    "local_detour_secondary_static_clearance_cm": 100.0,
    "local_detour_forward_clearance_cm": 350.0,
    "local_detour_actor_clearance_cm": 175.0,
    "local_detour_agent_clearance_cm": 325.0,
    "local_detour_sidewalk_or_crosswalk_only": True,
    "local_detour_max_waypoints": 3,
    "controller_heartbeat_scope": "scripted_pedestrian",
    "default_benchmark_speed_cm_s": 100.0,
    "teleport_recovery": False,
}


def _env_optional_nonnegative_int(name, default=None):
    """Read an optional non-negative integer environment override."""
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == '':
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{name} must be a non-negative integer, got {raw_value!r}'
        ) from exc
    if value < 0:
        raise ValueError(
            f'{name} must be a non-negative integer, got {value}'
        )
    return value


def offset_scripted_pedestrian_route(
    route_points,
    lateral_offset_cm,
    *,
    is_loop=False,
):
    """Build constant-offset right-hand lanes with beveled turns.

    Each authored arm contributes its own shifted start and end. Adjacent arms
    are connected directly, producing a short bevel rather than a diagonal
    miter. Opposing walkers therefore retain the full lane separation along
    every arm without placing a turn waypoint beyond both arm boundaries.
    """
    points = [Vector(float(point.x), float(point.y)) for point in route_points]
    if len(points) < 2:
        return points

    offset = float(lateral_offset_cm)

    def direction(start, end):
        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return None
        return dx / length, dy / length

    segment_pairs = list(zip(points, points[1:]))
    if is_loop:
        segment_pairs.append((points[-1], points[0]))

    shifted_segments = []
    for start, end in segment_pairs:
        selected_direction = direction(start, end)
        if selected_direction is None:
            continue
        direction_x, direction_y = selected_direction
        normal_x, normal_y = direction_y, -direction_x
        shifted_segments.append((
            Vector(
                start.x + normal_x * offset,
                start.y + normal_y * offset,
            ),
            Vector(
                end.x + normal_x * offset,
                end.y + normal_y * offset,
            ),
        ))
    if not shifted_segments:
        return points

    shifted = [shifted_segments[0][0], shifted_segments[0][1]]
    for start, end in shifted_segments[1:]:
        if shifted[-1].distance(start) > 1e-6:
            shifted.append(start)
        if shifted[-1].distance(end) > 1e-6:
            shifted.append(end)
    return shifted


def build_scripted_pedestrian_patrol_loop(
    route_points,
    lateral_offset_cm,
    *,
    is_loop=False,
):
    """Build a cyclic patrol with right-hand lanes in both directions.

    An authored non-loop route must not be traversed backward on its outbound
    offset: doing so sends returning walkers head-on into outbound walkers.
    Instead, offset the reversed route independently and join both sides only
    at the two endpoints. The returned placement indices exclude the endpoint
    lane-change connector so actors begin on a longitudinal walking segment.
    """
    points = [Vector(float(point.x), float(point.y)) for point in route_points]
    if len(points) < 2:
        return points, []

    if is_loop:
        nodes = offset_scripted_pedestrian_route(
            points,
            lateral_offset_cm,
            is_loop=True,
        )
        return nodes, list(range(max(0, len(nodes) - 1)))

    outward = offset_scripted_pedestrian_route(
        points,
        lateral_offset_cm,
    )
    returning = offset_scripted_pedestrian_route(
        list(reversed(points)),
        lateral_offset_cm,
    )
    nodes = outward + returning
    endpoint_connector_index = len(outward) - 1
    placement_indices = [
        index
        for index in range(max(0, len(nodes) - 1))
        if index != endpoint_connector_index
    ]
    return nodes, placement_indices


def rotate_scripted_pedestrian_waypoints_from_position(
    waypoints,
    position,
    *,
    endpoint_tolerance_cm=125.0,
):
    """Continue a cyclic patrol from the closest forward path segment.

    ``SetWaypoints`` resets the packaged Blueprint's consumed list. Reusing the
    original list while an actor is midway around its cycle can therefore send
    it backward toward an already-consumed waypoint. This helper keeps the
    same cyclic geometry and only rotates its list so the first target is the
    end of the segment currently occupied by the actor. At a segment endpoint,
    the following target is selected.
    """
    points = [Vector(float(point.x), float(point.y)) for point in waypoints]
    if len(points) <= 1:
        return points

    current = Vector(float(position.x), float(position.y))
    best = None
    for target_index, end in enumerate(points):
        start = points[target_index - 1]
        segment_x = float(end.x - start.x)
        segment_y = float(end.y - start.y)
        length_sq = segment_x * segment_x + segment_y * segment_y
        if length_sq <= 1e-9:
            projection = 0.0
            closest = start
        else:
            projection = (
                (current.x - start.x) * segment_x
                + (current.y - start.y) * segment_y
            ) / length_sq
            projection = max(0.0, min(1.0, projection))
            closest = Vector(
                start.x + projection * segment_x,
                start.y + projection * segment_y,
            )
        candidate = (current.distance(closest), target_index)
        if best is None or candidate < best:
            best = candidate

    target_index = int(best[1])
    if current.distance(points[target_index]) <= float(endpoint_tolerance_cm):
        target_index = (target_index + 1) % len(points)
    return points[target_index:] + points[:target_index]


def build_scripted_pedestrian_recovery_detour(
    position,
    forward_waypoint,
    *,
    actor_id='',
    blocking_obstacle_position=None,
    static_obstacle_positions=(),
    other_actor_positions=(),
    agent_position=None,
    legal_point=None,
):
    """Build deterministic, physically executed waypoints around a blocker.

    The actor is never teleported. Inserted points are rejected unless they
    stay in authored pedestrian geometry, remain clear of the agent, and do
    not introduce a new static-obstacle conflict.
    """
    if position is None or forward_waypoint is None:
        return []
    start = Vector(float(position.x), float(position.y))
    target = Vector(float(forward_waypoint.x), float(forward_waypoint.y))
    direction = target - start
    direction_length = direction.length()
    if direction_length <= 1e-6:
        return []
    direction = direction * (1.0 / direction_length)
    lateral = Vector(-direction.y, direction.x)
    policy = SCRIPTED_PEDESTRIAN_MOTION_POLICY
    static_clearance = float(policy['local_detour_static_clearance_cm'])
    secondary_static_clearance = float(
        policy['local_detour_secondary_static_clearance_cm']
    )
    forward_clearance = float(policy['local_detour_forward_clearance_cm'])
    actor_clearance = float(policy['local_detour_actor_clearance_cm'])
    agent_clearance = float(policy['local_detour_agent_clearance_cm'])

    def numeric_actor_id():
        match = re.search(r'(\d+)$', str(actor_id))
        return int(match.group(1)) if match else 0

    def is_legal(point):
        return legal_point is None or bool(legal_point(point))

    def point_clearances(point, *, allow_actor_queue=False):
        if not is_legal(point):
            return None
        if (
            agent_position is not None
            and point.distance(agent_position) < agent_clearance
        ):
            return None
        static_distances = [
            point.distance(obstacle_position)
            for obstacle_position in static_obstacle_positions
        ]
        if (
            static_distances
            and min(static_distances) < secondary_static_clearance
        ):
            return None
        actor_distances = [
            point.distance(other_position)
            for other_position in other_actor_positions
        ]
        if (
            not allow_actor_queue
            and actor_distances
            and min(actor_distances) < actor_clearance
        ):
            return None
        return (
            min(static_distances) if static_distances else 1e9,
            min(actor_distances) if actor_distances else 1e9,
        )

    plans = []
    blocker = (
        Vector(
            float(blocking_obstacle_position.x),
            float(blocking_obstacle_position.y),
        )
        if blocking_obstacle_position is not None
        else None
    )
    if blocker is not None:
        relative = blocker - start
        longitudinal = relative.dot(direction)
        cross_track = relative.dot(lateral)
        blocker_ahead = (
            longitudinal >= -100.0
            and longitudinal <= direction_length + static_clearance
            and abs(cross_track) < static_clearance
        )
        if blocker_ahead:
            clearance_slot = numeric_actor_id() % 3
            bypass_clearance = static_clearance + 45.0 * clearance_slot
            for side in (-1.0, 1.0):
                desired_lateral = lateral * (side * bypass_clearance)
                current_lateral = (start - blocker).dot(lateral)
                sidestep = start + lateral * (
                    side * bypass_clearance - current_lateral
                )
                before = blocker + desired_lateral - direction * 50.0
                after = (
                    blocker
                    + desired_lateral
                    + direction * (forward_clearance + 50.0)
                )
                plan = []
                for point in (sidestep, before, after):
                    if not plan or point.distance(plan[-1]) > 25.0:
                        plan.append(point)
                plan = plan[: int(policy['local_detour_max_waypoints'])]
                clearances = [
                    point_clearances(point, allow_actor_queue=True)
                    for point in plan
                ]
                if all(item is not None for item in clearances):
                    min_static = min(item[0] for item in clearances)
                    path_length = start.distance(plan[0]) + sum(
                        first.distance(second)
                        for first, second in zip(plan, plan[1:])
                    )
                    plans.append((min_static, -path_length, side, plan))

    if not plans:
        preferred_side = -1.0 if numeric_actor_id() % 2 == 0 else 1.0
        for side in (preferred_side, -preferred_side):
            first = (
                start
                + direction * 200.0
                + lateral * (side * 250.0)
            )
            second = first + direction * 300.0
            plan = [first, second]
            clearances = [point_clearances(point) for point in plan]
            if all(item is not None for item in clearances):
                min_actor = min(item[1] for item in clearances)
                min_static = min(item[0] for item in clearances)
                path_length = start.distance(first) + first.distance(second)
                plans.append((
                    min(min_static, min_actor),
                    -path_length,
                    side,
                    plan,
                ))

    if not plans:
        return []
    plans.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return plans[0][3]


def _point_on_segment(point, start, end, tolerance_cm=1.0):
    """Return whether a task point lies on a finite ordered route edge."""
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay) <= tolerance_cm
    projection = ((px - ax) * dx + (py - ay) * dy) / length_sq
    if projection < -1e-6 or projection > 1.0 + 1e-6:
        return False
    projection = max(0.0, min(1.0, projection))
    qx, qy = ax + projection * dx, ay + projection * dy
    return math.hypot(px - qx, py - qy) <= tolerance_cm


def reconstruct_route_points_from_task_edges(task_data, tolerance_cm=1.0):
    """Build a fail-closed centerline route from the task's ordered edges.

    Some generated tasks contain a stale ``route_info.shortest_path`` which
    skips a crosswalk and connects two sidewalks straight through a roadway.
    The ordered task edges are the authoritative topology: each consecutive
    pair must share exactly one endpoint, while start/end may lie inside the
    first/last edge.
    """
    edges = list(task_data.get('edges') or [])
    if not edges:
        route_info = task_data.get('route_info') or {}
        return [list(point) for point in route_info.get('shortest_path', [])]

    start = list(task_data['start_point'])
    end = list(task_data['end_point'])
    if not _point_on_segment(start, edges[0]['node1'], edges[0]['node2'], tolerance_cm):
        raise ValueError('Task start point is outside the first ordered route edge')
    if not _point_on_segment(end, edges[-1]['node1'], edges[-1]['node2'], tolerance_cm):
        raise ValueError('Task end point is outside the last ordered route edge')

    route = [start]
    for index, edge in enumerate(edges[:-1]):
        next_edge = edges[index + 1]
        shared = []
        for endpoint in (edge['node1'], edge['node2']):
            if any(
                math.hypot(
                    float(endpoint[0]) - float(candidate[0]),
                    float(endpoint[1]) - float(candidate[1]),
                ) <= tolerance_cm
                for candidate in (next_edge['node1'], next_edge['node2'])
            ):
                shared.append(list(endpoint))
        if len(shared) != 1:
            raise ValueError(
                f'Ordered route edges {index} and {index + 1} do not share '
                f'exactly one endpoint: {shared!r}'
            )
        route.append(shared[0])
    route.append(end)
    return route


def rendered_native_crosswalk_segments(
    roads_payload,
    crosswalk_offset_cm,
    sidewalk_offset_cm,
):
    """Reconstruct native BP_Road_Small zebra centerlines from road JSON."""
    segments = []
    for road_index, road in enumerate(roads_payload.get('roads') or []):
        start = [
            float(road['start']['x']) * 100.0,
            float(road['start']['y']) * 100.0,
        ]
        end = [
            float(road['end']['x']) * 100.0,
            float(road['end']['y']) * 100.0,
        ]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        direction = [dx / length, dy / length]
        perpendicular = [-direction[1], direction[0]]
        for endpoint_name, intersection_center, inward_sign in (
            ('start', start, 1.0),
            ('end', end, -1.0),
        ):
            midpoint = [
                intersection_center[0]
                + direction[0] * float(crosswalk_offset_cm) * inward_sign,
                intersection_center[1]
                + direction[1] * float(crosswalk_offset_cm) * inward_sign,
            ]
            segments.append({
                'road_index': road_index,
                'endpoint': endpoint_name,
                'intersection_center': list(intersection_center),
                'start': [
                    midpoint[0]
                    + perpendicular[0] * float(sidewalk_offset_cm),
                    midpoint[1]
                    + perpendicular[1] * float(sidewalk_offset_cm),
                ],
                'end': [
                    midpoint[0]
                    - perpendicular[0] * float(sidewalk_offset_cm),
                    midpoint[1]
                    - perpendicular[1] * float(sidewalk_offset_cm),
                ],
            })
    return segments


def align_task_crosswalks_to_rendered_geometry(
    task_data,
    rendered_crosswalk_offset_cm,
    tolerance_cm=1.0,
    intersection_centers=None,
    rendered_crosswalk_segments=None,
):
    """Move task crosswalk corners onto the packaged road paint.

    Task files encode a crosswalk's radial setback from its intersection
    centre.  Keep that setback equal to the packaged road asset's native zebra
    centreline.  This function remains generic so a map with a genuinely
    different native setback can still be calibrated explicitly; Map1's
    packaged ``BP_Road_Small`` and task data both use 700 cm.

    The correction is topology preserving: only the two endpoints of a
    crosswalk and sidewalk endpoints sharing those corners move.  Interior
    task start/end points remain unchanged when they are still on the shortened
    sidewalk; a point in the removed corner section is translated with that
    corner so it remains on the legal sidewalk.
    """
    task = copy.deepcopy(task_data)
    edges = list(task.get('edges') or [])
    intersections = list(intersection_centers or [])
    if not intersections:
        intersections = list(
            (task.get('route_info') or {}).get('intersections') or []
        )
    target_offset = float(rendered_crosswalk_offset_cm)
    if not edges or not intersections or target_offset <= 0:
        return task

    replacements = {}
    replacement_shifts = {}
    calibration_records = []
    non_crossing_connectors = []
    for edge_index, edge in enumerate(edges):
        if edge.get('type') != 'crosswalk':
            continue
        node1 = [float(value) for value in edge['node1']]
        node2 = [float(value) for value in edge['node2']]
        midpoint = [
            (node1[0] + node2[0]) / 2.0,
            (node1[1] + node2[1]) / 2.0,
        ]
        center = min(
            intersections,
            key=lambda candidate: math.hypot(
                midpoint[0] - float(candidate[0]),
                midpoint[1] - float(candidate[1]),
            ),
        )
        radial = [
            midpoint[0] - float(center[0]),
            midpoint[1] - float(center[1]),
        ]
        current_offset = math.hypot(radial[0], radial[1])
        if current_offset <= tolerance_cm:
            raise ValueError(
                'Crosswalk midpoint coincides with its intersection centre; '
                'cannot align rendered geometry'
            )
        scale = target_offset / current_offset
        shift = [
            radial[0] * scale - radial[0],
            radial[1] * scale - radial[1],
        ]
        rendered_node1 = [node1[0] + shift[0], node1[1] + shift[1]]
        rendered_node2 = [node2[0] + shift[0], node2[1] + shift[1]]
        if rendered_crosswalk_segments is not None:
            best_native_error = float('inf')
            for native in rendered_crosswalk_segments:
                native_start, native_end = native['start'], native['end']
                direct_error = (
                    math.dist(rendered_node1, native_start)
                    + math.dist(rendered_node2, native_end)
                )
                reverse_error = (
                    math.dist(rendered_node1, native_end)
                    + math.dist(rendered_node2, native_start)
                )
                best_native_error = min(
                    best_native_error,
                    direct_error,
                    reverse_error,
                )
            if best_native_error > tolerance_cm * 2.0:
                edge['type'] = 'sidewalk'
                non_crossing_connectors.append({
                    'edge_index': edge_index,
                    'node1': list(node1),
                    'node2': list(node2),
                    'reason': 'no_matching_native_zebra',
                    'nearest_native_endpoint_error_cm': (
                        None
                        if math.isinf(best_native_error)
                        else round(best_native_error, 3)
                    ),
                })
                continue
        if abs(current_offset - target_offset) <= tolerance_cm:
            # An already aligned task still needs an explicit calibration
            # record.  Returning an empty payload made fail-closed audits
            # indistinguishable from runs that never checked the packaged
            # BP_Road_Small zebra geometry at all.
            calibration_records.append({
                'intersection_center': [float(center[0]), float(center[1])],
                'original_midpoint': midpoint,
                'rendered_midpoint': list(midpoint),
                'original_offset_cm': current_offset,
                'rendered_offset_cm': target_offset,
                'shift_cm': [0.0, 0.0],
            })
            continue
        for endpoint in (node1, node2):
            endpoint_key = (endpoint[0], endpoint[1])
            replacements[endpoint_key] = [
                round(endpoint[0] + shift[0], 6),
                round(endpoint[1] + shift[1], 6),
            ]
            replacement_shifts[endpoint_key] = list(shift)
        calibration_records.append({
            'intersection_center': [float(center[0]), float(center[1])],
            'original_midpoint': midpoint,
            'rendered_midpoint': [
                round(midpoint[0] + shift[0], 6),
                round(midpoint[1] + shift[1], 6),
            ],
            'original_offset_cm': current_offset,
            'rendered_offset_cm': target_offset,
            'shift_cm': shift,
        })

    task['edges'] = edges
    task['crosswalk_hops'] = sum(
        edge.get('type') == 'crosswalk' for edge in edges
    )

    if not replacements:
        if edges:
            task['start_edge'] = copy.deepcopy(edges[0])
            task['end_edge'] = copy.deepcopy(edges[-1])
        route_info = dict(task.get('route_info') or {})
        route_info['shortest_path'] = reconstruct_route_points_from_task_edges(
            task
        )
        task['route_info'] = route_info
        task['render_geometry_calibration'] = {
            'asset': 'BP_Road_Small',
            'crosswalks': calibration_records,
            'non_crossing_connectors': non_crossing_connectors,
        }
        return task

    original_edges = copy.deepcopy(edges)
    original_route = reconstruct_route_points_from_task_edges(task)

    def remap_exact(point):
        key = (float(point[0]), float(point[1]))
        replacement = replacements.get(key)
        return list(replacement) if replacement is not None else list(point)

    calibrated_edges = []
    for edge_index, edge in enumerate(original_edges):
        node1 = [float(value) for value in edge['node1']]
        node2 = [float(value) for value in edge['node2']]
        if edge.get('type') == 'crosswalk':
            pieces = [{
                'node1': remap_exact(node1),
                'node2': remap_exact(node2),
                'type': 'crosswalk',
            }]
        else:
            main_nodes = [list(node1), list(node2)]
            endpoint_connectors = [None, None]
            for endpoint_index, (endpoint, other) in enumerate(
                ((node1, node2), (node2, node1))
            ):
                key = (endpoint[0], endpoint[1])
                moved = replacements.get(key)
                if moved is None:
                    continue
                shift = replacement_shifts[key]
                edge_dx = other[0] - endpoint[0]
                edge_dy = other[1] - endpoint[1]
                edge_length = math.hypot(edge_dx, edge_dy)
                shift_length = math.hypot(shift[0], shift[1])
                if edge_length <= tolerance_cm or shift_length <= tolerance_cm:
                    main_nodes[endpoint_index] = list(moved)
                    continue
                outward = [edge_dx / edge_length, edge_dy / edge_length]
                alignment = abs(
                    (outward[0] * shift[0] + outward[1] * shift[1])
                    / shift_length
                )
                if alignment <= 0.25:
                    # At a ninety-degree road corner the native sidewalk first
                    # reaches its tangent point and then rounds toward the
                    # zebra.  Preserve that legal corner instead of creating a
                    # long diagonal from the task start to the moved crossing.
                    corner = [
                        round(endpoint[0] + outward[0] * shift_length, 6),
                        round(endpoint[1] + outward[1] * shift_length, 6),
                    ]
                    main_nodes[endpoint_index] = corner
                    endpoint_connectors[endpoint_index] = {
                        'node1': list(moved),
                        'node2': corner,
                        'type': 'sidewalk',
                    }
                else:
                    main_nodes[endpoint_index] = list(moved)

            pieces = []
            if endpoint_connectors[0] is not None:
                pieces.append(endpoint_connectors[0])
            pieces.append({
                'node1': main_nodes[0],
                'node2': main_nodes[1],
                'type': edge.get('type', 'sidewalk'),
            })
            if endpoint_connectors[1] is not None:
                pieces.append(endpoint_connectors[1])

            route_start = original_route[edge_index]
            route_end = original_route[edge_index + 1]
            edge_vector = [node2[0] - node1[0], node2[1] - node1[1]]
            route_vector = [
                float(route_end[0]) - float(route_start[0]),
                float(route_end[1]) - float(route_start[1]),
            ]
            if (
                edge_vector[0] * route_vector[0]
                + edge_vector[1] * route_vector[1]
            ) < 0:
                pieces = [
                    {
                        'node1': list(piece['node2']),
                        'node2': list(piece['node1']),
                        'type': piece['type'],
                    }
                    for piece in reversed(pieces)
                ]
        calibrated_edges.extend(pieces)

    edges = calibrated_edges
    task['edges'] = edges

    def keep_endpoint_on_calibrated_edge(point_key, edge_index):
        point = remap_exact(task[point_key])
        calibrated_edge = edges[edge_index]
        if _point_on_segment(
            point,
            calibrated_edge['node1'],
            calibrated_edge['node2'],
            tolerance_cm,
        ):
            task[point_key] = point
            return
        original_edge = original_edges[0 if edge_index == 0 else -1]
        for endpoint_name in ('node1', 'node2'):
            original_corner = original_edge[endpoint_name]
            moved_corner = remap_exact(original_corner)
            delta = [
                float(moved_corner[0]) - float(original_corner[0]),
                float(moved_corner[1]) - float(original_corner[1]),
            ]
            if math.hypot(delta[0], delta[1]) <= tolerance_cm:
                continue
            translated = [
                float(point[0]) + delta[0],
                float(point[1]) + delta[1],
            ]
            if _point_on_segment(
                translated,
                calibrated_edge['node1'],
                calibrated_edge['node2'],
                tolerance_cm,
            ):
                task[point_key] = translated
                return
            other_name = 'node2' if endpoint_name == 'node1' else 'node1'
            other = original_edge[other_name]
            edge_dx = float(other[0]) - float(original_corner[0])
            edge_dy = float(other[1]) - float(original_corner[1])
            edge_length = math.hypot(edge_dx, edge_dy)
            shift_length = math.hypot(delta[0], delta[1])
            if edge_length <= tolerance_cm or shift_length <= tolerance_cm:
                continue
            outward = [edge_dx / edge_length, edge_dy / edge_length]
            alignment = abs(
                (outward[0] * delta[0] + outward[1] * delta[1])
                / shift_length
            )
            if alignment > 0.25:
                continue
            distance_from_corner = (
                (float(point[0]) - float(original_corner[0])) * outward[0]
                + (float(point[1]) - float(original_corner[1])) * outward[1]
            )
            if -tolerance_cm <= distance_from_corner <= shift_length + tolerance_cm:
                fraction = max(
                    0.0,
                    min(1.0, distance_from_corner / shift_length),
                )
                tangent = [
                    float(original_corner[0]) + outward[0] * shift_length,
                    float(original_corner[1]) + outward[1] * shift_length,
                ]
                corner_point = [
                    round(
                        float(moved_corner[0])
                        + (tangent[0] - float(moved_corner[0])) * fraction,
                        6,
                    ),
                    round(
                        float(moved_corner[1])
                        + (tangent[1] - float(moved_corner[1])) * fraction,
                        6,
                    ),
                ]
                if any(
                    _point_on_segment(
                        corner_point,
                        candidate['node1'],
                        candidate['node2'],
                        tolerance_cm,
                    )
                    for candidate in edges
                ):
                    task[point_key] = corner_point
                    return
        raise ValueError(
            f'Task {point_key} cannot be placed on its rendered sidewalk '
            f'after crosswalk alignment: {point!r}'
        )

    keep_endpoint_on_calibrated_edge('start_point', 0)
    keep_endpoint_on_calibrated_edge('end_point', -1)

    start_edge_index = next(
        index
        for index, edge in enumerate(edges)
        if _point_on_segment(
            task['start_point'],
            edge['node1'],
            edge['node2'],
            tolerance_cm,
        )
    )
    end_edge_index = max(
        index
        for index, edge in enumerate(edges)
        if index >= start_edge_index
        and _point_on_segment(
            task['end_point'],
            edge['node1'],
            edge['node2'],
            tolerance_cm,
        )
    )
    edges = edges[start_edge_index:end_edge_index + 1]
    task['edges'] = edges
    task['start_edge'] = copy.deepcopy(edges[0])
    task['end_edge'] = copy.deepcopy(edges[-1])

    for route_key in ('pedestrian_routes', 'scooter_routes'):
        for route in task.get(route_key) or []:
            route['route'] = [
                remap_exact(point) for point in route.get('route', [])
            ]

    route_info = dict(task.get('route_info') or {})
    route_info['shortest_path'] = reconstruct_route_points_from_task_edges(task)
    task['route_info'] = route_info
    task['render_geometry_calibration'] = {
        'asset': 'BP_Road_Small',
        'crosswalks': calibration_records,
        'non_crossing_connectors': non_crossing_connectors,
    }
    return task


class BenchmarkAlignmentError(ValueError):
    """Deterministic benchmark/configuration failure that retries cannot fix."""

    retryable = False


def validate_selected_task_crosswalk_preflight(
    scenario_data,
    *,
    signal_traffic_enabled,
):
    """Reject authored crosswalks that have no matching rendered corridor.

    Geometry calibration may classify a task-file crosswalk as a non-crossing
    connector when the packaged map has no corresponding native zebra. That
    classification is useful evidence, but silently running the connector as a
    sidewalk would let a traffic benchmark cross an unpainted roadway without
    signal evaluation. Fail the selected cell before task actors or agent
    execution instead.
    """
    if not signal_traffic_enabled:
        return
    calibration = scenario_data.get('render_geometry_calibration') or {}
    connectors = calibration.get('non_crossing_connectors') or []
    if not connectors:
        return
    task_id = scenario_data.get('task_id')
    edge_indices = sorted(
        int(record['edge_index'])
        for record in connectors
        if isinstance(record, dict)
        and isinstance(record.get('edge_index'), (int, float))
    )
    raise BenchmarkAlignmentError(
        'Signalized-traffic preflight failed: authored route crosswalk '
        f'has no matching native zebra corridor (task_id={task_id}, '
        f'edge_indices={edge_indices}). Reconcile the task/map topology or '
        'exclude this cell from the signalized benchmark.'
    )

DIFFICULTY_DYNAMIC_ACTIVATION_RATIO = {
    "default": 1.0,
    "medium": 0.8,
    "easy": 0.6,
}

# Every eligible traffic-rule violation launches its scripted conflict vehicle
# in the maintained benchmark. Difficulty continues to control ambient actor
# activation only. ``hard`` is normalized to ``default`` elsewhere.
CONFLICT_VEHICLE_LAUNCH_PROBABILITY = {
    "default": 1.0,
    "medium": 1.0,
    "easy": 1.0,
}


def evaluation_decision_metrics(
    evaluation_summary,
    decision_count,
    invalid_decision_count,
):
    """Keep evaluator quality separate from protocol/action validity."""
    attempted = max(0, int(decision_count or 0))
    invalid = max(0, int(invalid_decision_count or 0))
    decision_accuracy = float(
        (evaluation_summary or {}).get("evaluation_coverage", {}).get(
            "optimal_choice_rate_all_decisions",
            0.0,
        )
        or 0.0
    )
    action_validity_rate = (
        max(0, attempted - invalid) / attempted if attempted else 0.0
    )
    return decision_accuracy, action_validity_rate


def summarize_conflict_vehicle_events(*event_groups):
    """Return trigger-agnostic consequence counts for result reporting.

    A vehicle can be assigned and staged without ever being released. Count a
    launch only when the final disposition is ``launched``; collision and
    completion update ``status`` but deliberately retain that disposition.
    Older legacy records did not include a disposition, so recognize their
    post-launch terminal statuses only as a compatibility fallback.
    """
    events = [
        event
        for group in event_groups
        for event in list(group or [])
        if isinstance(event, dict)
    ]
    legacy_launched_statuses = {
        'launched',
        'collision',
        'completed',
        'impact',
        'passed_launch_target',
        'stalled_or_distance_limit',
        'target_reached_awaiting_ue_collision',
    }

    def effective_disposition(event):
        disposition = event.get('disposition')
        if disposition:
            return str(disposition)
        status = str(event.get('status') or 'unknown')
        if (
            event.get('vehicle_id') is not None
            and status in legacy_launched_statuses
        ):
            return 'launched'
        return status

    dispositions = [effective_disposition(event) for event in events]
    statuses = [str(event.get('status') or 'unknown') for event in events]
    return {
        'event_count': len(events),
        'launch_count': sum(value == 'launched' for value in dispositions),
        'collision_count': sum(
            event.get('collision_triggered') is True
            or str(event.get('status') or '') == 'collision'
            for event in events
        ),
        'disposition_counts': {
            value: dispositions.count(value)
            for value in sorted(set(dispositions))
        },
        'status_counts': {
            value: statuses.count(value)
            for value in sorted(set(statuses))
        },
        'events': events,
    }


def episode_termination_reason(agent, max_steps):
    """Return a stable terminal label for completed and failed episodes."""
    if getattr(agent, 'success', False):
        return 'success'
    if getattr(agent, 'failed', False):
        return getattr(agent, 'failure_reason', None) or 'failed'
    if int(getattr(agent, 'step_num', 0)) >= int(max_steps):
        return 'max_steps'
    return 'incomplete'

LEVEL_DIFFICULTY_CONFIGS = {
    "level0": {
        "label": "Level 0",
        "description": "Traffic rules only: crosswalk and traffic-light compliance, no pedestrians or dynamic obstacles.",
        "pedestrian_ratio": 0.0,
        "pedestrian_speeds": [],
        "movable_obstacle_ratio": 0.0,
        "irregular_ratio": 0.0,
        "falling_ratio": 0.0,
        "load_static_obstacles_for_internal_geometry": False,
    },
    "level1": {
        "label": "Level 1",
        "description": "Uniform-speed pedestrians on predefined paths.",
        "pedestrian_ratio": 1.0,
        "pedestrian_speeds": [100],
        "movable_obstacle_ratio": 0.0,
        "irregular_ratio": 0.0,
        "falling_ratio": 0.0,
        "load_static_obstacles_for_internal_geometry": False,
    },
    "level2": {
        "label": "Level 2",
        "description": "Variable-speed pedestrians plus additional moving objects.",
        "pedestrian_ratio": 1.0,
        "pedestrian_speeds": [100, 150, 200],
        "movable_obstacle_ratio": 1.0,
        "irregular_ratio": 0.0,
        "falling_ratio": 0.0,
        "load_static_obstacles_for_internal_geometry": True,
    },
    "level3": {
        "label": "Level 3",
        "description": "Level 2 plus irregular crossings by dogs/children-style NPCs.",
        "pedestrian_ratio": 1.0,
        "pedestrian_speeds": [100, 150, 200],
        "movable_obstacle_ratio": 1.0,
        "irregular_ratio": 1.0,
        "falling_ratio": 0.0,
        "load_static_obstacles_for_internal_geometry": True,
    },
    "level4": {
        "label": "Level 4",
        "description": "Level 3 plus sudden falling hazards.",
        "pedestrian_ratio": 1.0,
        "pedestrian_speeds": [100, 150, 200],
        "movable_obstacle_ratio": 1.0,
        "irregular_ratio": 1.0,
        "falling_ratio": 1.0,
        "load_static_obstacles_for_internal_geometry": True,
    },
}

LEVEL_DIFFICULTY_ALIASES = {
    # Human-facing three-band name; result files retain the historical
    # canonical label ``default`` for backward compatibility.
    "hard": "default",
    "0": "level0",
    "1": "level1",
    "2": "level2",
    "3": "level3",
    "4": "level4",
    "l0": "level0",
    "l1": "level1",
    "l2": "level2",
    "l3": "level3",
    "l4": "level4",
    "level_0": "level0",
    "level_1": "level1",
    "level_2": "level2",
    "level_3": "level3",
    "level_4": "level4",
}

class WorldManager:
    @staticmethod
    def _native_intersection_controller_path():
        """Return the native controller class for the packaged UE module.

        The lightweight native runtime exports the class from ``SimWorld``,
        while the full-city compatibility build exports the same controller
        from the legacy ``gym_citynav`` module so its cooked city assets keep
        resolving.  Launchers can select the matching class without changing
        benchmark logic.
        """
        return os.environ.get(
            'SIMWORLD_INTERSECTION_CONTROLLER_MODEL_PATH',
            '/Script/SimWorld.RTTrafficIntersectionController',
        ).strip() or '/Script/SimWorld.RTTrafficIntersectionController'

    def __init__(self, communicator: RTCommunicator, 
                 agent_path: str = "data/agents.json", task_file_path: str = "data/map1_10roads/tasks.json", 
                 seed: int = 0, control_mode: str = 'llm',
                 token_based: bool = False, use_tick: bool = True, realtime_thinking: bool = True, 
                 record_per_step: bool = False, use_action_frames: bool = False,
                 prompt_style: str = 'instructional', max_steps: int = -1,
                 results_dir: str = None,
                 terminate_on_touched_road: bool = True,
                 disable_background_agents: bool = False,
                 traffic_policy: str = TRAFFIC_POLICY_VISUAL_ONLY,
                 red_light_conflict_vehicle_enabled: bool = True,
                 red_light_conflict_launch_distance_min_cm: float = 300.0,
                 red_light_conflict_launch_distance_max_cm: float = 900.0,
                 red_light_conflict_collision_radius_cm: float = 100.0,
                 red_light_conflict_vehicle_probability: float | None = None,
                 red_light_conflict_nominal_speed_cm_s: float = 450.0,
                 conflict_vehicle_release_wait_timeout_s: float = 12.0,
                 static_signal_vehicles: bool | None = None,
                 pedestrian_signal_compliance_probability: float = 1.0,
                 pedestrians_enabled: bool = True,
                 movable_obstacles_enabled: bool = True,
                 irregular_npcs_enabled: bool = True,
                 falling_objects_enabled: bool = True,
                 load_all_unsafe_triggers: bool = False,
                 conflict_vehicle_min_launch_distance_m: float = None,
                 conflict_vehicle_max_launch_distance_m: float = None,
                 conflict_vehicle_impact_radius_m: float = None):

        self.seed = seed  # Store seed for later use
        random.seed(seed)
        self.pedestrians = []
        self.irregular_pedestrians = []
        self.communicator = communicator
        self._max_steps_init = max_steps  # -1: use computed; >=0: use specified
        self.terminate_on_touched_road = terminate_on_touched_road
        self.disable_background_agents = disable_background_agents
        self.traffic_policy = normalize_traffic_policy(traffic_policy)
        self.red_light_conflict_vehicle_enabled = bool(
            red_light_conflict_vehicle_enabled
        )
        # The clean benchmark runner exposes these values in metres, while the
        # collision-reliability branch uses centimetres internally. Accept
        # both APIs and let explicit benchmark metre values take precedence.
        if conflict_vehicle_min_launch_distance_m is not None:
            red_light_conflict_launch_distance_min_cm = (
                float(conflict_vehicle_min_launch_distance_m) * 100.0
            )
        if conflict_vehicle_max_launch_distance_m is not None:
            red_light_conflict_launch_distance_max_cm = (
                float(conflict_vehicle_max_launch_distance_m) * 100.0
            )
        if conflict_vehicle_impact_radius_m is not None:
            red_light_conflict_collision_radius_cm = (
                float(conflict_vehicle_impact_radius_m) * 100.0
            )
        self.red_light_conflict_launch_distance_min_cm = self._finite_float(
            'red_light_conflict_launch_distance_min_cm',
            red_light_conflict_launch_distance_min_cm,
            minimum=1.0,
        )
        self.red_light_conflict_launch_distance_max_cm = self._finite_float(
            'red_light_conflict_launch_distance_max_cm',
            red_light_conflict_launch_distance_max_cm,
            minimum=self.red_light_conflict_launch_distance_min_cm,
        )
        self.red_light_conflict_collision_radius_cm = self._finite_float(
            'red_light_conflict_collision_radius_cm',
            red_light_conflict_collision_radius_cm,
            minimum=0.0,
        )
        # The kinematic consequence vehicle uses a swept centre-distance
        # collision envelope because SetLocation does not reliably emit UE hit
        # events. This is scoped to the one launched vehicle and does not
        # mutate collision bounds for background traffic.
        self._red_light_conflict_vehicle_probability_override = None
        if red_light_conflict_vehicle_probability is not None:
            self._red_light_conflict_vehicle_probability_override = (
                self._finite_float(
                    'red_light_conflict_vehicle_probability',
                    red_light_conflict_vehicle_probability,
                    minimum=0.0,
                    maximum=1.0,
                )
            )
        # The effective value is finalized when the task difficulty is known.
        self.red_light_conflict_vehicle_probability = (
            self._red_light_conflict_vehicle_probability_override
            if self._red_light_conflict_vehicle_probability_override is not None
            else 1.0
        )
        self.red_light_conflict_nominal_speed_cm_s = self._finite_float(
            'red_light_conflict_nominal_speed_cm_s',
            red_light_conflict_nominal_speed_cm_s,
            minimum=1.0,
        )
        self.red_light_conflict_release_wait_timeout_s = self._finite_float(
            'conflict_vehicle_release_wait_timeout_s',
            conflict_vehicle_release_wait_timeout_s,
            minimum=0.0,
        )
        if static_signal_vehicles is None:
            # Reliability is the benchmark default: packaged vehicle
            # Blueprints can reactivate chassis physics after spawn and roll
            # over while they are only waiting in the conflict pool. Dynamic
            # traffic remains available through the explicit false override.
            static_signal_vehicles = os.environ.get(
                'SIMWORLD_SIGNAL_TRAFFIC_STATIC_VEHICLES',
                '1',
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
        self.static_signal_vehicles = bool(static_signal_vehicles)
        self.pedestrian_signal_compliance_probability = self._finite_float(
            'pedestrian_signal_compliance_probability',
            pedestrian_signal_compliance_probability,
            minimum=0.0,
            maximum=1.0,
        )
        self.pedestrians_enabled = bool(pedestrians_enabled)
        self.movable_obstacles_enabled = bool(movable_obstacles_enabled)
        self.irregular_npcs_enabled = bool(irregular_npcs_enabled)
        self.falling_objects_enabled = bool(falling_objects_enabled)
        self.load_all_unsafe_triggers = bool(load_all_unsafe_triggers)
        self.selected_movable_obstacle_ids = []
        self.selected_falling_object_ids = []
        self.activated_movable_obstacle_ids = []
        self.activated_falling_object_ids = []
        self.dynamic_obstacle_metadata = {}
        self.dynamic_activation_requests = []
        # Isolate consequence sampling from actor placement so changing the
        # launch range does not silently change the rest of the scene.
        self._traffic_consequence_rng = random.Random(self.seed ^ 0x5A17C0DE)
        self.traffic_phase_timing = None
        self.traffic_system_snapshots = {}
        self.signal_traffic_communicator = None
        self.signal_traffic_update_count = 0
        self.signal_traffic_error_count = 0
        self.rollout_error = None
        self.red_light_conflict_vehicle_events = []
        self.illegal_crossing_conflict_vehicle_events = []
        self.vehicle_conflict_events = []
        self._active_red_light_conflict = None
        self._active_vehicle_conflicts = []
        self._retired_conflict_vehicles = {}
        # Conflict launches use a dedicated RNG so changing their geometry does
        # not perturb task, pedestrian, or obstacle sampling for the same seed.
        self._conflict_vehicle_rng = random.Random(int(seed) ^ 0x52414345)
        self.conflict_vehicle_min_launch_distance_cm = (
            self.red_light_conflict_launch_distance_min_cm
        )
        self.conflict_vehicle_max_launch_distance_cm = (
            self.red_light_conflict_launch_distance_max_cm
        )
        self.conflict_vehicle_impact_radius_cm = (
            self.red_light_conflict_collision_radius_cm
        )
        self.vehicle_impact_zone_count = 0
        self._conflict_vehicle_geometry_overrides_m = {
            'minimum': conflict_vehicle_min_launch_distance_m,
            'maximum': conflict_vehicle_max_launch_distance_m,
            'impact_radius': conflict_vehicle_impact_radius_m,
        }
        self.signal_vehicle_initial_positions = {}
        self.signal_vehicle_max_displacement_cm = {}
        self.signal_vehicle_pose_sample_count = {}
        self.signal_vehicle_max_tilt_deg = {}
        self.signal_vehicle_z_range_cm = {}
        self.signal_vehicle_instability_events = []
        self._unstable_signal_vehicle_ids = set()
        self._quiesced_static_signal_vehicle_ids = set()
        self.signal_vehicle_tilt_limit_deg = self._finite_float(
            'SIMWORLD_SIGNAL_VEHICLE_MAX_TILT_DEG',
            os.environ.get('SIMWORLD_SIGNAL_VEHICLE_MAX_TILT_DEG', 30.0),
            minimum=0.0,
            maximum=90.0,
        )
        self.signal_vehicle_stop_reasons = {}
        self.signal_vehicle_crossing_status = {}
        self._signal_vehicle_routes = {}
        self.signal_conflict_route_vehicle_ids = []
        self.signal_conflict_route_geometry_available = False
        self.signal_conflict_route_source = None
        self.signal_conflict_route_max_launch_distance_cm_verified = None
        self._retired_signal_vehicle_ids = set()
        self.frozen_decorative_vehicle_ids = []
        self.decorative_vehicle_freeze_passes = []
        self.signal_traffic_min_distance_cm = {
            'vehicle_vehicle': float('inf'),
            'vehicle_pedestrian': float('inf'),
            'vehicle_agent': float('inf'),
            'pedestrian_agent': float('inf'),
        }
        self.signal_traffic_removed_lane_furniture = []
        self.signal_traffic_removed_disabled_background_hazards = []
        self._signal_traffic_lane_pairs = []
        self._signal_traffic_demo_intersection = None
        self._signal_traffic_demo_crosswalk = None
        self._integrated_traffic_recorder = None
        self.integrated_traffic_demo_path = None
        self.integrated_traffic_demo_frame_count = 0
        self.route_crosswalk_marking_ids = []
        self.route_crosswalk_marking_records = []
        self.signal_traffic_drain_sim_time_s = 0.0
        self.signal_traffic_drain_completed = False
        self.use_intersection_phase_api = False
        self.intersection_controller_model_path = (
            '/Script/SimWorld.RTTrafficIntersectionController'
        )
        self.traffic_light_model_path = (
            '/Game/city_props/BP/props/street_light/'
            'BP_street_light.BP_street_light_C'
        )
        self.pedestrian_light_model_path = (
            '/Game/city_props/BP/props/street_light/'
            'BP_street_light_ped.BP_street_light_ped_C'
        )
        self.traffic_light_scale = 1.0
        self.pedestrian_light_scale = 1.0
        self.ped_num_per_edge = _env_optional_nonnegative_int(
            'SIMWORLD_BACKGROUND_PEDESTRIANS_PER_EDGE',
            PED_NUM_PER_EDGE,
        )
        self.irregular_npc_max = _env_optional_nonnegative_int(
            'SIMWORLD_BACKGROUND_IRREGULAR_NPC_MAX',
            None,
        )
        self.control_mode = control_mode
        self.agent_path = agent_path
        self.logger = Logger.get_logger('WorldManager')
        
        # Time configuration (three independent parameters)
        # token_based: True = sim_time from completion_tokens, False = sim_time from LLM real time (considering SLOMO)
        self.token_based = token_based
        # use_tick: True = use tick to advance (experiments), False = env runs in real time (video recording)
        self.use_tick = use_tick
        # realtime_thinking: True = env advances during agent thinking, False = env pauses during thinking
        self.realtime_thinking = realtime_thinking
        
        # Recording configuration
        self.record_per_step = record_per_step
        self.use_action_frames = use_action_frames
        # prompt_style:
        # - 'instructional': always include concise reasoning
        # - 'future_state': instructional plus explicit prediction of future obstacle/light states
        # - 'adaptive': decide per step whether reasoning is needed
        # - 'naive': direct output only (no reasoning)
        self.prompt_style = prompt_style

        # Create session timestamp for organizing results
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = results_dir if results_dir else f"results/{self.session_timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.current_task_id = None  # Track current task ID for saving results
        self._batch_suffix = None  # Used by run_multiple_tasks for unique filenames
        self.rollout_error = None

        # Load model configs (agent_path: JSON file with "llm" list)
        self.model_configs = self._get_model_configs(agent_path)
        if not self.model_configs:
            self.model_configs = [{'llm': {'model': 'gpt-4o', 'provider': 'openai', 'url': '', 'reasoning': False}}]
        self.config = self.model_configs[0]
        
        self.all_scenarios = []  # Store all scenarios for batch processing
        if task_file_path and os.path.exists(task_file_path):
            with open(task_file_path, 'r', encoding='utf-8') as f:
                scenario_file = json.load(f)
                
                if 'tasks' in scenario_file:
                    self.all_scenarios = self._convert_tasks_to_scenarios(scenario_file['tasks'])

        # Create LLM from first config
        self.llm = self._create_llm_from_config(self.config)

        # Setup Qt application for human control mode
        if self.control_mode == 'human':
            self._setup_qt_application()

        # use_tick: sync mode for tick-based advance; no tick: async + game speed for real-time env
        if self.use_tick:
            self.communicator.unrealcv.set_mode('sync')
            self.logger.info(f'Time mode: use_tick=True (tick-based advance for experiments)')
        else:
            self.logger.info(f'Time mode: use_tick=False (real-time env for video recording)')

        self.communicator.set_game_speed(SLOMO)
        
        self.logger.info(f'token_based={self.token_based} (sim_time from {"completion_tokens" if self.token_based else "real time"})')
        if self.realtime_thinking:
            self.logger.info(f'Realtime thinking ENABLED: environment advances during agent thinking')
        else:
            self.logger.info(f'Realtime thinking DISABLED: environment pauses during agent thinking')

    @staticmethod
    def _finite_float(name, value, minimum=None, maximum=None):
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'{name} must be a finite number, got {value!r}'
            ) from exc
        if not math.isfinite(parsed):
            raise ValueError(f'{name} must be a finite number, got {value!r}')
        if minimum is not None and parsed < minimum:
            raise ValueError(f'{name} must be >= {minimum}, got {parsed}')
        if maximum is not None and parsed > maximum:
            raise ValueError(f'{name} must be <= {maximum}, got {parsed}')
        return parsed

    def _reset_task_conflict_rngs(self, task_identity) -> None:
        """Seed consequence and distance draws reproducibly per authored task."""
        try:
            task_component = int(task_identity)
        except (TypeError, ValueError):
            task_component = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(str(task_identity))
            )
        task_seed = (int(self.seed) + 1) * 1_000_003 + task_component * 9_176
        self._traffic_consequence_rng_seed = task_seed + 29
        self._conflict_vehicle_rng_seed = task_seed + 31
        self._traffic_consequence_rng = random.Random(
            self._traffic_consequence_rng_seed
        )
        self._conflict_vehicle_rng = random.Random(
            self._conflict_vehicle_rng_seed
        )
        self._traffic_consequence_draw_count = 0

    def run_single_task(self, task_index: int, difficulty: str = "default"):
        """Run a single task with the given difficulty level.
        
        Args:
            task_index: Index of the task in all_scenarios
            difficulty: "default"(highest), "medium", "easy"
        """
        self.difficulty = self._normalize_difficulty_label(difficulty)
        self.difficulty_config = self._get_difficulty_config()
        self.red_light_conflict_vehicle_probability = (
            self._get_conflict_vehicle_launch_probability()
        )
        self.difficulty_config["conflict_vehicle_launch_probability"] = (
            self.red_light_conflict_vehicle_probability
        )
        # Reset random seed for reproducibility
        # Each task gets a deterministic random state based on seed and task_index
        random.seed(self.seed + task_index)
        self.current_task_index = task_index
        task_seed = (self.seed + 1) * 1_000_003 + task_index * 9_176
        self._pedestrian_compliance_rng = random.Random(task_seed + 23)
        self.vehicle_proximity_collision_events = []
        self._vehicle_proximity_collision_detected = False
        self.rollout_error = None
        
        self.scenario_data = self.all_scenarios[task_index]
        # Set current task ID from scenario data
        self.current_task_id = self.scenario_data.get('task_id', task_index)
        self._reset_task_conflict_rngs(self.current_task_id)
        self._initialize_world()
        # Signal-traffic enablement is derived from the map configuration and
        # environment by _initialize_world(), so run this guard only after that
        # state exists. It still fires before task actors or agent execution.
        validate_selected_task_crosswalk_preflight(
            self.scenario_data,
            signal_traffic_enabled=self.signal_traffic_enabled,
        )
        self._apply_render_settings()
        task_data = self.scenario_data['task']
        start_point = Vector(task_data['start_point'][0], task_data['start_point'][1])
        end_point = Vector(task_data['end_point'][0], task_data['end_point'][1])
        # The ordered task edges are authoritative.  Reconstructing their
        # centerline prevents stale generated shortest paths from cutting
        # across the roadway instead of traversing the task crosswalk.
        path_points = reconstruct_route_points_from_task_edges(task_data)
        generated_path_points = self.scenario_data['route_info']['shortest_path']
        if path_points != generated_path_points:
            self.logger.warning(
                'Replaced stale route_info shortest path with ordered task-edge '
                'route: generated=%s reconstructed=%s',
                generated_path_points,
                path_points,
            )
        shortest_path = [Vector(x, y) for x, y in path_points[1:]]
        first_waypoint = shortest_path[0]
        initial_direction = Vector(
            first_waypoint.x - start_point.x,
            first_waypoint.y - start_point.y
        ).normalize()
        required_time = task_data.get('required_time', 1000)
        
        # Calculate path length and set max_steps
        path_length = route_length_cm(path_points)
        if self._max_steps_init < 0:
            self.max_steps = route_length_to_max_steps(path_length)
            self.logger.info(f"Task path length: {path_length:.2f}cm, max_steps computed: {self.max_steps}")
        else:
            self.max_steps = self._max_steps_init
            self.logger.info(f"Task path length: {path_length:.2f}cm, max_steps specified: {self.max_steps}")
        
        # Get task edges for red light violation detection
        task_edges = task_data.get('edges', [])
        
        # Get traffic signals list
        traffic_signals = self.traffic_controller.traffic_signals if hasattr(self, 'traffic_controller') and self.traffic_controller else []
        route_crosswalks = self.traffic_controller.get_route_crosswalks(
            path_points,
            task_edges=task_edges,
        )
        if (
            route_crosswalks
            and self.signal_traffic_enabled
            and os.environ.get(
                'SIMWORLD_STAGE_ROUTE_SIGNAL_TRAFFIC',
                '0',
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
        ):
            self._stage_signal_traffic_at_crosswalk(route_crosswalks[0])
        self._remove_disabled_background_hazards(path_points)
        traffic_crosswalks = list(
            self.traffic_controller.crosswalks
            if self.traffic_controller is not None
            else route_crosswalks
        )
        crosswalk_signal_groups = (
            self.traffic_controller.get_crosswalk_signal_groups(
                traffic_crosswalks
            )
        )
        crosswalk_intersection_names = (
            self.traffic_controller.get_crosswalk_intersection_names(
                traffic_crosswalks
            )
            if self.use_intersection_phase_api
            else {}
        )
        self.logger.info(
            'Matched route crosswalks for synchronized traffic-light input: %s',
            [crosswalk.id for crosswalk in route_crosswalks],
        )
        self._spawn_route_crosswalk_markings(route_crosswalks)
        traffic_intersections = self.traffic_controller.intersections if hasattr(self, 'traffic_controller') and self.traffic_controller else []
        traffic_signal_config = {}
        if hasattr(self, 'traffic_controller') and self.traffic_controller:
            traffic_signal_config = {
                'green_light_duration': self.traffic_controller.config['traffic.traffic_signal.green_light_duration'],
                'yellow_light_duration': self.traffic_controller.config['traffic.traffic_signal.yellow_light_duration'],
                'pedestrian_green_light_duration': self.traffic_controller.config['traffic.traffic_signal.pedestrian_green_light_duration'],
                'pedestrian_phase_duration': self.traffic_controller.config['traffic.traffic_signal.pedestrian_phase_duration'],
            }
        
        # Use static obstacles already loaded during _initialize_world
        static_obstacles = self.static_obstacles
        
        # Setup recording directory if enabled (include difficulty and model to avoid overwriting)
        record_dir = None
        if self.record_per_step:
            # _batch_suffix is "d{difficulty}_{model}" when from run_multiple_tasks; else use d{difficulty} only
            diff_suffix = self._batch_suffix if self._batch_suffix else f"d{self.difficulty}"
            record_dir = os.path.join(self.results_dir, f'task_{self.current_task_id}_{diff_suffix}_steps')
            os.makedirs(record_dir, exist_ok=True)
            self.logger.info(f'Per-step recording enabled for task {self.current_task_id}: {record_dir}')
        
        self.agent = RTAgent(start_point, initial_direction, end_point, 
                                shortest_path, required_time, self.communicator, self.llm,
                                token_based=self.token_based, use_tick=self.use_tick,
                                time_alpha=TIME_ALPHA, time_beta=TIME_BETA,
                                realtime_thinking=self.realtime_thinking,
                                 task_edges=task_edges, traffic_signals=traffic_signals,
                                 traffic_intersections=traffic_intersections,
                                 traffic_signal_config=traffic_signal_config,
                                 route_crosswalks=route_crosswalks,
                                 crosswalk_signal_groups=crosswalk_signal_groups,
                                 crosswalk_intersection_names=crosswalk_intersection_names,
                                static_obstacles=static_obstacles, record_per_step=self.record_per_step,
                                record_dir=record_dir, slomo=SLOMO, use_action_frames=self.use_action_frames,
                                prompt_style=self.prompt_style,
                                terminate_on_touched_road=self.terminate_on_touched_road,
                                traffic_phase_timing=self.traffic_phase_timing,
                                traffic_policy=self.traffic_policy,
                                simulation_step_callback=(
                                    self._update_signal_following_traffic
                                    if self.signal_traffic_enabled
                                    else None
                                ),
                                red_light_violation_callback=(
                                    self._handle_red_light_violation
                                    if self.signal_traffic_enabled
                                    else None
                                ),
                                illegal_crossing_callback=(
                                    self._handle_illegal_crossing
                                    if self.signal_traffic_enabled
                                    else None
                                ),
                                dynamic_obstacles=self.dynamic_obstacle_metadata,
                                traffic_roads=(
                                    self.traffic_controller.roads
                                    if self.traffic_controller is not None
                                    else []
                                ),
                                traffic_crosswalks=traffic_crosswalks,
                                traffic_sidewalks=(
                                    self.traffic_controller.sidewalks
                                    if self.traffic_controller is not None
                                    else []
                                ))

        # 2026-09-17: task-scoped wedge recovery (see base.rt_agent).
        try:
            self.agent.wedge_recovery_enabled = (
                WEDGE_RECOVERY_ALL_TASKS or int(self.current_task_id) in WEDGE_RECOVERY_TASK_IDS)
        except (TypeError, ValueError):
            self.agent.wedge_recovery_enabled = False

        self.spawn_agents()
        self.run()

    def _apply_render_settings(self):
        """Apply reproducible off-screen exposure after world initialization."""
        auto_exposure = getattr(self, 'render_auto_exposure', None)
        exposure_offset = getattr(self, 'render_exposure_offset', None)
        if auto_exposure is None and exposure_offset is None:
            return
        commands = []
        if auto_exposure is not None:
            commands.append(
                'vrun r.DefaultFeature.AutoExposure '
                f'{1 if auto_exposure else 0}'
            )
        if exposure_offset is not None:
            # r.ExposureOffset is not an Unreal Engine console variable.  The
            # UnrealCV ``vrun`` command still returns ``ok`` for it, which made
            # dark off-screen captures look configured even though UE ignored
            # the value.  AutoExposure.Bias is the engine-supported setting.
            commands.append(
                'vrun r.DefaultFeature.AutoExposure.Bias '
                f'{exposure_offset:g}'
            )
        responses = {}
        for command in commands:
            response = self.communicator.unrealcv.client.request(command)
            responses[command] = response
            self.logger.info('Render setting %s => %s', command, response)
        self.render_setting_responses = responses

    

    def run(self):
        rollout_exception = None
        try:
            while self.agent.step_num < self.max_steps:
                # Stop only on normal completion, the requested building-collision limit, or max_steps.
                if hasattr(self.agent, 'success') and self.agent.success:
                    if self.control_mode == 'human' and hasattr(self.agent, 'human_interface'):
                        self.agent.human_interface.task_completed()
                    break
                if getattr(self.agent, 'failed', False):
                    break
                self.agent.step(mode=self.control_mode)
                self._record_ue_vehicle_collision_outcome()
            if getattr(self.agent, 'success', False):
                self._drain_signal_traffic_after_agent_completion()
        except Exception as e:
            rollout_exception = e
            self.rollout_error = {
                'type': type(e).__name__,
                'message': str(e),
            }
            # A transport/UE failure is not a valid navigation outcome.  Mark
            # it explicitly so orchestration cannot mistake a partial JSON
            # (success=False, failed=False) for a completed benchmark.
            self.agent.success = False
            self.agent.failed = True
            self.agent.failure_reason = 'runtime_error'
            self.logger.error(f"Error during agent step: {e}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(
                f"Benchmark rollout stopped after a runtime/UE communication "
                f"error: {type(e).__name__}: {e}"
            ) from e
        finally:
            # Save evaluation data when run ends (naturally or interrupted)
            self.save_evaluation_data()
        if rollout_exception is not None:
            raise RuntimeError(
                'Rollout terminated by a runtime/UE communication error'
            ) from rollout_exception

    def save_evaluation_data(self, task_id=None):
        """Save agent evaluation data to JSON file
        
        Args:
            task_id: Optional task identifier to include in the filename and data.
                     If not provided, uses self.current_task_id
        """
        # Use current_task_id if no task_id is provided
        if task_id is None:
            task_id = self.current_task_id

        # Finalize the MP4 before publishing its path and frame count.
        self._close_integrated_traffic_recorder()
        
        # Calculate final time cost
        if hasattr(self.agent, 'start_time'):
            self.agent.time_cost = time.time() - self.agent.start_time
        
        # A terminal failure always takes precedence over reaching the
        # destination in the same action (for example, a vehicle impact at the
        # far curb).
        if getattr(self.agent, 'failed', False):
            self.agent.success = False
        elif (
            hasattr(self.agent, 'current_destination')
            and self.agent.current_destination is None
        ):
            self.agent.success = True
        
        # Calculate shortest path length for SPL metric using original path
        shortest_path_length = 0
        if self.agent.original_shortest_path:
            # Add distance from start point to first waypoint
            shortest_path_length += self.agent.start_pos.distance(self.agent.original_shortest_path[0])
            
            # Calculate total length of the shortest path
            for i in range(len(self.agent.original_shortest_path) - 1):
                shortest_path_length += self.agent.original_shortest_path[i].distance(self.agent.original_shortest_path[i + 1])
        # Get re-planning quality evaluation data
        evaluation_summary = self.agent.get_evaluation_summary()
        step_records = self.agent.evaluator.step_records if self.agent.evaluator else []

        evaluator = getattr(self.agent, 'evaluator', None)
        if evaluator is not None and self.dynamic_obstacle_metadata:
            try:
                evaluator._get_npc_positions()
            except Exception:
                self.logger.exception(
                    'Failed to capture final dynamic actor positions'
                )
        dynamic_actor_telemetry = (
            evaluator.dynamic_actor_telemetry()
            if evaluator is not None
            else {
                'registered_count': 0,
                'sampled_live_count': 0,
                'observed_moving_count': 0,
                'movement_threshold_cm': 10.0,
                'max_displacement_cm': 0.0,
                'actors': [],
            }
        )
        observed_moving_by_kind = {
            kind: sum(
                bool(actor.get('observed_moving'))
                and actor.get('kind') == kind
                for actor in dynamic_actor_telemetry.get('actors', [])
            )
            for kind in ('movable_obstacle', 'falling_object')
        }
        decision_accuracy, action_validity_rate = evaluation_decision_metrics(
            evaluation_summary,
            self.agent.decision_count,
            self.agent.invalid_decision_count,
        )
        termination_reason = episode_termination_reason(
            self.agent,
            self.max_steps,
        )
        timeout_text = " ".join(
            str(value or "")
            for value in (
                getattr(self, "rollout_error", None),
                getattr(self.agent, "failure_reason", None),
                getattr(self.agent, "stuck_reason", None),
            )
        ).lower()
        benchmark_metrics = episode_metrics(
            success=bool(self.agent.success),
            shortest_path_length_cm=shortest_path_length,
            traveled_path_length_cm=getattr(
                self.agent, "traveled_path_length_cm", 0.0
            ),
            sim_time_seconds=self.agent.sim_time_elapsed,
            collision_count=self.agent.collision_count,
            passive_collision_count=self.agent.passive_collision_count,
            fall_count=self.agent.fall_count,
            oil_count=self.agent.oil_count,
            water_count=self.agent.water_count,
            red_light_violation_count=self.agent.red_light_violations_count,
            illegal_crossing_violation_count=getattr(
                self.agent, "illegal_crossing_violations_count", 0
            ),
            decision_trace=getattr(self.agent, "decision_trace", []),
            conflict_event_groups=(
                self.red_light_conflict_vehicle_events,
                self.illegal_crossing_conflict_vehicle_events,
            ),
            total_tokens=self.agent.total_token_count,
            completion_tokens=self.agent.total_completion_tokens,
            reasoning_tokens=self.agent.total_reasoning_tokens,
            parse_error_count=self.agent.parse_error_count,
            invalid_decision_count=self.agent.invalid_decision_count,
            timeout_count=int("timeout" in timeout_text or "timed out" in timeout_text),
            collision_type_counts=getattr(
                self.agent, "collision_type_counts", {}
            ),
            passive_collision_type_counts=getattr(
                self.agent, "passive_collision_type_counts", {}
            ),
            vehicle_collision_count=getattr(
                self.agent, "vehicle_collision_count", 0
            ),
        )
        red_conflict_summary = summarize_conflict_vehicle_events(
            self.red_light_conflict_vehicle_events
        )
        illegal_conflict_summary = summarize_conflict_vehicle_events(
            self.illegal_crossing_conflict_vehicle_events
        )
        traffic_conflict_summary = summarize_conflict_vehicle_events(
            self.red_light_conflict_vehicle_events,
            self.illegal_crossing_conflict_vehicle_events,
        )

        # Prepare evaluation data
        evaluation_data = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "route_crosswalk_hops": int(
                getattr(self, 'scenario_data', {}).get('task', {}).get(
                    'crosswalk_hops', 0
                )
            ),
            "route_reconstruction": {
                "applied": bool(
                    getattr(self, 'scenario_data', {}).get('route_info', {}).get(
                        'ordered_edge_route_reconstructed'
                    )
                ),
                "runtime_route_points": getattr(self, 'scenario_data', {}).get(
                    'route_info', {}
                ).get('shortest_path', []),
                "generated_route_points": getattr(self, 'scenario_data', {}).get(
                    'route_info', {}
                ).get('generated_shortest_path', []),
            },
            "seed": self.seed,
            "traffic_consequence_rng_seed": getattr(
                self, '_traffic_consequence_rng_seed', None
            ),
            "conflict_vehicle_rng_seed": getattr(
                self, '_conflict_vehicle_rng_seed', None
            ),
            "difficulty": getattr(self, 'difficulty', None),
            "difficulty_config": getattr(self, 'difficulty_config', None),
            "static_obstacle_geometry_exposure": (
                "internal_runtime_geometry_not_model_prompt"
            ),
            "model_input_geometry_policy": {
                "static_obstacle_geometry": (
                    "internal_runtime_geometry_not_model_prompt"
                ),
                "mapped_static_obstacles_in_model_context": 0,
            },
            # Retained for result-schema compatibility; mapped obstacle
            # records are not part of the ordinary VLM prompt.
            "num_static_obstacles_in_agent_context": 0,
            "num_static_obstacles_in_model_context": 0,
            "num_static_obstacles_available_to_internal_geometry": len(
                getattr(self, 'static_obstacles', [])
            ),
            "num_static_obstacles_loaded": len(
                getattr(
                    self,
                    'map_static_obstacles',
                    getattr(self, 'static_obstacles', []),
                )
            ),
            "num_pedestrians": len(self.pedestrians),
            "num_irregular_pedestrians": len(self.irregular_pedestrians),
            "scripted_pedestrian_motion_policy": dict(
                SCRIPTED_PEDESTRIAN_MOTION_POLICY
            ),
            "scripted_pedestrian_motion": (
                self._scripted_pedestrian_motion_summary()
            ),
            "dynamic_actor_telemetry": dynamic_actor_telemetry,
            "num_movable_obstacles_total": len(getattr(self, 'movable_obstacle_ids', [])),
            "num_movable_obstacles_selected_for_activation": len(getattr(self, 'selected_movable_obstacle_ids', [])),
            "num_movable_obstacles_activation_requests_issued": len(getattr(self, 'activated_movable_obstacle_ids', [])),
            "num_movable_obstacles_activated": observed_moving_by_kind['movable_obstacle'],
            "num_falling_objects_total": len(getattr(self, 'falling_object_ids', [])),
            "num_falling_objects_activated": len(getattr(self, 'activated_falling_object_ids', [])),
            "optional_asset_activation_failures": list(
                getattr(self, 'optional_asset_activation_failures', [])
            ),
            "control_mode": self.control_mode,
            "alpha": TIME_ALPHA,
            "beta": TIME_BETA,
            "token_based": self.token_based,
            "use_tick": self.use_tick,
            "realtime_thinking": self.realtime_thinking,
            "use_action_frames": self.use_action_frames,
            "observation_resolution_px": {
                "width": self.agent.camera_resolution[0],
                "height": self.agent.camera_resolution[1],
            },
            "prompt_style": self.prompt_style,
            "traffic_policy": self.traffic_policy,
            "traffic_assistance_enabled": (
                self.agent.traffic_assistance_enabled
            ),
            "red_light_conflict_vehicle_enabled": (
                self.red_light_conflict_vehicle_enabled
            ),
            "red_light_conflict_launch_distance_min_cm": (
                self.red_light_conflict_launch_distance_min_cm
            ),
            "red_light_conflict_launch_distance_max_cm": (
                self.red_light_conflict_launch_distance_max_cm
            ),
            "red_light_conflict_collision_radius_cm": (
                self.red_light_conflict_collision_radius_cm
            ),
            "red_light_conflict_vehicle_probability": (
                self.red_light_conflict_vehicle_probability
            ),
            "red_light_conflict_nominal_speed_cm_s": (
                self.red_light_conflict_nominal_speed_cm_s
            ),
            "conflict_vehicle_release_wait_timeout_s": (
                self.red_light_conflict_release_wait_timeout_s
            ),
            "static_signal_vehicles": self.static_signal_vehicles,
            "pedestrian_signal_compliance_probability": (
                self.pedestrian_signal_compliance_probability
            ),
            "render_auto_exposure": getattr(
                self,
                'render_auto_exposure',
                None,
            ),
            "render_exposure_offset": getattr(
                self,
                'render_exposure_offset',
                None,
            ),
            "render_setting_responses": getattr(
                self,
                'render_setting_responses',
                {},
            ),
            "agent_camera_resolution": list(self.agent.camera_resolution),
            "agent_camera_fov_deg": self.agent.fov,
            "agent_camera_id": self.agent.camera_id,
            "agent_camera_mode": self.agent.first_person_camera_mode,
            "agent_camera_eye_height_offset_cm": (
                self.agent.first_person_eye_height_offset_cm
            ),
            "agent_camera_pitch_deg": self.agent.first_person_camera_pitch_deg,
            "route_crosswalk_marking_ids": list(
                getattr(self, 'route_crosswalk_marking_ids', [])
            ),
            "route_crosswalk_marking_records": list(
                getattr(self, 'route_crosswalk_marking_records', [])
            ),
            "render_geometry_calibration": self.scenario_data.get(
                'render_geometry_calibration', {}
            ),
            "terminate_on_touched_road": self.terminate_on_touched_road,
            "background_agents_disabled": self.disable_background_agents,
            "frozen_decorative_vehicle_count": len(
                getattr(self, 'frozen_decorative_vehicle_ids', [])
            ),
            "frozen_decorative_vehicle_ids": list(
                getattr(self, 'frozen_decorative_vehicle_ids', [])
            ),
            "decorative_vehicle_freeze_passes": list(
                getattr(self, 'decorative_vehicle_freeze_passes', [])
            ),
            "signal_controlled_vehicle_count": (
                self.signal_traffic_vehicle_count
            ),
            "signal_controlled_pedestrian_count": (
                self.signal_traffic_pedestrian_count
            ),
            "signal_traffic_update_count": self.signal_traffic_update_count,
            "signal_traffic_error_count": self.signal_traffic_error_count,
            "signal_vehicle_max_displacement_cm": {
                str(key): round(value, 2)
                for key, value in self.signal_vehicle_max_displacement_cm.items()
            },
            "signal_vehicle_pose_sample_count": {
                str(key): int(value)
                for key, value in self.signal_vehicle_pose_sample_count.items()
            },
            "signal_vehicle_max_tilt_deg": {
                str(key): round(value, 3)
                for key, value in self.signal_vehicle_max_tilt_deg.items()
            },
            "signal_vehicle_z_range_cm": {
                str(key): {
                    'min': round(value['min'], 3),
                    'max': round(value['max'], 3),
                }
                for key, value in self.signal_vehicle_z_range_cm.items()
            },
            "signal_vehicle_tilt_limit_deg": (
                self.signal_vehicle_tilt_limit_deg
            ),
            "signal_vehicle_instability_events": list(
                self.signal_vehicle_instability_events
            ),
            "signal_vehicle_stop_reasons": {
                str(key): sorted(value)
                for key, value in self.signal_vehicle_stop_reasons.items()
            },
            "signal_vehicle_crossing_status": {
                str(key): value
                for key, value in self.signal_vehicle_crossing_status.items()
            },
            "signal_vehicle_cleared_intersection_count": sum(
                1
                for value in self.signal_vehicle_crossing_status.values()
                if value.get('cleared_intersection')
            ),
            "signal_conflict_route_vehicle_ids": list(
                self.signal_conflict_route_vehicle_ids
            ),
            "signal_conflict_route_coverage_count": len(
                self.signal_conflict_route_vehicle_ids
            ),
            "signal_conflict_route_geometry_available": bool(
                self.signal_conflict_route_geometry_available
            ),
            "signal_conflict_route_source": (
                self.signal_conflict_route_source
            ),
            "signal_conflict_route_max_launch_distance_cm_verified": (
                self.signal_conflict_route_max_launch_distance_cm_verified
            ),
            "signal_traffic_min_distance_cm": {
                key: (None if math.isinf(value) else round(value, 2))
                for key, value in self.signal_traffic_min_distance_cm.items()
            },
            "signal_traffic_removed_lane_furniture": (
                self.signal_traffic_removed_lane_furniture
            ),
            "signal_traffic_removed_disabled_background_hazards": (
                self.signal_traffic_removed_disabled_background_hazards
            ),
            "signal_traffic_removed_runtime_furniture": list(
                getattr(
                    self.traffic_controller.vehicle_manager,
                    'removed_generated_route_furniture',
                    [],
                )
            ),
            "integrated_traffic_demo": self.integrated_traffic_demo_path,
            "integrated_traffic_demo_frame_count": (
                self.integrated_traffic_demo_frame_count
            ),
            "signal_traffic_drain_sim_time_s": round(
                self.signal_traffic_drain_sim_time_s,
                2,
            ),
            "signal_traffic_drain_completed": (
                self.signal_traffic_drain_completed
            ),
            "red_light_conflict_vehicle_count": (
                red_conflict_summary['launch_count']
            ),
            "red_light_conflict_vehicle_launch_count": (
                red_conflict_summary['launch_count']
            ),
            "red_light_conflict_vehicle_collision_count": (
                red_conflict_summary['collision_count']
            ),
            "red_light_conflict_disposition_count": (
                red_conflict_summary['event_count']
            ),
            "red_light_conflict_disposition_counts": (
                red_conflict_summary['disposition_counts']
            ),
            "red_light_conflict_status_counts": (
                red_conflict_summary['status_counts']
            ),
            "red_light_conflict_vehicle_events": (
                self.red_light_conflict_vehicle_events
            ),
            "illegal_crossing_violations_count": int(
                getattr(self.agent, 'illegal_crossing_violations_count', 0)
            ),
            "illegal_crossing_events": list(
                getattr(self.agent, 'illegal_crossing_events', [])
            ),
            "illegal_crossing_conflict_vehicle_events": list(
                self.illegal_crossing_conflict_vehicle_events
            ),
            "illegal_crossing_conflict_vehicle_count": (
                illegal_conflict_summary['launch_count']
            ),
            "illegal_crossing_conflict_vehicle_launch_count": (
                illegal_conflict_summary['launch_count']
            ),
            "illegal_crossing_conflict_vehicle_collision_count": (
                illegal_conflict_summary['collision_count']
            ),
            "illegal_crossing_conflict_disposition_count": (
                illegal_conflict_summary['event_count']
            ),
            "illegal_crossing_conflict_disposition_counts": (
                illegal_conflict_summary['disposition_counts']
            ),
            "illegal_crossing_conflict_status_counts": (
                illegal_conflict_summary['status_counts']
            ),
            "traffic_conflict_vehicle_count": (
                traffic_conflict_summary['launch_count']
            ),
            "traffic_conflict_vehicle_launch_count": (
                traffic_conflict_summary['launch_count']
            ),
            "traffic_conflict_vehicle_collision_count": (
                traffic_conflict_summary['collision_count']
            ),
            "traffic_conflict_disposition_count": (
                traffic_conflict_summary['event_count']
            ),
            "traffic_conflict_disposition_counts": (
                traffic_conflict_summary['disposition_counts']
            ),
            "traffic_conflict_status_counts": (
                traffic_conflict_summary['status_counts']
            ),
            "traffic_conflict_vehicle_events": list(
                traffic_conflict_summary['events']
            ),
            "signal_pedestrian_compliant_count": sum(
                1
                for pedestrian in getattr(
                    self.traffic_controller,
                    'pedestrians',
                    [],
                )
                if getattr(pedestrian, 'follows_traffic_signal', True)
            ),
            "signal_pedestrian_noncompliant_count": sum(
                1
                for pedestrian in getattr(
                    self.traffic_controller,
                    'pedestrians',
                    [],
                )
                if not getattr(pedestrian, 'follows_traffic_signal', True)
            ),
            "signal_pedestrian_compliance": {
                str(pedestrian.id): bool(
                    getattr(pedestrian, 'follows_traffic_signal', True)
                )
                for pedestrian in getattr(
                    self.traffic_controller,
                    'pedestrians',
                    [],
                )
            },
            "signal_pedestrian_entry_events": list(
                getattr(
                    self.traffic_controller.pedestrian_manager,
                    'signal_entry_events',
                    [],
                )
            ),
            "model": self.agent.llm.model_name,
            "provider": getattr(self.agent.llm, 'provider', None),
            "api_mode": getattr(self.agent.llm, 'api_mode', None),
            "image_detail": getattr(self.agent.llm, 'image_detail', None),
            "text_verbosity": getattr(self.agent.llm, 'text_verbosity', None),
            "service_tier": getattr(self.agent.llm, 'service_tier', None),
            "store_api_response": getattr(self.agent.llm, 'store', None),
            "max_steps": self.max_steps,
            "final_step": self.agent.step_num,
            "termination_policy": {
                "version": "destination_collision_route_budget_v5",
                "destination_success_terminal": True,
                "vehicle_collision_terminal": True,
                "consecutive_building_collision_limit": 3,
                "stagnation_prompt_guidance": False,
                "stagnation_progress_epsilon_cm": 25.0,
                "stagnation_terminal": False,
                "signal_required_waits_exempt_from_stagnation": True,
                "default_max_steps": "3_per_full_shortest_route_meter",
            },
            "movement_timing_policy": {
                "version": "distance_over_speed_v1",
                "nominal_speed_cm_s": 200.0,
                "move_duration": "commanded_distance_cm / nominal_speed_cm_s",
                "post_move_tail_seconds": 0.0,
            },
            "evaluation_metrics_policy": {
                "version": "paper_metrics_v1",
                "spl": "S * L_star / max(L_star, L)",
                "simulation_time_efficiency": "S * T_star / T_sim",
                "optimal_time": "L_star / 2_m_per_s",
                "safe_success": "success with no recorded safety event",
            },
            "traffic_evaluation_policy": {
                "version": "two_rule_traffic_events_v6",
                "collision_authority": (
                    "ue_counters_plus_launched_vehicle_swept_radius"
                ),
                "vehicle_collision_scope": "launched_conflict_vehicle_only",
                "launched_vehicle_collision_radius_cm": (
                    self.red_light_conflict_collision_radius_cm
                ),
                "conflict_vehicle_release_wait_timeout_s": (
                    self.red_light_conflict_release_wait_timeout_s
                ),
                "painted_crosswalk_half_width_cm": 210,
                "roadway_occupancy_guard_half_width_cm": 600,
                "rendered_sidewalk_half_width_cm": 200,
                "illegal_crossing_authority": (
                    "ue_touched_road_outside_strict_pedestrian_geometry"
                ),
                "traffic_rule_events": [
                    "illegal_crossing",
                    "red_light_violation",
                ],
                "red_light_rule": (
                    "walk_entry_admission_persists_to_far_curb"
                ),
                "conflict_vehicle_triggers": [
                    "illegal_crossing",
                    "red_light_violation",
                ],
            },
            "artifact_schema_version": "two_rule_traffic_events_v6",
            "success": self.agent.success,
            "failed": self.agent.failed,
            "termination_reason": termination_reason,
            "failure_reason": getattr(self.agent, 'failure_reason', None),
            "rollout_error": getattr(self, 'rollout_error', None),
            "stuck": getattr(self.agent, 'stuck', False),
            "stuck_reason": getattr(self.agent, 'stuck_reason', None),
            "decision_count": self.agent.decision_count,
            "parse_error_count": self.agent.parse_error_count,
            "invalid_decision_count": self.agent.invalid_decision_count,
            "time_cost": self.agent.time_cost,         # wall clock time cost
            "avg_response_time": self.agent.avg_response_time,           # average response time
            "avg_token_count": self.agent.total_token_count / self.agent.decision_count if self.agent.decision_count > 0 else 0,
            "total_prompt_tokens": benchmark_metrics["tokens"]["prompt"],
            "total_completion_tokens": self.agent.total_completion_tokens,
            "total_reasoning_tokens": self.agent.total_reasoning_tokens,
            "reasoning_token_usage_reported_decisions": getattr(
                self.agent, "reasoning_token_usage_reported_decisions", 0
            ),
            "reasoning_token_usage_complete": (
                self.agent.decision_count > 0
                and getattr(
                    self.agent, "reasoning_token_usage_reported_decisions", 0
                ) == self.agent.decision_count
            ),
            "total_api_cost_usd": getattr(self.agent, "total_api_cost_usd", 0.0),
            "total_cached_prompt_tokens": getattr(self.agent, "total_cached_prompt_tokens", 0),
            "total_cache_write_tokens": getattr(self.agent, "total_cache_write_tokens", 0),
            "total_tokens": self.agent.total_token_count,
            "avg_char_count": self.agent.total_char_count / self.agent.decision_count if self.agent.decision_count > 0 else 0,
            "avg_completion_tokens": self.agent.total_completion_tokens / self.agent.decision_count if self.agent.decision_count > 0 else 0,
            "avg_reasoning_tokens": self.agent.total_reasoning_tokens / self.agent.decision_count if self.agent.decision_count > 0 else 0,
            "reasoning_enabled": self.agent.llm.reasoning if hasattr(self.agent.llm, 'reasoning') else False,
            "reasoning_effort": self.agent.llm.reasoning_effort if hasattr(self.agent.llm, 'reasoning_effort') else None,
            "max_llm_tokens": self.agent.llm.max_tokens if hasattr(self.agent.llm, 'max_tokens') else None,
            "llm_extra_body": self.agent.llm.extra_body if hasattr(self.agent.llm, 'extra_body') else None,
            "required_time": self.agent.required_time,           # required time
            "sim_time": self.agent.sim_time_elapsed,          # simulated time
            "optimal_time": shortest_path_length / 200 if shortest_path_length > 0 else 0,  # optimal time = path length / speed
            "wait_count": self.agent.wait_count,
            "move_count": self.agent.move_count,
            "move_1m_count": self.agent.move_1m_count,
            "move_2m_count": self.agent.move_2m_count,
            "move_4m_count": self.agent.move_4m_count,
            "turn_count": self.agent.turn_count,
            "adjust_speed_count": self.agent.adjust_speed_count,
            "collision_count": self.agent.collision_count,
            "passive_collision_count": self.agent.passive_collision_count,
            "vehicle_collision_count": getattr(
                self.agent,
                'vehicle_collision_count',
                0,
            ),
            "collision_type_counts": getattr(self.agent, 'collision_type_counts', {}),
            "passive_collision_type_counts": getattr(self.agent, 'passive_collision_type_counts', {}),
            "building_collision_count": getattr(self.agent, 'collision_type_counts', {}).get('building', 0),
            "human_collision_count": getattr(self.agent, 'collision_type_counts', {}).get('human', 0),
            "object_collision_count": getattr(self.agent, 'collision_type_counts', {}).get('object', 0),
            "building_collision_recovery_count": getattr(self.agent, 'building_collision_recovery_count', 0),
            "off_route_move_no_progress_count": getattr(
                self.agent, 'off_route_move_no_progress_count', 0,
            ),
            "building_collision_recovery_events": getattr(self.agent, 'building_collision_recovery_events', []),
            "wedge_recovery_policy": {
                "version": "global_wedge_recovery_v2",
                "enabled": bool(getattr(self.agent, 'wedge_recovery_enabled', False)),
                "task_ids": "all" if WEDGE_RECOVERY_ALL_TASKS else sorted(WEDGE_RECOVERY_TASK_IDS),
                "consecutive_blocked_moves": WEDGE_RECOVERY_CONSECUTIVE_MOVES,
            },
            "wedge_recovery_count": getattr(self.agent, 'wedge_recovery_count', 0),
            "wedge_recovery_events": getattr(self.agent, 'wedge_recovery_events', []),
            "consecutive_same_building_action_count": getattr(self.agent, 'consecutive_same_building_action_count', 0),
            "consecutive_building_collision_count": getattr(self.agent, 'consecutive_same_building_action_count', 0),
            "fall_count": self.agent.fall_count,
            "oil_count": self.agent.oil_count,
            "water_count": self.agent.water_count,
            "red_light_violations_count": self.agent.red_light_violations_count,
            "hazard_overlap_policy": {
                "version": "continuous_region_occupancy_v1",
                "count_once_per_continuous_occupancy": True,
                "spatial_rearm_distance_cm": 450.0,
            },
            "hazard_overlap_raw_counts": dict(
                getattr(self.agent, 'hazard_overlap_raw_counts', {})
            ),
            "hazard_overlap_suppressed_counts": dict(
                getattr(self.agent, 'hazard_overlap_suppressed_counts', {})
            ),
            "stagnation_count": int(
                getattr(self.agent, 'stagnation_count', 0) or 0
            ),
            "max_stagnation_count": int(
                getattr(self.agent, 'max_stagnation_count', 0) or 0
            ),
            "stagnation_prompt_guidance": False,
            "stagnation_events": list(
                getattr(self.agent, 'stagnation_events', [])
            ),
            "red_light_violation_events": list(
                getattr(self.agent, 'red_light_violation_events', [])
            ),
            "traffic_light_gate_count": self.agent.traffic_light_gate_count,
            "traffic_phase_timing": (
                self.traffic_phase_timing.to_dict()
                if self.traffic_phase_timing is not None
                else None
            ),
            "traffic_system_initial_snapshots": self.traffic_system_snapshots,
            "final_position": {
                "x": self.agent.position.x,
                "y": self.agent.position.y
            },
            "final_destination": {
                "x": self.agent.destination.x,
                "y": self.agent.destination.y
            } if self.agent.destination else None,
            # Decision accuracy is evaluator quality, not merely whether the
            # response happened to parse.  Preserve the latter under an
            # explicit name so older result consumers can migrate without
            # losing information.
            "decision_accuracy": decision_accuracy,
            "action_validity_rate": action_validity_rate,
            "parse_success_rate": (self.agent.decision_count - self.agent.parse_error_count) / self.agent.decision_count if self.agent.decision_count > 0 else 0,
            "shortest_path_length": shortest_path_length,
            "traveled_path_length_cm": benchmark_metrics["traveled_path_length_cm"],
            "spl": benchmark_metrics["spl"],
            "simulation_time_efficiency": benchmark_metrics["simulation_time_efficiency"],
            "benchmark_metrics": benchmark_metrics,
            "position_history": self.agent.position_history,
            "decision_trace": getattr(self.agent, 'decision_trace', []),
            "safe_trajectory_history": getattr(self.agent, 'safe_trajectory_history', []),
            "replanning_quality": {
                "step_records": step_records,
                "summary": evaluation_summary
            }
        }
        
        # Save to JSON file in session directory
        if task_id is not None:
            suffix = f"_{self._batch_suffix}" if self._batch_suffix else ""
            filename = f"{self.results_dir}/task_{task_id}{suffix}.json"
        else:
            filename = f"{self.results_dir}/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation data saved to {filename}")
        return filename


    def _activate_optional_background_objects(self, object_ids, asset_kind):
        """Activate optional scene assets without aborting the benchmark.

        Unreal may occasionally execute an ``ActivateMovement`` blueprint but
        fail to return its acknowledgement before the bounded request timeout.
        Replaying that mutating request is unsafe, while failing the pedestrian
        task because one decorative oil patch or falling prop did not animate
        is unnecessarily destructive.  Keep the asset in the scene, record the
        degraded activation, and continue.  Programming/data errors still
        propagate instead of being hidden.
        """
        activated = []
        failures = getattr(self, 'optional_asset_activation_failures', None)
        if failures is None:
            failures = []
            self.optional_asset_activation_failures = failures

        for object_id in object_ids:
            try:
                self.communicator.activate_object_movement(object_id)
            except (TimeoutError, ConnectionError, OSError) as exc:
                failure = {
                    'asset_id': str(object_id),
                    'asset_kind': str(asset_kind),
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                    'disposition': 'kept_static',
                }
                failures.append(failure)
                self.logger.warning(
                    'Optional %s %s movement activation failed; keeping the '
                    'asset static and continuing: %s',
                    asset_kind,
                    object_id,
                    exc,
                )
                continue
            activated.append(object_id)
        return activated

    def spawn_agents(self):
        # Construction-script child actors can register a frame after their
        # parent road blueprint.  Repeat the freeze immediately before any
        # managed traffic is spawned, while leaving RT_SIGNAL_VEHICLE actors
        # untouched and dynamic.
        if hasattr(
            self.communicator,
            'freeze_generated_decorative_vehicles',
        ):
            self._freeze_generated_decorative_vehicles('pre_agent_spawn')
        self.communicator.spawn_agent(self.agent, self.agent.name, '/Game/RealTimeBench/Agent/BP_RT_Agent.BP_RT_Agent_C', type='agent')
        self.communicator.rt_agent_adjust_speed(self.agent.name, self.agent.speed)
        if getattr(self.agent, 'first_person_camera_enabled', False):
            self.agent.initialize_first_person_camera()
            # The free camera is created after the initial world render setup.
            # Reapply exposure so its post-process state matches the rollout.
            self._apply_render_settings()
        self.logger.info(f"Spawned agent {self.agent.name} with speed {self.agent.speed} cm/s")

        def spawn_configured_pedestrians(pedestrians, actor_type, label):
            """Fully configure each actor before spawning the next actor."""
            for pedestrian in pedestrians:
                batch = [pedestrian]
                self.communicator.spawn_pedestrians(batch, type=actor_type)
                self.communicator.set_pedestrians_waypoints(batch)
                self.communicator.set_pedestrians_speed(batch)
            self.logger.info(
                "Spawned and configured %s %s", len(pedestrians), label
            )
        
        # Spawn pedestrians if available (type=2: pedestrian model)
        if hasattr(self, 'pedestrians') and self.pedestrians:
            spawn_configured_pedestrians(self.pedestrians, 2, "pedestrians")
        
        # Spawn irregular NPCs if available (type=0: robot_dog model, generated for difficulty >= 2)
        if hasattr(self, 'irregular_pedestrians') and self.irregular_pedestrians:
            spawn_configured_pedestrians(
                self.irregular_pedestrians,
                0,
                "irregular NPCs (robot_dog)",
            )

        self.activated_movable_obstacle_ids = []
        self.activated_falling_object_ids = []
        self.dynamic_activation_requests = []

        if (
            self.dynamic_obstacle_metadata
            and getattr(self.agent, 'evaluator', None) is not None
        ):
            try:
                # Establish a live pre-activation baseline. Movement telemetry
                # must not confuse an authored JSON/UE placement offset with a
                # successful ActivateMovement request.
                self.agent.evaluator._get_npc_positions()
            except Exception:
                self.logger.exception(
                    'Failed to capture pre-activation dynamic actor positions'
                )

        # Selection and static/dynamic partitioning happen before RTAgent is
        # constructed. Here we only issue UE activation requests and retain
        # the legacy ``activated_*`` fields as request-issued compatibility
        # metrics; observed motion is reported separately by RTEvaluator.
        selected_obstacles = list(self.selected_movable_obstacle_ids)
        if selected_obstacles:
            for obstacle_id in selected_obstacles:
                response = self.communicator.activate_object_movement(obstacle_id)
                self.dynamic_activation_requests.append({
                    'actor_id': obstacle_id,
                    'kind': 'movable_obstacle',
                    'response': (
                        response
                        if isinstance(response, (type(None), bool, int, float, str, list, dict))
                        else repr(response)
                    ),
                    'response_nonempty': response not in (None, '', {}, []),
                })
            self.activated_movable_obstacle_ids = list(selected_obstacles)
            activation_ratio = self._get_movable_obstacle_activation_ratio()
            total_movable = len(self.movable_obstacle_ids)
            self.logger.info(
                f"Issued movement activation requests for {len(selected_obstacles)}/{total_movable} movable obstacles "
                f"(difficulty={self.difficulty}, ratio={activation_ratio:.0%})"
            )

        # Activate falling objects according to difficulty/level configuration.
        selected_falling_ids = list(self.selected_falling_object_ids)
        if selected_falling_ids:
            for obj_id in selected_falling_ids:
                response = self.communicator.activate_object_movement(obj_id)
                self.dynamic_activation_requests.append({
                    'actor_id': obj_id,
                    'kind': 'falling_object',
                    'response': (
                        response
                        if isinstance(response, (type(None), bool, int, float, str, list, dict))
                        else repr(response)
                    ),
                    'response_nonempty': response not in (None, '', {}, []),
                })
            self.activated_falling_object_ids = list(selected_falling_ids)
            activation_ratio = self._get_falling_object_activation_ratio()
            total_falling = len(self.falling_object_ids)
            self.logger.info(
                f"Issued movement activation requests for {len(selected_falling_ids)}/{total_falling} falling objects "
                f"(difficulty={self.difficulty}, ratio={activation_ratio:.0%})"
            )

        # Repair procedural-map authoring conflicts before managed traffic is
        # spawned.  Genuine road debris remains and is handled by braking.
        self._remove_misplaced_lane_furniture()

        # Spawn traffic signals if available
        self._spawn_and_setup_traffic_signals()

        # Spawn a distinct set of traffic participants controlled by the map
        # managers.  Unlike scripted scenario NPCs, these actors query the UE
        # traffic heads before entering an intersection or crosswalk.
        if self.signal_traffic_enabled:
            self.traffic_controller.spawn_vehicles()
            self.traffic_controller.spawn_pedestrians()
            self.signal_traffic_communicator.spawn_ue_manager(
                self.traffic_controller.config['simworld.ue_manager_path']
            )
            self.signal_traffic_communicator.update_objects()
            self.traffic_controller.pedestrian_manager.set_pedestrians_max_speed(
                self.signal_traffic_communicator
            )
            self._update_signal_following_traffic(0.0)
            self.logger.info(
                'Spawned signal-controlled traffic: %s vehicles, %s pedestrians',
                self.signal_traffic_vehicle_count,
                self.signal_traffic_pedestrian_count,
            )

        time.sleep(3)

        if hasattr(self, 'pedestrians') and self.pedestrians:
            self.communicator.start_pedestrians_simulation(self.pedestrians)
            self.logger.info(f"Started {len(self.pedestrians)} pedestrians simulation")
        if hasattr(self, 'irregular_pedestrians') and self.irregular_pedestrians:
            self.communicator.start_pedestrians_simulation(self.irregular_pedestrians)
            self.logger.info(f"Started {len(self.irregular_pedestrians)} irregular NPCs simulation")
        self._scripted_pedestrians_started = bool(
            getattr(self, 'pedestrians', [])
            or getattr(self, 'irregular_pedestrians', [])
        )

        self._initialize_integrated_traffic_recorder()

    def _stage_signal_traffic_at_crosswalk(self, crosswalk):
        """Place managed actors on every approach of the route intersection.

        With only three background cars, map-wide random spawning usually puts
        no traffic near the benchmark agent.  This deterministic staging keeps
        the ordinary VehicleManager/PedestrianManager controls but guarantees
        that the VLM, cars, and walkers exercise the same real intersection.
        Two opposing cars take straight connections; the T-leg car takes its
        legal turn, so the validation is not a straight-only special case.
        """
        # A WorldManager can be reused by local validation harnesses.  Route
        # coverage is per staging pass, not cumulative across tasks/resets.
        self.signal_conflict_route_vehicle_ids = []
        self.signal_conflict_route_geometry_available = False
        self.signal_conflict_route_source = None
        self.signal_conflict_route_max_launch_distance_cm_verified = None
        self._signal_vehicle_routes = {}
        self.signal_vehicle_crossing_status = {}
        intersection = self.traffic_controller._get_route_crosswalk_intersection(
            crosswalk
        )
        if intersection is None:
            self.logger.warning(
                'No native intersection resolves route crosswalk %s; '
                'route signal traffic cannot be staged.',
                crosswalk.id,
            )
            return
        incoming_lanes = sorted(
            (
                lane for lane in intersection.lane_mapping
                if any(
                    light.lane_id == lane.id
                    for light in intersection.traffic_lights
                )
            ),
            # If fewer managed cars exist than approaches, cover the agent's
            # crosswalk first. A violation consequence can then use a real
            # authored lane rather than an off-lane synthetic trajectory.
            key=lambda lane: (
                not self._incoming_lane_has_crosswalk_route(
                    lane,
                    intersection,
                    crosswalk,
                ),
                lane.id,
            ),
        )
        self.signal_conflict_route_geometry_available = any(
            self._incoming_lane_has_crosswalk_route(
                lane,
                intersection,
                crosswalk,
            )
            for lane in incoming_lanes
        )
        lane_pairs = []
        for index, (vehicle, lane) in enumerate(
            zip(self.traffic_controller.vehicles, incoming_lanes)
        ):
            if vehicle in vehicle.current_lane.vehicles:
                vehicle.current_lane.remove_vehicle(vehicle)
            lane.add_vehicle(vehicle)
            vehicle.current_lane = lane
            approach_distance = max(
                550.0,
                vehicle.length / 2.0
                + float(vehicle.config['traffic.vehicle.distance_to_end'])
                + 100.0,
            )
            vehicle.position = lane.end - lane.direction * approach_distance
            vehicle.direction = math.degrees(
                math.atan2(lane.direction.y, lane.direction.x)
            )
            vehicle.waypoints = cal_waypoints(
                vehicle.position,
                lane.end,
                vehicle.config['traffic.gap_between_waypoints'],
            )
            vehicle.state = VehicleState.STOPPED
            vehicle.stop_reason = 'initial_signal_approach'
            vehicle.set_attributes(0.0, 1.0, 0.0)

            possible_lanes = list(intersection.lane_mapping[lane])
            route_candidates = []
            for candidate in possible_lanes:
                route_points = self._vehicle_route_points(
                    lane,
                    candidate,
                    intersection,
                )
                blockers = self._count_route_blockers(route_points, vehicle)
                cross = lane.direction.cross(candidate.direction)
                dot = lane.direction.dot(candidate.direction)
                # Prefer an unobstructed route first.  For equal clearance,
                # straight is preferred, then a non-conflicting right turn,
                # then a left turn.
                if dot > 0.95:
                    movement_rank = 0
                elif cross < 0:
                    movement_rank = 1
                else:
                    movement_rank = 2
                covers_agent_crosswalk = self._route_intersects_crosswalk(
                    route_points,
                    crosswalk,
                )
                route_candidates.append(
                    (
                        # Guarantee one genuinely conflicting authored route
                        # when the intersection geometry provides one.  After
                        # coverage is established, ordinary blocker/movement
                        # preferences remain authoritative for other cars.
                        (
                            0
                            if (
                                not self.signal_conflict_route_vehicle_ids
                                and covers_agent_crosswalk
                            )
                            else (
                                1
                                if not self.signal_conflict_route_vehicle_ids
                                else 0
                            )
                        ),
                        blockers,
                        movement_rank,
                        candidate.id,
                        covers_agent_crosswalk,
                        candidate,
                        route_points,
                    )
                )
            (
                _,
                blockers,
                _,
                _,
                _,
                next_lane,
                route_points,
            ) = min(route_candidates)
            # One managed vehicle is staged per incoming approach, so pinning
            # that approach is deterministic without constraining other map
            # intersections.  At the T-leg, max-dot is still a real turn.
            intersection.lane_mapping[lane] = [next_lane]
            lane_pairs.append((lane, next_lane))
            self._signal_vehicle_routes[vehicle.id] = {
                'incoming_lane': lane,
                'outgoing_lane': next_lane,
                'stop_line': lane.end,
                'exit_line': next_lane.start,
                'approach_start': Vector(vehicle.position.x, vehicle.position.y),
                'path_points': route_points,
            }
            if self._route_intersects_crosswalk(
                [vehicle.position, *route_points],
                crosswalk,
            ):
                self.signal_conflict_route_vehicle_ids.append(vehicle.id)
            self.signal_vehicle_crossing_status[vehicle.id] = {
                'incoming_lane_id': lane.id,
                'outgoing_lane_id': next_lane.id,
                'entered_intersection': False,
                'cleared_intersection': False,
                'max_incoming_progress_cm': round(-approach_distance, 2),
                'max_outgoing_progress_cm': None,
                'planned_route_blocker_count': blockers,
            }
            self.logger.info(
                'Staged vehicle %s route lane %s -> %s (%s mapped blockers)',
                vehicle.id,
                lane.id,
                next_lane.id,
                blockers,
            )

        # Keep managed background walkers visible in the same intersection but
        # off the benchmark agent's active crosswalk.  Previously all three
        # actors shared one narrow centerline, so a compliant stopped walker
        # physically blocked the VLM at the curb.
        crosswalk_options = {}
        for connections in intersection.sidewalk_mapping.values():
            for _, candidate_crosswalk in connections:
                if candidate_crosswalk is not None:
                    crosswalk_options[candidate_crosswalk.id] = candidate_crosswalk
        alternate_crosswalks = [
            candidate for candidate in crosswalk_options.values()
            if candidate.id != crosswalk.id
        ]
        if alternate_crosswalks:
            route_center = (crosswalk.start + crosswalk.end) * 0.5
            pedestrian_crosswalk = max(
                alternate_crosswalks,
                key=lambda candidate: (
                    ((candidate.start + candidate.end) * 0.5).distance(route_center),
                    candidate.id,
                ),
            )
        else:
            pedestrian_crosswalk = crosswalk

        approaches = []
        for sidewalk, connections in intersection.sidewalk_mapping.items():
            for next_sidewalk, candidate_crosswalk in connections:
                if candidate_crosswalk is pedestrian_crosswalk:
                    incoming_point = min(
                        (sidewalk.start, sidewalk.end),
                        key=lambda point: point.distance(intersection.center),
                    )
                    approaches.append(
                        (
                            sidewalk,
                            incoming_point,
                            (next_sidewalk, pedestrian_crosswalk),
                        )
                    )
                    break
        approaches.sort(key=lambda item: item[1].distance(crosswalk.start))
        if len(approaches) >= 2:
            selected = (approaches[0], approaches[-1])
            for pedestrian, (sidewalk, incoming_point, target) in zip(
                self.traffic_controller.pedestrians,
                selected,
            ):
                if pedestrian in pedestrian.current_sidewalk.pedestrians:
                    pedestrian.current_sidewalk.remove_pedestrian(pedestrian)
                sidewalk.add_pedestrian(pedestrian)
                pedestrian.current_sidewalk = sidewalk
                far_endpoint = max(
                    (sidewalk.start, sidewalk.end),
                    key=lambda point: point.distance(incoming_point),
                )
                toward_curb = (incoming_point - far_endpoint).normalize()
                pedestrian.position = incoming_point - toward_curb * 100.0
                pedestrian.direction = math.degrees(
                    math.atan2(toward_curb.y, toward_curb.x)
                )
                pedestrian.waypoints = [incoming_point]
                pedestrian.state = PedestrianState.STOP
                pedestrian.speed = 140.0
                intersection.sidewalk_mapping[sidewalk] = [target]

        self._signal_traffic_lane_pairs = lane_pairs
        self._signal_traffic_demo_intersection = intersection
        self._signal_traffic_demo_crosswalk = crosswalk
        self.logger.info(
            'Staged route traffic at intersection %s / agent crosswalk %s / '
            'managed-pedestrian crosswalk %s: %s vehicles, %s pedestrians',
            intersection.id,
            crosswalk.id,
            pedestrian_crosswalk.id,
            min(len(incoming_lanes), len(self.traffic_controller.vehicles)),
            min(len(approaches), len(self.traffic_controller.pedestrians)),
        )
        probe_distance_cm = float(
            getattr(
                self,
                'red_light_conflict_launch_distance_max_cm',
                900.0,
            )
        )
        probe_vehicles = list(
            getattr(self.traffic_controller, 'vehicles', []) or []
        )
        route_probe = None
        if probe_vehicles:
            route_probe = self._lane_aligned_conflict_launch(
                {
                    'crosswalk_projection': 0.0,
                    'crosswalk_entry_direction': 1,
                    'agent_position': {
                        'x': crosswalk.start.x,
                        'y': crosswalk.start.y,
                    },
                    'agent_speed_cm_s': 200.0,
                },
                crosswalk,
                probe_vehicles,
                probe_distance_cm,
            )
        self.signal_conflict_route_geometry_available = route_probe is not None
        if route_probe is not None:
            self.signal_conflict_route_source = route_probe['route'].get(
                'route_source',
                'staged_intersection_route',
            )
            self.signal_conflict_route_max_launch_distance_cm_verified = (
                probe_distance_cm
            )
        if not self.signal_conflict_route_geometry_available:
            self.logger.warning(
                'No real-lane route supports the full %.1f cm conflict launch '
                'distance at agent crosswalk %s; red-light consequence will '
                'be unavailable.',
                probe_distance_cm,
                crosswalk.id,
            )
        elif not self.signal_conflict_route_vehicle_ids:
            self.logger.info(
                'No background staged route intersects agent crosswalk %s; '
                'red-light conflicts will use runtime real-lane source %s.',
                crosswalk.id,
                self.signal_conflict_route_source,
            )

    def _vehicle_route_points(self, incoming_lane, outgoing_lane, intersection):
        """Return the same centerline geometry VehicleManager will follow."""
        dot = max(
            -1.0,
            min(1.0, incoming_lane.direction.dot(outgoing_lane.direction)),
        )
        angle = math.degrees(math.acos(dot))
        if angle <= 5.0:
            transition = cal_waypoints(
                incoming_lane.end,
                outgoing_lane.start,
                self.traffic_controller.config['traffic.gap_between_waypoints'],
            )
        elif abs(angle - 180.0) <= 1e-6:
            transition = [outgoing_lane.start]
        else:
            transition = get_bezier_points(
                incoming_lane.end,
                outgoing_lane.start,
                extend_control_point(
                    incoming_lane.end,
                    outgoing_lane.start,
                    intersection.center,
                    0.15,
                ),
                self.traffic_controller.config['traffic.steering_point_num'],
            )
        # Cover almost the complete outgoing lane so the isolated traffic demo
        # shows vehicles continuing normally after the junction rather than
        # immediately braking for a cooked task prop just beyond the metric's
        # 5 m intersection-clearance line.
        exit_probe = outgoing_lane.start + outgoing_lane.direction * 5000.0
        return [incoming_lane.end, *transition, exit_probe]

    def _incoming_lane_has_crosswalk_route(
        self,
        incoming_lane,
        intersection,
        crosswalk,
    ):
        """Return whether any legal authored movement crosses ``crosswalk``."""
        return any(
            self._route_intersects_crosswalk(
                self._vehicle_route_points(
                    incoming_lane,
                    outgoing_lane,
                    intersection,
                ),
                crosswalk,
            )
            for outgoing_lane in intersection.lane_mapping[incoming_lane]
        )

    def _count_route_blockers(self, route_points, vehicle):
        """Count non-furniture hazards occupying a candidate route."""
        removable_markers = self._route_cleanup_markers()
        swept_half_width = vehicle.width / 2.0 + float(
            vehicle.config.get(
                'traffic.vehicle.static_obstacle_lateral_margin',
                120.0,
            )
        )
        blockers = 0
        for obstacle in self.static_obstacles:
            obstacle_type = str(obstacle.get('type', '')).lower()
            if any(marker in obstacle_type for marker in removable_markers):
                # These are removed later only if they intersect the selected
                # route.  They must not force an unnatural route choice.
                continue
            point = Vector(float(obstacle['x']), float(obstacle['y']))
            distance = min(
                self._point_segment_distance(point, start, end)
                for start, end in zip(route_points, route_points[1:])
            )
            if distance <= swept_half_width:
                blockers += 1
        return blockers

    def _route_cleanup_markers(self):
        """Obstacle types that may be removed from a staged traffic route.

        Street furniture is a map-authoring error when its collision center is
        inside a lane.  Falling-object actors are benchmark hazards rather than
        normal traffic, so they are removable only for runs that explicitly
        disable background agents/hazards.
        """
        markers = ['tree', 'trash', 'hydrant', 'scooter']
        if self.disable_background_agents:
            markers.append('rt_')
        return tuple(markers)

    def _remove_disabled_background_hazards(self, agent_route_points):
        """Remove generated task hazards only from active isolated routes.

        The packaged city contains all ``RT_*`` obstacle actors before Python
        decides which benchmark difficulty/background mode to run.  In an
        isolated traffic-system run we remove only actors whose collision
        centers overlap the benchmark route or one of the staged controlled
        vehicle routes.  Ordinary city props and off-route task assets remain.
        """
        if not self.disable_background_agents:
            return

        points = [Vector(point[0], point[1]) for point in agent_route_points]
        segments = list(zip(points, points[1:]))
        for route in self._signal_vehicle_routes.values():
            route_points = [route['approach_start'], *route['path_points']]
            segments.extend(zip(route_points, route_points[1:]))
        if not segments:
            return

        scene_objects = set(self.communicator.unrealcv.get_objects())
        removed_ids = set()
        retained = []
        for obstacle in self.static_obstacles:
            obstacle_type = str(obstacle.get('type', '')).lower()
            obstacle_id = str(obstacle.get('id', ''))
            is_generated_hazard = (
                obstacle_type.startswith('rt_')
                or obstacle_id.startswith('GEN_RT_FO_')
                or obstacle_id.startswith('GEN_RT_RT_')
            )
            if not is_generated_hazard:
                retained.append(obstacle)
                continue
            point = Vector(float(obstacle['x']), float(obstacle['y']))
            distance = min(
                self._point_segment_distance(point, start, end)
                for start, end in segments
            )
            # UE physics vehicles can temporarily deviate several metres while
            # correcting after a long junction.  In explicit traffic-isolation
            # mode clear generated task hazards across the full controlled
            # corridor, not just the mathematical centerline.
            if distance > 1000.0 or obstacle_id not in scene_objects:
                retained.append(obstacle)
                continue
            self.communicator.unrealcv.destroy(obstacle_id)
            removed_ids.add(obstacle_id)
            self.signal_traffic_removed_disabled_background_hazards.append(
                {**obstacle, 'route_distance_cm': round(distance, 2)}
            )

        if removed_ids:
            self.communicator.unrealcv.clean_garbage()
        self.static_obstacles = retained
        self.static_obstacle_positions = [
            Vector(obstacle['x'], obstacle['y'])
            for obstacle in retained
        ]
        self.logger.info(
            'Removed %s generated background-hazard actors overlapping active '
            'agent/vehicle routes for isolated traffic validation',
            len(removed_ids),
        )

    @staticmethod
    def _point_segment_distance(point, start, end):
        segment = end - start
        length_sq = segment.dot(segment)
        if length_sq <= 0:
            return point.distance(start)
        projection = max(
            0.0,
            min(1.0, (point - start).dot(segment) / length_sq),
        )
        return point.distance(start + segment * projection)

    def _remove_misplaced_lane_furniture(self):
        """Remove fixed curb furniture whose authored pose intersects a lane.

        RT12 contains trees, hydrants, bins, and parked scooters with centers on
        generated lane centerlines.  These are map-authoring conflicts, not
        meaningful road hazards.  Real debris is retained and handled by the
        vehicle's forward obstacle detector.
        """
        if not self._signal_traffic_lane_pairs:
            return
        removable_markers = self._route_cleanup_markers()
        segments = []
        for route in self._signal_vehicle_routes.values():
            # Only clean the short, actually selected controlled route.  Do not
            # alter unrelated furniture elsewhere on the same long map lanes.
            points = [route['approach_start'], *route['path_points']]
            segments.extend(zip(points, points[1:]))
        scene_objects = set(self.communicator.unrealcv.get_objects())
        removed_ids = set()
        for obstacle in self.static_obstacles:
            obstacle_type = str(obstacle.get('type', '')).lower()
            if not any(marker in obstacle_type for marker in removable_markers):
                continue
            if obstacle.get('id') not in scene_objects:
                continue
            point = Vector(float(obstacle['x']), float(obstacle['y']))
            distance = min(
                self._point_segment_distance(point, start, end)
                for start, end in segments
            )
            if distance <= 220.0:
                self.communicator.unrealcv.destroy(obstacle['id'])
                removed_ids.add(obstacle['id'])
                self.signal_traffic_removed_lane_furniture.append(
                    {
                        **obstacle,
                        'lane_distance_cm': round(distance, 2),
                        'cleanup_reason': (
                            'disabled_background_hazard_on_controlled_route'
                            if obstacle_type.startswith('rt_')
                            else 'street_furniture_inside_controlled_route'
                        ),
                    }
                )
        if removed_ids:
            self.communicator.unrealcv.clean_garbage()
            self.static_obstacles = [
                obstacle for obstacle in self.static_obstacles
                if obstacle.get('id') not in removed_ids
            ]
            self.static_obstacle_positions = [
                Vector(obstacle['x'], obstacle['y'])
                for obstacle in self.static_obstacles
            ]
        self.logger.info(
            'Removed %s route-conflicting street-furniture/background-hazard '
            'actors from the staged controlled corridors',
            len(removed_ids),
        )

    def _update_signal_traffic_safety_metrics(self):
        vehicles = list(self.traffic_controller.vehicles)
        pedestrians = list(self.traffic_controller.pedestrians)
        transforms = getattr(
            self.signal_traffic_communicator,
            'last_vehicle_transforms',
            {},
        )
        for vehicle in vehicles:
            initial = self.signal_vehicle_initial_positions.setdefault(
                vehicle.id,
                (vehicle.position.x, vehicle.position.y),
            )
            displacement = math.hypot(
                vehicle.position.x - initial[0],
                vehicle.position.y - initial[1],
            )
            self.signal_vehicle_max_displacement_cm[vehicle.id] = max(
                displacement,
                self.signal_vehicle_max_displacement_cm.get(vehicle.id, 0.0),
            )
            transform = transforms.get(vehicle.id)
            if transform is not None:
                rotation = transform.get('rotation') or []
                location = transform.get('location') or []
                if len(rotation) == 3 and len(location) == 3:
                    pitch = (
                        (float(rotation[0]) + 180.0) % 360.0
                    ) - 180.0
                    roll = (
                        (float(rotation[2]) + 180.0) % 360.0
                    ) - 180.0
                    tilt = max(abs(pitch), abs(roll))
                    z = float(location[2])
                    self.signal_vehicle_pose_sample_count[vehicle.id] = (
                        self.signal_vehicle_pose_sample_count.get(
                            vehicle.id,
                            0,
                        )
                        + 1
                    )
                    self.signal_vehicle_max_tilt_deg[vehicle.id] = max(
                        tilt,
                        self.signal_vehicle_max_tilt_deg.get(vehicle.id, 0.0),
                    )
                    z_range = self.signal_vehicle_z_range_cm.setdefault(
                        vehicle.id,
                        {'min': z, 'max': z},
                    )
                    z_range['min'] = min(z_range['min'], z)
                    z_range['max'] = max(z_range['max'], z)
                    if (
                        tilt > self.signal_vehicle_tilt_limit_deg
                        and vehicle.id
                        not in self._unstable_signal_vehicle_ids
                    ):
                        self._unstable_signal_vehicle_ids.add(vehicle.id)
                        event = {
                            'vehicle_id': vehicle.id,
                            'actor_name': transform.get('name'),
                            'sim_time_s': getattr(
                                getattr(self, 'agent', None),
                                'sim_time_elapsed',
                                None,
                            ),
                            'pitch_deg': round(pitch, 3),
                            'roll_deg': round(roll, 3),
                            'tilt_deg': round(tilt, 3),
                            'z_cm': round(z, 3),
                            'tilt_limit_deg': (
                                self.signal_vehicle_tilt_limit_deg
                            ),
                        }
                        self.signal_vehicle_instability_events.append(event)
                        vehicle.set_attributes(0.0, 1.0, 0.0)
                        vehicle.state = VehicleState.STOPPED
                        vehicle.stop_reason = 'physics_instability'
                        self.signal_traffic_communicator.update_vehicle(
                            vehicle.id,
                            0.0,
                            1.0,
                            0.0,
                        )
                        agent = getattr(self, 'agent', None)
                        if agent is not None and not getattr(
                            agent,
                            'failed',
                            False,
                        ):
                            agent.success = False
                            agent.failed = True
                            agent.failure_reason = (
                                'signal_vehicle_instability'
                            )
                        self.logger.error(
                            'Signal vehicle %s became physically unstable: '
                            'pitch=%.2f roll=%.2f z=%.2f (limit=%.2f)',
                            vehicle.id,
                            pitch,
                            roll,
                            z,
                            self.signal_vehicle_tilt_limit_deg,
                        )
            if vehicle.stop_reason:
                self.signal_vehicle_stop_reasons.setdefault(vehicle.id, set()).add(
                    vehicle.stop_reason
                )
            route = self._signal_vehicle_routes.get(vehicle.id)
            status = self.signal_vehicle_crossing_status.get(vehicle.id)
            if route is not None and status is not None:
                incoming_progress = (
                    vehicle.position - route['stop_line']
                ).dot(route['incoming_lane'].direction)
                outgoing_progress = (
                    vehicle.position - route['exit_line']
                ).dot(route['outgoing_lane'].direction)
                status['max_incoming_progress_cm'] = round(
                    max(
                        float(status['max_incoming_progress_cm']),
                        incoming_progress,
                    ),
                    2,
                )
                if incoming_progress >= -50.0:
                    status['entered_intersection'] = True
                if status['entered_intersection']:
                    previous = status['max_outgoing_progress_cm']
                    status['max_outgoing_progress_cm'] = round(
                        max(
                            float('-inf') if previous is None else float(previous),
                            outgoing_progress,
                        ),
                        2,
                    )
                # The rear of a typical car must be beyond the outgoing stop
                # line before this counts as a completed traversal.
                if outgoing_progress >= 500.0:
                    status['cleared_intersection'] = True

        def update_minimum(key, first, second, same_collection=False):
            values = []
            for first_index, first_actor in enumerate(first):
                for second_index, second_actor in enumerate(second):
                    if same_collection and first_index >= second_index:
                        continue
                    values.append(first_actor.position.distance(second_actor.position))
            if values:
                self.signal_traffic_min_distance_cm[key] = min(
                    self.signal_traffic_min_distance_cm[key],
                    min(values),
                )

        update_minimum('vehicle_vehicle', vehicles, vehicles, True)
        update_minimum('vehicle_pedestrian', vehicles, pedestrians)
        if getattr(self, 'agent', None) is not None:
            update_minimum('vehicle_agent', vehicles, [self.agent])
            update_minimum('pedestrian_agent', pedestrians, [self.agent])

        # This finite validation scene owns only one junction.  Once a managed
        # car is well beyond its outgoing stop line, retire it instead of
        # letting the generic map router send it through a later intersection
        # and back into the test junction, where it can block the next phase.
        for vehicle in vehicles:
            if vehicle.id in self._retired_signal_vehicle_ids:
                continue
            status = self.signal_vehicle_crossing_status.get(vehicle.id)
            if not status or not status.get('cleared_intersection'):
                continue
            if float(status.get('max_outgoing_progress_cm') or 0.0) < 2500.0:
                continue
            actor_name = self.signal_traffic_communicator.get_vehicle_name(
                vehicle.id
            )
            self.communicator.unrealcv.destroy(actor_name)
            if vehicle in vehicle.current_lane.vehicles:
                vehicle.current_lane.remove_vehicle(vehicle)
            if vehicle in self.traffic_controller.vehicle_manager.vehicles:
                self.traffic_controller.vehicle_manager.vehicles.remove(vehicle)
            self._retired_signal_vehicle_ids.add(vehicle.id)
            status['retired_after_clearance'] = True
            status['retired_outgoing_progress_cm'] = status[
                'max_outgoing_progress_cm'
            ]
            self.logger.info(
                'Retired signal vehicle %s after clearing %.1f cm beyond the '
                'controlled intersection',
                vehicle.id,
                float(status['max_outgoing_progress_cm']),
            )

    def _initialize_integrated_traffic_recorder(self):
        enabled = os.environ.get(
            'SIMWORLD_RECORD_INTEGRATED_TRAFFIC_DEMO',
            '0',
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        if enabled and self._signal_traffic_demo_intersection is None:
            self.logger.info(
                'Integrated traffic capture is waiting for a route crossing '
                'or the first red-light conflict.'
            )
            return
        if (
            not enabled
            or self._integrated_traffic_recorder is not None
        ):
            return
        try:
            import cv2

            unrealcv = self.communicator.unrealcv
            existing_cameras = str(unrealcv.get_cameras()).split()
            response = unrealcv.client.request('vset /cameras/spawn')
            if isinstance(response, str) and response.lower().startswith('error'):
                raise RuntimeError(response)
            camera_id = len(existing_cameras)
            resolution = (1280, 720)
            altitude_cm = 4200.0
            field_of_view_deg = 70.0
            center = self._signal_traffic_demo_intersection.center
            unrealcv.set_camera_resolution(camera_id, resolution)
            unrealcv.set_camera_location(
                camera_id,
                (center.x, center.y, altitude_cm),
            )
            unrealcv.set_camera_rotation(camera_id, (-90.0, 0.0, 0.0))
            unrealcv.set_camera_fov(camera_id, field_of_view_deg)
            # Newly spawned SceneCapture cameras initialize their own post-
            # process state.  Reapply the global defaults after the camera
            # exists so the demo and the agent view use the same exposure.
            self._apply_render_settings()
            path = os.path.join(
                self.results_dir,
                f'task_{self.current_task_id}_integrated_traffic_demo.mp4',
            )
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                6.0,
                resolution,
            )
            if not writer.isOpened():
                raise RuntimeError('OpenCV could not initialize MP4 writer')
            self._integrated_traffic_recorder = {
                'camera_id': camera_id,
                'writer': writer,
                'resolution': resolution,
                'altitude_cm': altitude_cm,
                'field_of_view_deg': field_of_view_deg,
                'frame_count': 0,
                'vehicle_trails': {},
            }
            self.integrated_traffic_demo_path = path
            self.logger.info('Integrated traffic recorder started: %s', path)
        except Exception:
            self.logger.warning(
                'Could not initialize integrated traffic recorder:\n%s',
                traceback.format_exc(),
            )

    def _record_integrated_traffic_frame(self):
        recorder = self._integrated_traffic_recorder
        if recorder is None:
            return
        try:
            import cv2

            unrealcv = self.communicator.unrealcv
            frame = unrealcv.get_image(
                recorder['camera_id'],
                'lit',
                'direct',
            )
            if frame is None:
                return
            width, height = recorder['resolution']
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            intersection = self._signal_traffic_demo_intersection
            crosswalk = self._signal_traffic_demo_crosswalk
            visible_width_cm = (
                2.0
                * recorder['altitude_cm']
                * math.tan(math.radians(recorder['field_of_view_deg'] / 2.0))
            )
            pixels_per_cm = width / visible_width_cm

            def project(position):
                return (
                    int(width / 2 + (position.y - intersection.center.y) * pixels_per_cm),
                    int(height / 2 - (position.x - intersection.center.x) * pixels_per_cm),
                )

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 112), (8, 10, 14), -1)
            frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
            pedestrian_state, remaining = intersection.get_crosswalk_light_state(
                crosswalk
            )
            walk = pedestrian_state.name == 'PEDESTRIAN_GREEN'
            remaining_label = (
                f'{float(remaining):.0f}s'
                if remaining is not None
                else 'unknown time'
            )
            active_lanes = [
                str(light.lane_id)
                for light in intersection.traffic_lights
                if light.get_state()[0].name == 'VEHICLE_GREEN'
            ]
            pedestrian_label = 'WALK' if walk else "DON'T WALK"
            active_conflicts = list(
                getattr(self, '_active_vehicle_conflicts', []) or []
            )
            active_conflict_ids = {
                conflict['vehicle'].id
                for conflict in active_conflicts
                if conflict.get('vehicle') is not None
            }
            cv2.putText(
                frame,
                f'QWEN AGENT + SIGNAL TRAFFIC | sim t={self.agent.sim_time_elapsed:.1f}s',
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            crossing_parts = []
            for conflict in active_conflicts:
                conflict_vehicle = conflict.get('vehicle')
                conflict_record = conflict.get('record') or {}
                if conflict_vehicle is None:
                    continue
                trigger = str(
                    conflict_record.get('trigger_type') or 'conflict'
                ).replace('illegal_crossing', 'illegal-crossing').replace(
                    'red_light_violation',
                    'red-light',
                )
                crossing_parts.append(
                    f'C{conflict_vehicle.id}:{trigger}:'
                    f"{conflict_record.get('status', 'active').upper()}"
                )
            for vehicle in self.traffic_controller.vehicles:
                status = self.signal_vehicle_crossing_status.get(vehicle.id, {})
                if status.get('cleared_intersection'):
                    state_label = 'CLEARED'
                elif status.get('entered_intersection'):
                    state_label = 'IN INTERSECTION'
                else:
                    state_label = 'APPROACH'
                crossing_parts.append(f'V{vehicle.id}:{state_label}')
            cv2.putText(
                frame,
                ' | '.join(crossing_parts),
                (18, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 210, 80),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"PED: {pedestrian_label} ({remaining_label}) | "
                f"GREEN VEHICLE LANES: {','.join(active_lanes) or 'NONE'}",
                (18, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (55, 220, 90) if walk else (70, 70, 245),
                2,
                cv2.LINE_AA,
            )

            actors = [
                ('A', self.agent, (255, 220, 40)),
                *[
                    (
                        f'V{vehicle.id}',
                        vehicle,
                        (40, 40, 255)
                        if vehicle.id in active_conflict_ids
                        else (40, 170, 255),
                    )
                    for vehicle in self.traffic_controller.vehicles
                ],
                *[
                    (f'P{pedestrian.id}', pedestrian, (80, 230, 100))
                    for pedestrian in self.traffic_controller.pedestrians
                ],
            ]
            for label, actor, color in actors:
                point = project(actor.position)
                if label.startswith('V'):
                    trail = recorder['vehicle_trails'].setdefault(label, [])
                    trail.append(point)
                    if len(trail) > 180:
                        del trail[0]
                    if len(trail) >= 2:
                        cv2.polylines(
                            frame,
                            [np.array(trail, dtype='int32')],
                            False,
                            color,
                            3,
                            cv2.LINE_AA,
                        )
                if not (0 <= point[0] < width and 112 <= point[1] < height):
                    continue
                cv2.circle(frame, point, 11 if label == 'A' else 8, color, -1)
                cv2.circle(frame, point, 14 if label == 'A' else 11, (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    label,
                    (point[0] + 12, point[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            recorder['writer'].write(frame)
            recorder['frame_count'] += 1
        except Exception:
            self.logger.warning(
                'Integrated traffic recorder stopped after capture error:\n%s',
                traceback.format_exc(),
            )
            self._close_integrated_traffic_recorder()

    def _close_integrated_traffic_recorder(self):
        recorder = self._integrated_traffic_recorder
        if recorder is None:
            return
        try:
            self.integrated_traffic_demo_frame_count = recorder['frame_count']
            recorder['writer'].release()
            self.logger.info(
                'Integrated traffic demo saved: %s (%s frames)',
                self.integrated_traffic_demo_path,
                recorder['frame_count'],
            )
        finally:
            self._integrated_traffic_recorder = None

    def _stabilize_static_signal_vehicles(self, active_vehicle_id=None):
        """Quiesce background signal cars before their pose is sampled.

        The packaged vehicle blueprint can re-enable or retain chassis motion
        after a one-shot physics command.  Reliability-mode cars are scenery
        until one is selected for a conflict, so enforce their kinematic,
        upright pose immediately before every state read.  The selected
        consequence car is excluded and remains owned by the conflict updater.
        """
        if not getattr(self, 'static_signal_vehicles', False):
            return
        unrealcv = self.signal_traffic_communicator.unrealcv
        for vehicle in self.traffic_controller.vehicles:
            if vehicle.id == active_vehicle_id:
                continue
            name = self.signal_traffic_communicator.get_vehicle_name(vehicle.id)
            if hasattr(unrealcv, 'set_physics'):
                unrealcv.set_physics(name, False)
            orientation = None
            if hasattr(unrealcv, 'get_orientation'):
                orientation = unrealcv.get_orientation(name)
            if (
                orientation is not None
                and len(orientation) == 3
                and hasattr(unrealcv, 'set_orientation')
            ):
                unrealcv.set_orientation(
                    (0.0, float(orientation[1]), 0.0),
                    name,
                )
            # Background signal vehicles are scenery in reliability mode.
            # Disable their collision every update so the aggregate UE
            # VehicleCollision counter can only be raised by the one active
            # launched consequence vehicle, which is explicitly excluded.
            if hasattr(unrealcv, 'set_collision'):
                unrealcv.set_collision(name, False)
            self._quiesced_static_signal_vehicle_ids.add(vehicle.id)

    def _stabilize_active_kinematic_conflict_vehicle(self, active):
        """Keep the selected consequence car on its commanded upright pose.

        ``v_set_state`` can reactivate chassis motion inside the packaged
        vehicle Blueprint even after ``set_physics(False)``.  A conflict car
        is advanced kinematically, so letting that latent motion run between
        controller updates can launch or roll the actor before its next pose
        sample.  Reassert the deterministic pose after every state command and
        immediately before sampling it.  Collision stays enabled so the UE
        counter and the swept-radius collision authority are unchanged.
        """
        if active is None or not active.get('kinematic_motion'):
            return
        communicator = getattr(self, 'signal_traffic_communicator', None)
        unrealcv = getattr(communicator, 'unrealcv', None)
        if unrealcv is None:
            return

        vehicle = active['vehicle']
        record = active.get('record', {})
        direction = active['direction']
        position = active['launch_position']
        release_sim_time_s = active.get('release_sim_time_s')
        if record.get('released', True) and release_sim_time_s is not None:
            current_sim_time_s = float(
                getattr(getattr(self, 'agent', None), 'sim_time_elapsed', 0.0)
            )
            elapsed_since_release = max(
                0.0,
                current_sim_time_s - float(release_sim_time_s),
            )
            position = (
                active['launch_position']
                + direction
                * self.red_light_conflict_nominal_speed_cm_s
                * elapsed_since_release
            )

        name = active['vehicle_name']
        yaw = math.degrees(math.atan2(direction.y, direction.x))
        if hasattr(unrealcv, 'set_physics'):
            unrealcv.set_physics(name, False)
        if hasattr(unrealcv, 'set_location'):
            unrealcv.set_location(
                (position.x, position.y, active['launch_z_cm']),
                name,
            )
        if hasattr(unrealcv, 'set_orientation'):
            unrealcv.set_orientation((0.0, yaw, 0.0), name)
        if hasattr(unrealcv, 'set_collision'):
            unrealcv.set_collision(name, True)
        vehicle.position = position
        record['kinematic_pose_enforcement_count'] = int(
            record.get('kinematic_pose_enforcement_count', 0)
        ) + 1
        record['kinematic_pose_enforced'] = True

    def _maintain_scripted_pedestrian_motion(self):
        """Keep regular Blueprint pedestrian patrols moving forward.

        Reloads rotate the same cyclic path to the actor's nearest forward
        segment, avoiding a reverse traversal toward an already-consumed first
        waypoint. If two such restarts fail, the existing Blueprint controller
        receives an in-place hard reset of its forward route, speed, and
        movement simulation. Live pedestrian actors are never destroyed and
        respawned during an episode: packaged UE can race render-resource
        cleanup against a same-name respawn and crash its render thread.
        Collision is never disabled and actor pose and identity are preserved.
        """
        if not getattr(self, '_scripted_pedestrians_started', False):
            return
        # Irregular actors have finite back-and-forth demonstrations and can
        # legitimately finish them. Only regular scripted pedestrians are
        # intended to keep looping for the full episode.
        actors = list(getattr(self, 'pedestrians', []))
        if not actors or getattr(self, 'agent', None) is None:
            return

        now = float(getattr(self.agent, 'sim_time_elapsed', 0.0))
        if now + 1e-6 < float(
            getattr(self, '_next_scripted_pedestrian_motion_sample_s', 0.0)
        ):
            return
        interval = SCRIPTED_PEDESTRIAN_MOTION_POLICY[
            'motion_sample_interval_s'
        ]
        self._next_scripted_pedestrian_motion_sample_s = now + interval

        names = [
            self.communicator.get_pedestrian_name(actor.id)
            for actor in actors
        ]
        locations = self.communicator.unrealcv.get_location_batch(names)
        state = getattr(self, '_scripted_pedestrian_motion_state', {})
        threshold = SCRIPTED_PEDESTRIAN_MOTION_POLICY['motion_threshold_cm']
        restart_after = SCRIPTED_PEDESTRIAN_MOTION_POLICY[
            'controller_restart_after_s'
        ]
        live_positions = {
            name: Vector(float(location[0]), float(location[1]))
            for name, location in zip(names, locations)
        }
        static_obstacle_records = []
        for obstacle in getattr(self, 'static_obstacles', []):
            try:
                obstacle_position = Vector(
                    float(obstacle['x']),
                    float(obstacle['y']),
                )
            except (KeyError, TypeError, ValueError):
                continue
            static_obstacle_records.append((
                str(obstacle.get('id', 'unknown')),
                str(obstacle.get('type', 'unknown')),
                obstacle_position,
            ))
        restart_actors = []
        for actor, name, location in zip(actors, names, locations):
            position = Vector(float(location[0]), float(location[1]))
            record = state.get(name)
            if record is None:
                state[name] = {
                    'actor_id': name,
                    'last_position_cm': [position.x, position.y],
                    'last_motion_sim_time_s': now,
                    'last_sample_sim_time_s': now,
                    'max_stationary_interval_s': 0.0,
                    'controller_restart_count': 0,
                    'endpoint_recycle_count': 0,
                    'stalled_controller_restart_count': 0,
                    'consecutive_stalled_restart_count': 0,
                    'forward_waypoint_reload_count': 0,
                    'local_detour_count': 0,
                    'local_detour_waypoint_count': 0,
                    'controller_recreation_count': 0,
                    'controller_recreation_deferred_count': 0,
                    'controller_recreation_verified_motion_count': 0,
                    'controller_recreation_pending_verification': False,
                    'controller_in_place_recovery_count': 0,
                    'controller_in_place_recovery_deferred_count': 0,
                    'controller_in_place_recovery_verified_motion_count': 0,
                    'controller_in_place_recovery_pending_verification': False,
                    'endpoint_recycle_armed': True,
                    'last_restart_sim_time_s': None,
                    'last_recreation_sim_time_s': None,
                }
                continue

            previous = Vector(*record['last_position_cm'])
            displacement = previous.distance(position)
            if displacement >= threshold:
                record['last_motion_sim_time_s'] = now
                record['consecutive_stalled_restart_count'] = 0
                if record.get('controller_recreation_pending_verification'):
                    record['controller_recreation_verified_motion_count'] += 1
                    record['controller_recreation_pending_verification'] = False
                if record.get(
                    'controller_in_place_recovery_pending_verification'
                ):
                    record[
                        'controller_in_place_recovery_verified_motion_count'
                    ] += 1
                    record[
                        'controller_in_place_recovery_pending_verification'
                    ] = False
            stationary_seconds = max(
                0.0,
                now - float(record['last_motion_sim_time_s']),
            )
            record['max_stationary_interval_s'] = round(max(
                float(record['max_stationary_interval_s']),
                stationary_seconds,
            ), 3)
            last_restart = record.get('last_restart_sim_time_s')
            endpoint = (
                actor.waypoints[-1]
                if getattr(actor, 'waypoints', None)
                else None
            )
            endpoint_distance = (
                position.distance(endpoint) if endpoint is not None else None
            )
            # A rotated finite list ends at the segment that the actor just
            # departed. Do not treat that nearby point as a completed patrol
            # until the actor has first moved outside its endpoint radius.
            if (
                not record.get('endpoint_recycle_armed', True)
                and (
                    endpoint_distance is None
                    or endpoint_distance > SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                        'endpoint_recycle_distance_cm'
                    ]
                )
            ):
                record['endpoint_recycle_armed'] = True
            at_patrol_endpoint = (
                record.get('endpoint_recycle_armed', True)
                and
                endpoint is not None
                and endpoint_distance <= (
                    SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                        'endpoint_recycle_distance_cm'
                    ]
                )
            )
            restart_interval_elapsed = (
                last_restart is None
                or now - float(last_restart) >= interval
            )
            endpoint_recycle_due = (
                at_patrol_endpoint and restart_interval_elapsed
            )
            stalled_restart_due = (
                not at_patrol_endpoint
                and stationary_seconds >= restart_after
                and (
                    last_restart is None
                    or now - float(last_restart) >= restart_after
                )
            )
            if endpoint_recycle_due or stalled_restart_due:
                record['controller_restart_count'] += 1
                if endpoint_recycle_due:
                    record['endpoint_recycle_count'] += 1
                    record['consecutive_stalled_restart_count'] = 0
                    # Endpoint completion is expected motion, not a stalled
                    # interval. Reset its motion clock before the new patrol.
                    record['last_motion_sim_time_s'] = now
                else:
                    record['stalled_controller_restart_count'] += 1
                    record['consecutive_stalled_restart_count'] += 1
                reset_controller_in_place = (
                    not endpoint_recycle_due
                    and record['consecutive_stalled_restart_count'] >= (
                        SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                            'in_place_controller_recovery_after_stalled_restarts'
                        ]
                    )
                )
                agent_position = getattr(self.agent, 'position', None)
                agent_distance = (
                    position.distance(agent_position)
                    if agent_position is not None
                    else None
                )
                other_distances = [
                    (other_name, position.distance(other_position))
                    for other_name, other_position in live_positions.items()
                    if other_name != name
                ]
                nearest_actor = (
                    min(other_distances, key=lambda item: item[1])
                    if other_distances
                    else None
                )
                static_distances = [
                    (
                        obstacle_id,
                        obstacle_type,
                        position.distance(obstacle_position),
                        obstacle_position,
                    )
                    for (
                        obstacle_id,
                        obstacle_type,
                        obstacle_position,
                    ) in static_obstacle_records
                ]
                nearest_static = (
                    min(static_distances, key=lambda item: item[2])
                    if static_distances
                    else None
                )
                in_place_recovery_deferred_reasons = []
                if reset_controller_in_place:
                    clear_of_agent = (
                        agent_distance is None
                        or agent_distance >= (
                            SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                'in_place_recovery_agent_clearance_cm'
                            ]
                        )
                    )
                    clear_of_actors = (
                        nearest_actor is None
                        or nearest_actor[1] >= (
                            SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                'in_place_recovery_actor_clearance_cm'
                            ]
                        )
                    )
                    if not (clear_of_agent and clear_of_actors):
                        if not clear_of_agent:
                            in_place_recovery_deferred_reasons.append('agent_clearance')
                        if not clear_of_actors:
                            in_place_recovery_deferred_reasons.append('actor_clearance')
                        reset_controller_in_place = False
                        record[
                            'controller_in_place_recovery_deferred_count'
                        ] += 1
                rotated_waypoints = (
                    rotate_scripted_pedestrian_waypoints_from_position(
                        actor.waypoints,
                        position,
                        endpoint_tolerance_cm=(
                            SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                'endpoint_recycle_distance_cm'
                            ]
                        ),
                    )
                    if getattr(actor, 'waypoints', None)
                    else []
                )
                forward_waypoint = (
                    rotated_waypoints[0] if rotated_waypoints else None
                )
                detour_waypoints = []
                detour_due = (
                    not endpoint_recycle_due
                    and bool(SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                        'local_detour_enabled'
                    ])
                    and record['consecutive_stalled_restart_count'] >= int(
                        SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                            'local_detour_after_stalled_restarts'
                        ]
                    )
                    and forward_waypoint is not None
                    and (
                        (
                            nearest_static is not None
                            and nearest_static[2] <= float(
                                SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                    'local_detour_trigger_distance_cm'
                                ]
                            )
                        )
                        or (
                            nearest_actor is not None
                            and nearest_actor[1] < float(
                                SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                    'observed_capsule_blocking_distance_cm'
                                ]
                            )
                        )
                    )
                )
                if detour_due:
                    sidewalk_check = getattr(
                        self.agent,
                        '_is_within_authored_sidewalk',
                        None,
                    )
                    crosswalk_check = getattr(
                        self.agent,
                        '_is_within_marked_crosswalk',
                        None,
                    )

                    def legal_pedestrian_point(candidate):
                        if not SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                            'local_detour_sidewalk_or_crosswalk_only'
                        ]:
                            return True
                        on_sidewalk = bool(
                            callable(sidewalk_check)
                            and sidewalk_check(candidate)
                        )
                        on_crosswalk = bool(
                            callable(crosswalk_check)
                            and crosswalk_check(candidate)
                        )
                        return on_sidewalk or on_crosswalk

                    detour_waypoints = (
                        build_scripted_pedestrian_recovery_detour(
                            position,
                            forward_waypoint,
                            actor_id=name,
                            blocking_obstacle_position=(
                                nearest_static[3]
                                if nearest_static is not None
                                and nearest_static[2] <= float(
                                    SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                        'local_detour_trigger_distance_cm'
                                    ]
                                )
                                else None
                            ),
                            static_obstacle_positions=[
                                item[2] for item in static_obstacle_records
                            ],
                            other_actor_positions=[
                                other_position
                                for other_name, other_position
                                in live_positions.items()
                                if other_name != name
                            ],
                            agent_position=agent_position,
                            legal_point=legal_pedestrian_point,
                        )
                    )
                    if detour_waypoints:
                        reset_controller_in_place = False
                        record['local_detour_count'] += 1
                        record['local_detour_waypoint_count'] += len(
                            detour_waypoints
                        )
                event = {
                    'actor_id': name,
                    'trigger': (
                        'endpoint_recycle'
                        if endpoint_recycle_due
                        else 'stalled_controller'
                    ),
                    'sim_time_s': round(now, 3),
                    'stationary_interval_s': round(stationary_seconds, 3),
                    'position_cm': [
                        round(position.x, 3),
                        round(position.y, 3),
                    ],
                    'forward_waypoint_cm': (
                        [
                            round(forward_waypoint.x, 3),
                            round(forward_waypoint.y, 3),
                        ]
                        if forward_waypoint is not None
                        else None
                    ),
                    'forward_waypoint_distance_cm': (
                        round(position.distance(forward_waypoint), 3)
                        if forward_waypoint is not None
                        else None
                    ),
                    'agent_distance_cm': (
                        round(agent_distance, 3)
                        if agent_distance is not None
                        else None
                    ),
                    'nearest_scripted_actor': (
                        {
                            'actor_id': nearest_actor[0],
                            'distance_cm': round(nearest_actor[1], 3),
                        }
                        if nearest_actor is not None
                        else None
                    ),
                    'nearest_static_obstacle': (
                        {
                            'actor_id': nearest_static[0],
                            'type': nearest_static[1],
                            'center_distance_cm': round(nearest_static[2], 3),
                        }
                        if nearest_static is not None
                        else None
                    ),
                    'consecutive_stalled_restart_count': int(
                        record['consecutive_stalled_restart_count']
                    ),
                    'recovery_action': (
                        'inject_local_detour_waypoints'
                        if detour_waypoints
                        else (
                            'reset_controller_in_place'
                            if reset_controller_in_place
                            else 'reload_forward_waypoints'
                        )
                    ),
                    'local_detour_waypoints_cm': [
                        [round(point.x, 3), round(point.y, 3)]
                        for point in detour_waypoints
                    ],
                    'in_place_recovery_deferred_reasons': in_place_recovery_deferred_reasons,
                }
                events = getattr(
                    self,
                    '_scripted_pedestrian_motion_events',
                    None,
                )
                if events is None:
                    events = []
                    self._scripted_pedestrian_motion_events = events
                events.append(event)
                restart_actors.append((
                    actor,
                    name,
                    position,
                    endpoint_recycle_due,
                    reset_controller_in_place,
                    detour_waypoints,
                ))
                record['last_restart_sim_time_s'] = now
                record['endpoint_recycle_armed'] = False
            record['last_position_cm'] = [
                round(position.x, 4),
                round(position.y, 4),
            ]
            record['last_sample_sim_time_s'] = now

        self._scripted_pedestrian_motion_state = state
        for (
            actor,
            name,
            position,
            endpoint_recycle,
            reset_controller_in_place,
            detour_waypoints,
        ) in restart_actors:
            # SetWaypoints resets the Blueprint's consumed waypoint list;
            # rotate it first so a mid-route restart continues forward rather
            # than walking backward to the original first waypoint.
            actor.waypoints = rotate_scripted_pedestrian_waypoints_from_position(
                actor.waypoints,
                position,
                endpoint_tolerance_cm=SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                    'endpoint_recycle_distance_cm'
                ],
            )
            if detour_waypoints:
                actor.waypoints = list(detour_waypoints) + actor.waypoints
            state[name]['forward_waypoint_reload_count'] += 1
            if reset_controller_in_place:
                actor.position = Vector(position.x, position.y)
                self.communicator.set_pedestrians_waypoints([actor])
                self.communicator.set_pedestrians_speed([actor])
                self.communicator.start_pedestrians_simulation([actor])
                state[name]['controller_in_place_recovery_count'] += 1
                state[name][
                    'controller_in_place_recovery_pending_verification'
                ] = True
                self.logger.warning(
                    'Reset stalled scripted pedestrian controller in place: '
                    '%s at sim t=%.2fs (forward route and speed reloaded; '
                    'pose, identity, and physical collision preserved; no '
                    'actor destruction or respawn)',
                    name,
                    now,
                )
                continue

            self.communicator.set_pedestrians_waypoints([actor])
            self.communicator.start_pedestrians_simulation([actor])
            if detour_waypoints:
                self.logger.warning(
                    'Injected %d local detour waypoints for stalled scripted '
                    'pedestrian %s at sim t=%.2fs (physical motion; no '
                    'teleport; collision remains enabled)',
                    len(detour_waypoints),
                    name,
                    now,
                )
                continue
            log = self.logger.debug if endpoint_recycle else self.logger.warning
            log(
                '%s scripted pedestrian controller: %s at sim t=%.2fs '
                '(waypoints reloaded; no teleport; collision remains enabled)',
                'Recycled completed' if endpoint_recycle else 'Restarted stalled',
                name,
                now,
            )

    def _scripted_pedestrian_motion_summary(self):
        records = sorted(
            (
                dict(record)
                for record in getattr(
                    self, '_scripted_pedestrian_motion_state', {}
                ).values()
            ),
            key=lambda item: item['actor_id'],
        )
        return {
            'diagnostic_schema_version': 'stall_context_v1',
            'policy': dict(SCRIPTED_PEDESTRIAN_MOTION_POLICY),
            'requested_count': int(getattr(
                self, '_requested_scripted_pedestrian_count', 0
            )),
            'spawned_pedestrian_count': len(getattr(self, 'pedestrians', [])),
            'spawned_irregular_npc_count': len(getattr(
                self, 'irregular_pedestrians', []
            )),
            'sampled_actor_count': len(records),
            'controller_restart_count': sum(
                int(item['controller_restart_count']) for item in records
            ),
            'endpoint_recycle_count': sum(
                int(item.get('endpoint_recycle_count', 0)) for item in records
            ),
            'stalled_controller_restart_count': sum(
                int(item.get('stalled_controller_restart_count', 0))
                for item in records
            ),
            'forward_waypoint_reload_count': sum(
                int(item.get('forward_waypoint_reload_count', 0))
                for item in records
            ),
            'local_detour_count': sum(
                int(item.get('local_detour_count', 0))
                for item in records
            ),
            'local_detour_waypoint_count': sum(
                int(item.get('local_detour_waypoint_count', 0))
                for item in records
            ),
            'controller_recreation_count': sum(
                int(item.get('controller_recreation_count', 0))
                for item in records
            ),
            'controller_recreation_deferred_count': sum(
                int(item.get('controller_recreation_deferred_count', 0))
                for item in records
            ),
            'controller_recreation_verified_motion_count': sum(
                int(item.get('controller_recreation_verified_motion_count', 0))
                for item in records
            ),
            'controller_in_place_recovery_count': sum(
                int(item.get('controller_in_place_recovery_count', 0))
                for item in records
            ),
            'controller_in_place_recovery_deferred_count': sum(
                int(item.get('controller_in_place_recovery_deferred_count', 0))
                for item in records
            ),
            'controller_in_place_recovery_verified_motion_count': sum(
                int(item.get(
                    'controller_in_place_recovery_verified_motion_count', 0
                ))
                for item in records
            ),
            'max_stationary_interval_s': max(
                (float(item['max_stationary_interval_s']) for item in records),
                default=0.0,
            ),
            'events': list(getattr(
                self,
                '_scripted_pedestrian_motion_events',
                [],
            )),
            'actors': records,
        }

    def _update_signal_following_traffic(self, _delta_time_s=0.0):
        """Refresh and control RT traffic against the rendered signal state."""
        if not getattr(self, 'signal_traffic_enabled', False):
            return
        try:
            active_conflict = getattr(
                self, '_active_red_light_conflict', None
            )
            active_vehicle_id = (
                active_conflict['vehicle'].id
                if active_conflict is not None
                else None
            )
            # The active consequence car is deliberately excluded from the
            # background-vehicle pass.  Stabilize its deterministic pose first
            # so update_objects never records a transient Blueprint physics
            # impulse as a real traffic outcome.
            self._stabilize_active_kinematic_conflict_vehicle(active_conflict)
            self._stabilize_static_signal_vehicles(active_vehicle_id)
            self.signal_traffic_communicator.update_objects()
            self.traffic_controller.update_states()
            dynamic_traffic_obstacles = list(
                self.traffic_controller.pedestrians
            )
            if getattr(self, 'agent', None) is not None:
                dynamic_traffic_obstacles.append(self.agent)
            externally_controlled_vehicle_ids = (
                {active_conflict['vehicle'].id}
                if active_conflict is not None
                else set()
            )
            if getattr(self, 'static_signal_vehicles', False):
                externally_controlled_vehicle_ids.update(
                    vehicle.id
                    for vehicle in self.traffic_controller.vehicles
                )
                for vehicle in self.traffic_controller.vehicles:
                    if vehicle.id == active_vehicle_id:
                        continue
                    if vehicle.get_attributes() != (0.0, 1.0, 0.0):
                        vehicle.state = VehicleState.STOPPED
                        vehicle.stop_reason = 'static_background'
                        vehicle.set_attributes(0.0, 1.0, 0.0)
                        self.signal_traffic_communicator.update_vehicle(
                            vehicle.id,
                            0.0,
                            1.0,
                            0.0,
                        )
            self.traffic_controller.vehicle_manager.update_vehicles(
                self.signal_traffic_communicator,
                self.traffic_controller.intersection_manager,
                dynamic_traffic_obstacles,
                static_obstacles=self.static_obstacles,
                externally_controlled_vehicle_ids=(
                    externally_controlled_vehicle_ids
                ),
            )
            self.traffic_controller.pedestrian_manager.update_pedestrians(
                self.signal_traffic_communicator,
                self.traffic_controller.intersection_manager,
                dynamic_agents=(
                    (
                        [self.agent]
                        if getattr(self, 'agent', None) is not None
                        else []
                    )
                    + list(self.traffic_controller.pedestrians)
                ),
            )
            if getattr(self, 'python_grouped_signal_control', False):
                self.traffic_controller.intersection_manager.update_intersections(
                    self.signal_traffic_communicator,
                    delta_time_s=_delta_time_s,
                    timing=self.traffic_phase_timing,
                )
            self._update_red_light_conflict_vehicle()
            self._update_signal_traffic_safety_metrics()
            self._maintain_scripted_pedestrian_motion()
            self._record_integrated_traffic_frame()
            self.signal_traffic_update_count += 1
        except Exception:
            self.signal_traffic_error_count += 1
            self.logger.error(
                'Signal-following traffic update failed:\n%s',
                traceback.format_exc(),
            )
            raise

    def _drain_signal_traffic_after_agent_completion(self):
        """Keep the rendered junction alive until admitted cars clear it.

        Completing the delivery used to close the recorder and destroy every
        actor immediately.  Cars that had legally entered near the final agent
        step therefore appeared to nudge into the junction and freeze in the
        demo, and their crossing metrics were truncated.  This bounded tail
        advances the same UE clock and ordinary signal controller without any
        further VLM calls, stopping early as soon as every staged vehicle's rear
        has passed the outgoing stop line.
        """
        if not getattr(self, 'signal_traffic_enabled', False):
            return
        statuses = getattr(self, 'signal_vehicle_crossing_status', {})
        if not statuses:
            return

        max_seconds = float(
            os.environ.get('SIMWORLD_SIGNAL_TRAFFIC_DRAIN_SECONDS', '60')
        )
        if max_seconds <= 0:
            return

        # Only vehicles that had already entered the intersection when the
        # delivery completed need a tail. Waiting for every staged vehicle is
        # both semantically wrong and, with ``static_signal_vehicles``, can
        # never finish early because those background cars are intentionally
        # frozen before admission.
        active_conflict_vehicle_ids = {
            conflict['vehicle'].id
            for conflict in list(
                getattr(self, '_active_vehicle_conflicts', []) or []
            )
            if conflict.get('vehicle') is not None
        }
        pending_vehicle_ids = {
            vehicle_id
            for vehicle_id, status in statuses.items()
            if status.get('entered_intersection')
            and not status.get('cleared_intersection')
            and (
                not getattr(self, 'static_signal_vehicles', False)
                or vehicle_id in active_conflict_vehicle_ids
            )
        }
        if not pending_vehicle_ids:
            self.signal_traffic_drain_sim_time_s = 0.0
            self.signal_traffic_drain_completed = True
            self.logger.info(
                'Signal-traffic drain skipped: no admitted vehicle remained '
                'inside the intersection'
            )
            return

        def all_cleared():
            return all(
                statuses[vehicle_id].get('cleared_intersection')
                for vehicle_id in pending_vehicle_ids
            )

        chunk = 0.5
        elapsed = 0.0
        self.logger.info(
            'Delivery complete; draining signal traffic for up to %.1f sim s',
            max_seconds,
        )
        while elapsed < max_seconds and not all_cleared():
            advance = min(chunk, max_seconds - elapsed)
            self.agent._advance_simulation_time(advance)
            elapsed += advance

        self.signal_traffic_drain_sim_time_s = elapsed
        self.signal_traffic_drain_completed = all_cleared()
        self.logger.info(
            'Signal-traffic drain finished after %.1f sim s: cleared=%s/%s '
            'vehicles admitted at delivery completion',
            elapsed,
            sum(
                1
                for vehicle_id in pending_vehicle_ids
                if statuses[vehicle_id].get('cleared_intersection')
            ),
            len(pending_vehicle_ids),
        )

    @staticmethod
    def _segment_intersection_point(start_a, end_a, start_b, end_b):
        """Return the finite-segment intersection and both projections."""
        route_delta = end_a - start_a
        crosswalk_delta = end_b - start_b
        denominator = route_delta.cross(crosswalk_delta)
        if abs(denominator) <= 1e-6:
            return None
        offset = start_b - start_a
        route_projection = offset.cross(crosswalk_delta) / denominator
        crosswalk_projection = offset.cross(route_delta) / denominator
        tolerance = 1e-4
        if not (
            -tolerance <= route_projection <= 1.0 + tolerance
            and -tolerance <= crosswalk_projection <= 1.0 + tolerance
        ):
            return None
        point = start_a + route_delta * route_projection
        return point, route_projection, crosswalk_projection

    @classmethod
    def _route_intersects_crosswalk(cls, route_points, crosswalk):
        """Return whether any authored route segment crosses this crosswalk."""
        return any(
            cls._segment_intersection_point(
                start,
                end,
                crosswalk.start,
                crosswalk.end,
            )
            is not None
            for start, end in zip(route_points, route_points[1:])
        )

    def _lane_aligned_conflict_launch(
        self,
        event,
        crosswalk,
        vehicles,
        requested_launch_distance_cm,
    ):
        """Choose a real staged vehicle route that crosses ahead of the agent.

        The old consequence placed an arbitrary vehicle on a synthetic line
        through the agent while leaving that vehicle attached to its original
        lane and waypoints.  UE then either steered it back toward the old lane
        or the ordinary avoidance controller braked for the pedestrian.  This
        selector instead uses a staged route's actual centerline/crosswalk
        intersection and places the car upstream on that same straight segment.
        """
        routes = getattr(self, '_signal_vehicle_routes', {}) or {}

        agent_position = Vector(
            event['agent_position']['x'],
            event['agent_position']['y'],
        )
        crosswalk_axis = (crosswalk.end - crosswalk.start).normalize()
        current_projection = float(
            event.get(
                'crosswalk_projection',
                (agent_position - crosswalk.start).dot(crosswalk_axis)
                / max(crosswalk.start.distance(crosswalk.end), 1e-6),
            )
        )
        entry_direction = event.get('crosswalk_entry_direction')
        if entry_direction not in (-1, 1):
            raw_direction = event.get('agent_direction') or {}
            agent_direction = Vector(
                raw_direction.get('x', 0.0),
                raw_direction.get('y', 0.0),
            )
            alignment = agent_direction.dot(crosswalk_axis)
            entry_direction = (
                1 if alignment > 0.05
                else -1 if alignment < -0.05
                else 1 if current_projection <= 0.5
                else -1
            )

        crosswalk_length = crosswalk.start.distance(crosswalk.end)
        agent_speed = max(float(event.get('agent_speed_cm_s') or 200.0), 1.0)
        candidates = []
        for vehicle in vehicles:
            route = routes.get(vehicle.id)
            if route is None:
                continue
            route_points = [route['approach_start'], *route['path_points']]
            for segment_index, (start, end) in enumerate(
                zip(route_points, route_points[1:])
            ):
                intersection = self._segment_intersection_point(
                    start,
                    end,
                    crosswalk.start,
                    crosswalk.end,
                )
                if intersection is None:
                    continue
                target, route_projection, target_projection = intersection
                ahead_fraction = (
                    target_projection - current_projection
                ) * entry_direction
                segment = end - start
                segment_length = segment.length()
                if segment_length <= 1e-6:
                    continue
                route_direction = segment.normalize()
                upstream_distance = segment_length * route_projection
                signed_distance_cm = ahead_fraction * crosswalk_length
                ahead_distance_cm = max(0.0, signed_distance_cm)
                time_to_target_s = ahead_distance_cm / agent_speed
                available_launch_distance = max(0.0, upstream_distance - 5.0)
                # The configured launch range describes the physical distance
                # from the vehicle to the lane/crosswalk intercept. Do not
                # silently shorten it to fit a small curved route segment.
                launch_distance = float(requested_launch_distance_cm)
                if available_launch_distance + 1e-6 < launch_distance:
                    continue
                launch_position = target - route_direction * launch_distance
                # Prefer the next lane in the pedestrian's travel direction.
                # If all lanes have just been crossed when WALK expires, use
                # the nearest passed real lane so the whole authored crossing
                # retains the configured vehicle consequence.
                score = (
                    0 if signed_distance_cm >= 0.0 else 1,
                    round(abs(signed_distance_cm), 3),
                    abs(
                        time_to_target_s
                        - launch_distance
                        / self.red_light_conflict_nominal_speed_cm_s
                    ),
                    vehicle.id,
                    segment_index,
                )
                candidates.append(
                    (
                        score,
                        {
                            'vehicle': vehicle,
                            'route': route,
                            'route_segment_index': segment_index,
                            'launch_position': launch_position,
                            'target_position': target,
                            'direction': route_direction,
                            'launch_distance_cm': launch_distance,
                            'target_crosswalk_projection': target_projection,
                            'agent_distance_to_target_cm': ahead_distance_cm,
                            'signed_agent_distance_to_target_cm': signed_distance_cm,
                            'target_behind_agent': signed_distance_cm < 0.0,
                            'estimated_agent_arrival_s': time_to_target_s,
                        },
                    )
                )
        if not candidates:
            # Illegal-crossing events can happen on any authored road, not
            # only on one of the few routes staged for background signal
            # traffic. When the virtual crossing was built from a nearby
            # authored lane, reuse one staged vehicle actor but keep the
            # consequence trajectory on that exact lane.
            authored_lane = getattr(crosswalk, 'authored_lane', None)
            if authored_lane is not None and vehicles:
                lane_start = getattr(authored_lane, 'start', None)
                lane_end = getattr(authored_lane, 'end', None)
                if lane_start is not None and lane_end is not None:
                    lane_delta = lane_end - lane_start
                    lane_length = lane_delta.length()
                    if lane_length > 1e-6:
                        lane_direction = lane_delta.normalize()
                        target = getattr(crosswalk, 'lane_target', None)
                        if target is None:
                            intersection = self._segment_intersection_point(
                                lane_start,
                                lane_end,
                                crosswalk.start,
                                crosswalk.end,
                            )
                            target = (
                                intersection[0]
                                if intersection is not None
                                else None
                            )
                        if target is not None:
                            upstream_distance = (
                                target - lane_start
                            ).dot(lane_direction)
                            available_launch_distance = max(
                                0.0,
                                upstream_distance - 5.0,
                            )
                            launch_distance = float(
                                requested_launch_distance_cm
                            )
                            # Generated lanes also stop short of the upstream
                            # intersection mouth. An illegal crossing near a
                            # lane start can therefore have less finite-lane
                            # runway than the sampled launch distance even
                            # though the same straight authored centreline
                            # continues through the real road mouth. Allow
                            # only that bounded upstream continuation; the car
                            # still travels on the authored lane axis.
                            lane_start_upstream_extension_cm = max(
                                0.0,
                                launch_distance - upstream_distance,
                            )
                            if (
                                available_launch_distance + 1e-6
                                >= launch_distance
                                or (
                                    0.0 < lane_start_upstream_extension_cm
                                    <= 3000.0 + 1e-6
                                )
                            ):
                                target_projection = (
                                    (target - crosswalk.start).dot(
                                        crosswalk_axis
                                    )
                                    / max(crosswalk_length, 1e-6)
                                )
                                ahead_fraction = (
                                    target_projection - current_projection
                                ) * entry_direction
                                if math.isfinite(ahead_fraction):
                                    signed_distance_cm = (
                                        ahead_fraction * crosswalk_length
                                    )
                                    # An illegal-crossing crosswalk is a
                                    # short virtual segment centred on the
                                    # nearest authored vehicle lane.  Its
                                    # axis is perpendicular to that lane, so
                                    # projection on the virtual segment can
                                    # say the lane target is still ahead even
                                    # after a diagonal pedestrian trajectory
                                    # has already passed it.  Use the actual
                                    # movement vector for the signed progress
                                    # that controls staged-car release.
                                    raw_movement = (
                                        event.get('movement_direction')
                                        or event.get('agent_direction')
                                        or {}
                                    )
                                    movement = Vector(
                                        float(raw_movement.get('x', 0.0)),
                                        float(raw_movement.get('y', 0.0)),
                                    )
                                    if movement.length() > 1e-6:
                                        signed_distance_cm = (
                                            target - agent_position
                                        ).dot(movement.normalize())
                                    ahead_distance_cm = max(
                                        0.0,
                                        signed_distance_cm,
                                    )
                                    vehicle = min(
                                        vehicles,
                                        key=lambda item: item.id,
                                    )
                                    return {
                                        'vehicle': vehicle,
                                        'route': {
                                            'incoming_lane': authored_lane,
                                            'outgoing_lane': authored_lane,
                                            'approach_start': lane_start,
                                            'path_points': [lane_end],
                                            'route_source': (
                                                'authored_lane_illegal_crossing'
                                            ),
                                        },
                                        'route_segment_index': 0,
                                        'launch_position': (
                                            target
                                            - lane_direction * launch_distance
                                        ),
                                        'target_position': target,
                                        'direction': lane_direction,
                                        'launch_distance_cm': launch_distance,
                                        'target_crosswalk_projection': (
                                            target_projection
                                        ),
                                        'agent_distance_to_target_cm': (
                                            ahead_distance_cm
                                        ),
                                        'signed_agent_distance_to_target_cm': (
                                            signed_distance_cm
                                        ),
                                        'target_behind_agent': (
                                            signed_distance_cm < 0.0
                                        ),
                                        'estimated_agent_arrival_s': (
                                            ahead_distance_cm / agent_speed
                                        ),
                                        'lane_endpoint_extension_cm': max(
                                            0.0,
                                            (target - lane_end).dot(lane_direction),
                                        ),
                                        'lane_start_upstream_extension_cm': (
                                            lane_start_upstream_extension_cm
                                        ),
                                    }
        if (
            not candidates
            and str(getattr(crosswalk, 'id', '')).startswith(
                'illegal-crossing-'
            )
            and getattr(crosswalk, 'authored_lane', None) is None
        ):
            # The closest staged route can be a short turning spline that
            # cannot hold the sampled 3--9 m launch. Retry the nearest authored
            # straight lane/road-mouth extension with sufficient upstream
            # runway before declaring an illegal-crossing response unavailable.
            authored_crosswalk = self._road_entry_conflict_crosswalk(
                event,
                authored_only=True,
                authored_minimum_upstream_cm=requested_launch_distance_cm,
            )
            if authored_crosswalk is not None:
                return self._lane_aligned_conflict_launch(
                    event,
                    authored_crosswalk,
                    vehicles,
                    requested_launch_distance_cm,
                )
        if not candidates:
            # Endpoint crosswalks are authored beyond the finite TrafficLane
            # segment, inside the road mouth between ``lane.end`` and the
            # intersection centre.  Such a crossing has no outgoing lane, so
            # it cannot appear in ``_signal_vehicle_routes`` even though an
            # approaching car has a real, unambiguous centreline through the
            # zebra.  Extend only an incoming lane on the *same authored road*
            # and only in its forward direction.  This is deliberately much
            # narrower than the removed synthetic perpendicular fallback: no
            # arbitrary line through the pedestrian is accepted.
            traffic_lanes = list(
                getattr(self.traffic_controller, 'lanes', []) or []
            )
            endpoint_candidates = []
            for lane in traffic_lanes:
                if getattr(lane, 'road_id', None) != getattr(
                    crosswalk, 'road_id', None
                ):
                    continue
                lane_delta = lane.end - lane.start
                lane_length = lane_delta.length()
                if lane_length <= 1e-6:
                    continue
                lane_direction = lane_delta.normalize()
                extension_end = lane.end + lane_direction * 3000.0
                intersection = self._segment_intersection_point(
                    lane.start,
                    extension_end,
                    crosswalk.start,
                    crosswalk.end,
                )
                if intersection is None:
                    continue
                target, route_projection, target_projection = intersection
                # The fallback is specifically for the road mouth after the
                # authored lane endpoint, never for a crossing behind or in
                # the middle of a lane (those are covered by staged routes).
                distance_past_lane_end = (target - lane.end).dot(
                    lane_direction
                )
                if distance_past_lane_end < -1e-4:
                    continue
                ahead_fraction = (
                    target_projection - current_projection
                ) * entry_direction
                available_upstream = (
                    target - lane.start
                ).dot(lane_direction)
                available_launch_distance = max(0.0, available_upstream - 5.0)
                launch_distance = float(requested_launch_distance_cm)
                if available_launch_distance + 1e-6 < launch_distance:
                    continue
                launch_position = target - lane_direction * launch_distance
                signed_distance_cm = ahead_fraction * crosswalk_length
                ahead_distance_cm = max(0.0, signed_distance_cm)
                time_to_target_s = ahead_distance_cm / agent_speed
                endpoint_candidates.append(
                    (
                        (
                            0 if signed_distance_cm >= 0.0 else 1,
                            round(abs(signed_distance_cm), 3),
                            abs(
                                time_to_target_s
                                - launch_distance
                                / self.red_light_conflict_nominal_speed_cm_s
                            ),
                            lane.id,
                        ),
                        lane,
                        target,
                        target_projection,
                        lane_direction,
                        launch_position,
                        launch_distance,
                        ahead_distance_cm,
                        time_to_target_s,
                        distance_past_lane_end,
                        signed_distance_cm,
                    )
                )
            if endpoint_candidates and vehicles:
                (
                    _,
                    lane,
                    target,
                    target_projection,
                    lane_direction,
                    launch_position,
                    launch_distance,
                    ahead_distance_cm,
                    time_to_target_s,
                    distance_past_lane_end,
                    signed_distance_cm,
                ) = min(endpoint_candidates, key=lambda item: item[0])
                vehicle = min(vehicles, key=lambda item: item.id)
                return {
                    'vehicle': vehicle,
                    'route': {
                        'incoming_lane': lane,
                        # Endpoint approaches intentionally have no outgoing
                        # road.  Retain the lane object for stable audit IDs.
                        'outgoing_lane': lane,
                        'approach_start': lane.start,
                        'path_points': [lane.end, target],
                        'route_source': 'authored_lane_endpoint_extension',
                    },
                    'route_segment_index': 0,
                    'launch_position': launch_position,
                    'target_position': target,
                    'direction': lane_direction,
                    'launch_distance_cm': launch_distance,
                    'target_crosswalk_projection': target_projection,
                    'agent_distance_to_target_cm': ahead_distance_cm,
                    'signed_agent_distance_to_target_cm': signed_distance_cm,
                    'target_behind_agent': signed_distance_cm < 0.0,
                    'estimated_agent_arrival_s': time_to_target_s,
                    'lane_endpoint_extension_cm': distance_past_lane_end,
                }
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _record_vehicle_conflict_event(self, record):
        """Store one launch record in aggregate and trigger-specific logs."""
        aggregate = getattr(self, 'vehicle_conflict_events', None)
        if aggregate is None:
            aggregate = []
            self.vehicle_conflict_events = aggregate
        aggregate.append(record)
        trigger_type = record.get('trigger_type')
        if trigger_type == 'illegal_crossing':
            targeted = getattr(self, 'illegal_crossing_conflict_vehicle_events', None)
            if targeted is None:
                targeted = []
                self.illegal_crossing_conflict_vehicle_events = targeted
        else:
            targeted = getattr(self, 'red_light_conflict_vehicle_events', None)
            if targeted is None:
                targeted = []
                self.red_light_conflict_vehicle_events = targeted
        targeted.append(record)

    def _handle_red_light_violation(self, event, crosswalk):
        """Launch one lane-aligned consequence car for a red-light entry."""
        use_lane_aligned = (
            not getattr(self, 'red_light_conflict_vehicle_enabled', True)
            or hasattr(self, 'red_light_conflict_vehicle_probability')
            or hasattr(self, 'red_light_conflict_collision_radius_cm')
            or hasattr(self, 'red_light_conflict_launch_distance_min_cm')
        )
        if not use_lane_aligned:
            return self._handle_legacy_conflict_trigger(event, crosswalk)
        return self._handle_conflict_trigger(
            event,
            crosswalk,
            self.red_light_conflict_vehicle_events,
            trigger_type='red_light_violation',
        )

    def _handle_legacy_conflict_trigger(self, event, crosswalk):
        """Preserve the benchmark-clean launch contract for legacy callers.

        Fully initialized managers always take the lane-aligned path above.
        This adapter is limited to older integrations that construct a manager
        without the new collision configuration fields.
        """
        if not getattr(self, 'red_light_conflict_vehicle_enabled', True):
            return None
        trigger_type = 'red_light_violation'
        targeted = self.red_light_conflict_vehicle_events
        active_conflicts = list(
            getattr(self, '_active_vehicle_conflicts', []) or []
        )
        active_ids = {
            item['vehicle'].id for item in active_conflicts
            if item.get('vehicle') is not None
        }
        retired_ids = set(
            getattr(self, '_retired_signal_vehicle_ids', set()) or set()
        )
        vehicles = [
            vehicle
            for vehicle in list(
                getattr(self.traffic_controller, 'vehicles', []) or []
            )
            if vehicle.id not in active_ids and vehicle.id not in retired_ids
        ]
        record = {
            'trigger_type': trigger_type,
            'event_id': event.get('event_id'),
            'crosswalk_id': event.get('crosswalk_id'),
            'sim_time_s': event.get('sim_time_s'),
            'pedestrian_state': event.get('pedestrian_state'),
            'crosswalk_projection': event.get('crosswalk_projection'),
            'status': 'unavailable',
        }
        if not vehicles:
            targeted.append(record)
            aggregate = getattr(self, 'vehicle_conflict_events', None)
            if aggregate is not None and aggregate is not targeted:
                aggregate.append(record)
            event['conflict_vehicle'] = record
            return record

        vehicle = min(vehicles, key=lambda candidate: candidate.id)
        if crosswalk is not None:
            pedestrian_axis = (crosswalk.end - crosswalk.start).normalize()
        else:
            raw_direction = event.get('movement_direction') or {}
            pedestrian_axis = Vector(
                float(raw_direction.get('x', 0.0)),
                float(raw_direction.get('y', 0.0)),
            )
            if pedestrian_axis.length() <= 1e-6:
                pedestrian_axis = Vector(1.0, 0.0)
            pedestrian_axis = pedestrian_axis.normalize()
        road_axis = Vector(-pedestrian_axis.y, pedestrian_axis.x).normalize()
        if len(getattr(self, 'vehicle_conflict_events', [])) % 2:
            road_axis = road_axis * -1

        minimum_distance = float(
            getattr(self, 'conflict_vehicle_min_launch_distance_cm', 300.0)
        )
        maximum_distance = float(
            getattr(self, 'conflict_vehicle_max_launch_distance_cm', 900.0)
        )
        if minimum_distance > maximum_distance:
            minimum_distance, maximum_distance = maximum_distance, minimum_distance
        rng = getattr(self, '_conflict_vehicle_rng', None)
        if rng is None:
            rng = random.Random(int(getattr(self, 'seed', 0)) ^ 0x52414345)
            self._conflict_vehicle_rng = rng
        launch_clearance_cm = rng.uniform(minimum_distance, maximum_distance)
        impact_radius_cm = float(
            getattr(self, 'conflict_vehicle_impact_radius_cm', 100.0)
        )
        launch_center_distance_cm = launch_clearance_cm + impact_radius_cm
        raw_agent_position = event.get('agent_position') or {}
        agent_position = Vector(
            float(raw_agent_position.get('x', 0.0)),
            float(raw_agent_position.get('y', 0.0)),
        )
        launch_position = agent_position - road_axis * launch_center_distance_cm
        yaw = math.degrees(math.atan2(road_axis.y, road_axis.x))
        vehicle.position = launch_position
        vehicle.direction = yaw
        vehicle.state = VehicleState.MOVING
        vehicle.set_attributes(1.0, 0.0, 0.0)

        name = self.signal_traffic_communicator.get_vehicle_name(vehicle.id)
        unrealcv = self.signal_traffic_communicator.unrealcv
        actor_respawned = False
        get_objects = getattr(unrealcv, 'get_objects', None)
        if callable(get_objects) and name not in {
            str(item) for item in get_objects()
        }:
            vehicle_reference = getattr(vehicle, 'vehicle_reference', None)
            if not vehicle_reference:
                raise RuntimeError(
                    f'Conflict vehicle {vehicle.id} has no live UE actor and '
                    'no vehicle_reference for respawn'
                )
            unrealcv.spawn_bp_asset(vehicle_reference, name)
            if name not in {str(item) for item in get_objects()}:
                raise RuntimeError(
                    f'Conflict vehicle actor {name} did not appear after respawn'
                )
            actor_respawned = True
        unrealcv.set_location((launch_position.x, launch_position.y, 0), name)
        unrealcv.set_orientation((0, yaw, 0), name)
        unrealcv.set_collision(name, True)
        unrealcv.set_movable(name, True)
        self.signal_traffic_communicator.update_vehicle(
            vehicle.id, 1.0, 0.0, 0.0
        )
        record.update({
            'status': 'launched',
            'vehicle_id': vehicle.id,
            'vehicle_name': name,
            'vehicle_actor_respawned': actor_respawned,
            'launch_position': {
                'x': launch_position.x,
                'y': launch_position.y,
            },
            'target_position': {
                'x': agent_position.x,
                'y': agent_position.y,
            },
            'road_direction': {'x': road_axis.x, 'y': road_axis.y},
            'launch_distance_cm': round(launch_clearance_cm, 2),
            'launch_clearance_cm': round(launch_clearance_cm, 2),
            'launch_center_distance_cm': round(launch_center_distance_cm, 2),
            'launch_distance_range_cm': [
                round(minimum_distance, 2), round(maximum_distance, 2)
            ],
            'minimum_agent_distance_cm': round(
                launch_center_distance_cm, 2
            ),
            'impact_zone_reached': False,
        })
        targeted.append(record)
        aggregate = getattr(self, 'vehicle_conflict_events', None)
        if aggregate is not None and aggregate is not targeted:
            aggregate.append(record)
        event['conflict_vehicle'] = record
        active = {
            'vehicle': vehicle,
            'record': record,
            'launch_position': launch_position,
            'target_position': Vector(agent_position.x, agent_position.y),
            'direction': road_axis,
            'yaw': yaw,
            'launch_sim_time_s': float(event.get('sim_time_s') or 0.0),
            'update_count': 0,
            'stalled_update_count': 0,
            'best_travelled_cm': 0.0,
            'previous_vehicle_position': Vector(
                launch_position.x, launch_position.y
            ),
            'previous_agent_position': Vector(
                agent_position.x, agent_position.y
            ),
        }
        active_conflicts.append(active)
        self._active_vehicle_conflicts = active_conflicts
        self._active_red_light_conflict = active
        return record

    def _road_entry_conflict_crosswalk(
        self,
        event,
        *,
        authored_only=False,
        authored_minimum_upstream_cm=0.0,
    ):
        """Build a short virtual crossing through the nearest real car lane."""
        raw_position = event.get('agent_position') or {}
        point = Vector(
            float(raw_position.get('x', 0.0)),
            float(raw_position.get('y', 0.0)),
        )
        candidates = []
        staged_routes = (
            (getattr(self, '_signal_vehicle_routes', {}) or {}).items()
            if not authored_only
            else ()
        )
        for vehicle_id, route in staged_routes:
            route_points = [route['approach_start'], *route['path_points']]
            for segment_index, (start, end) in enumerate(
                zip(route_points, route_points[1:])
            ):
                segment = end - start
                length_sq = segment.dot(segment)
                if length_sq <= 1e-9:
                    continue
                projection = max(
                    0.0,
                    min(1.0, (point - start).dot(segment) / length_sq),
                )
                lane_point = start + segment * projection
                lane_distance = point.distance(lane_point)
                candidates.append(
                    (
                        lane_distance,
                        vehicle_id,
                        segment_index,
                        lane_point,
                        segment.normalize(),
                        None,
                    )
                )
        # Signal traffic intentionally stages only a few background routes.
        # Illegal crossing is a global rule, so also search every authored
        # vehicle lane and preserve that lane for the launch fallback.
        for lane in list(getattr(self.traffic_controller, 'lanes', []) or []):
            start = getattr(lane, 'start', None)
            end = getattr(lane, 'end', None)
            if start is None or end is None:
                continue
            lane_segment = end - start
            lane_length = lane_segment.length()
            if lane_length <= 1e-6:
                continue
            lane_direction = lane_segment.normalize()
            # Generated lanes stop before the intersection mouth. Continue
            # only their forward authored centerline through that mouth, as
            # the existing endpoint-crosswalk launch path already does.
            lookup_end = end + lane_direction * 3000.0
            segment = lookup_end - start
            length_sq = segment.dot(segment)
            projection = max(
                0.0,
                min(1.0, (point - start).dot(segment) / length_sq),
            )
            lane_point = start + segment * projection
            lane_distance = point.distance(lane_point)
            available_upstream = (
                lane_point - start
            ).dot(lane_direction)
            if (
                float(authored_minimum_upstream_cm) > 0.0
                and available_upstream - 5.0 + 1e-6
                < float(authored_minimum_upstream_cm)
            ):
                continue
            candidates.append(
                (
                    lane_distance,
                    -1,
                    int(getattr(lane, 'id', 0)),
                    lane_point,
                    lane_direction,
                    lane,
                )
            )
        if not candidates:
            return None
        (
            lane_distance,
            vehicle_id,
            segment_index,
            lane_point,
            lane_direction,
            authored_lane,
        ) = min(candidates, key=lambda item: item[:3])
        maximum_lane_distance_cm = max(
            2.0 * ROADWAY_OCCUPANCY_GUARD_HALF_WIDTH_CM,
            self.red_light_conflict_collision_radius_cm + 200.0,
        )
        if lane_distance > maximum_lane_distance_cm:
            return None
        perpendicular = Vector(-lane_direction.y, lane_direction.x)
        half_length = maximum_lane_distance_cm + 150.0
        return SimpleNamespace(
            id=(
                f'illegal-crossing-authored-lane-{authored_lane.id}'
                if authored_lane is not None
                else f'illegal-crossing-lane-{vehicle_id}-{segment_index}'
            ),
            start=lane_point - perpendicular * half_length,
            end=lane_point + perpendicular * half_length,
            lane_distance_cm=lane_distance,
            lane_vehicle_id=vehicle_id,
            lane_segment_index=segment_index,
            road_id=(
                getattr(authored_lane, 'road_id', None)
                if authored_lane is not None
                else None
            ),
            authored_lane=authored_lane,
            lane_target=lane_point,
        )

    def _handle_illegal_crossing(self, event):
        """Launch one lane-aligned car for an off-crosswalk road entry."""
        crosswalk = self._road_entry_conflict_crosswalk(event)
        if crosswalk is None:
            record = {
                'event_id': event.get('event_id'),
                'trigger_type': 'illegal_crossing',
                'sim_time_s': event.get('sim_time_s'),
                'disposition': 'unavailable',
                'status': 'unavailable',
                'unavailable_reason': 'no_nearby_staged_vehicle_lane',
            }
            self.illegal_crossing_conflict_vehicle_events.append(record)
            return record
        routed_event = dict(event)
        routed_event['crosswalk_id'] = crosswalk.id
        routed_event['vehicle_lane_distance_cm'] = round(
            float(crosswalk.lane_distance_cm),
            2,
        )
        return self._handle_conflict_trigger(
            routed_event,
            crosswalk,
            self.illegal_crossing_conflict_vehicle_events,
            trigger_type='illegal_crossing',
        )

    def _handle_conflict_trigger(
        self,
        event,
        crosswalk,
        event_records,
        *,
        trigger_type,
    ):
        """Launch one real-lane vehicle for a pedestrian traffic violation."""
        record = {
            'event_id': event.get('event_id'),
            'trigger_type': trigger_type,
            'crosswalk_id': event.get('crosswalk_id'),
            'sim_time_s': event.get('sim_time_s'),
            'pedestrian_state': event.get('pedestrian_state'),
            'crosswalk_projection': event.get('crosswalk_projection'),
            'disposition': 'unavailable',
            'status': 'unavailable',
        }
        if not getattr(self, 'red_light_conflict_vehicle_enabled', True):
            record['disposition'] = 'disabled'
            record['status'] = 'disabled'
            event_records.append(record)
            return record

        active_conflicts = list(
            getattr(self, '_active_vehicle_conflicts', []) or []
        )
        legacy_active = getattr(self, '_active_red_light_conflict', None)
        if legacy_active is not None and legacy_active not in active_conflicts:
            active_conflicts.append(legacy_active)
        active_vehicle_ids = {
            active['vehicle'].id
            for active in active_conflicts
            if active.get('vehicle') is not None
        }
        pooled_vehicles = dict(
            getattr(self, '_retired_conflict_vehicles', {}) or {}
        )
        vehicles_by_id = {
            vehicle.id: vehicle
            for vehicle in list(
                getattr(self.traffic_controller, 'vehicles', []) or []
            )
        }
        vehicles_by_id.update(pooled_vehicles)
        vehicles = [
            vehicle
            for vehicle_id, vehicle in sorted(vehicles_by_id.items())
            if vehicle_id not in active_vehicle_ids
        ]
        consequence_rng = getattr(self, '_traffic_consequence_rng', None)
        if consequence_rng is None:
            consequence_rng = random.Random(
                int(getattr(self, 'seed', 0)) ^ 0x5A17C0DE
            )
            self._traffic_consequence_rng = consequence_rng
        probability_draw = consequence_rng.random()
        record['vehicle_probability'] = (
            self.red_light_conflict_vehicle_probability
        )
        record['vehicle_probability_draw'] = round(probability_draw, 6)
        draw_index = int(
            getattr(self, '_traffic_consequence_draw_count', 0) or 0
        )
        record['vehicle_probability_rng_seed'] = getattr(
            self, '_traffic_consequence_rng_seed', None
        )
        record['vehicle_probability_draw_index'] = draw_index
        self._traffic_consequence_draw_count = draw_index + 1
        if probability_draw >= self.red_light_conflict_vehicle_probability:
            record['disposition'] = 'probability_skipped'
            record['status'] = 'probability_skipped'
            event_records.append(record)
            return record
        if not vehicles:
            event_records.append(record)
            self.logger.warning(
                '%s recorded, but no signal-controlled vehicle is available '
                'for the conflict response.',
                trigger_type,
            )
            return record

        agent_position = Vector(
            event['agent_position']['x'],
            event['agent_position']['y'],
        )
        requested_launch_distance_cm = consequence_rng.uniform(
            self.red_light_conflict_launch_distance_min_cm,
            self.red_light_conflict_launch_distance_max_cm,
        )
        lane_launch = self._lane_aligned_conflict_launch(
            event,
            crosswalk,
            vehicles,
            requested_launch_distance_cm,
        )
        if lane_launch is None:
            # Do not invent a perpendicular trajectory through the agent.  A
            # synthetic fallback can steer a staged car off its authored lane
            # and makes the consequence look unrelated to the traffic system.
            # Missing route coverage is an explicit, auditable disposition.
            record.update(
                {
                    'disposition': 'unavailable',
                    'status': 'unavailable',
                    'unavailable_reason': 'no_lane_aligned_conflict_route',
                    'requested_launch_distance_cm': round(
                        requested_launch_distance_cm,
                        2,
                    ),
                }
            )
            event_records.append(record)
            self.logger.warning(
                '%s %s has no staged vehicle route that intersects crossing '
                '%s ahead of the pedestrian.',
                trigger_type,
                event.get('event_id'),
                event.get('crosswalk_id'),
            )
            return record

        vehicle = lane_launch['vehicle']
        road_axis = lane_launch['direction']
        launch_distance_cm = lane_launch['launch_distance_cm']
        launch_position = lane_launch['launch_position']
        target_position = lane_launch['target_position']
        route = lane_launch['route']
        control_mode = 'lane_aligned_intercept'
        estimated_agent_arrival_s = float(
            lane_launch['estimated_agent_arrival_s']
        )
        lane_metadata = {
            'incoming_lane_id': route['incoming_lane'].id,
            'outgoing_lane_id': route['outgoing_lane'].id,
            'route_source': route.get(
                'route_source',
                'staged_intersection_route',
            ),
            'route_segment_index': lane_launch['route_segment_index'],
            'target_crosswalk_projection': round(
                lane_launch['target_crosswalk_projection'], 4
            ),
            'agent_distance_to_target_cm': round(
                lane_launch['agent_distance_to_target_cm'], 2
            ),
            'estimated_agent_arrival_s': round(
                lane_launch['estimated_agent_arrival_s'], 3
            ),
            'signed_agent_distance_to_target_cm': round(
                lane_launch.get('signed_agent_distance_to_target_cm', 0.0),
                2,
            ),
            'target_behind_agent': bool(
                lane_launch.get('target_behind_agent', False)
            ),
        }
        if 'lane_endpoint_extension_cm' in lane_launch:
            lane_metadata['lane_endpoint_extension_cm'] = round(
                lane_launch['lane_endpoint_extension_cm'],
                2,
            )
        if 'lane_start_upstream_extension_cm' in lane_launch:
            lane_metadata['lane_start_upstream_extension_cm'] = round(
                lane_launch['lane_start_upstream_extension_cm'],
                2,
            )
        vehicle_arrival_time_s = (
            launch_distance_cm
            / self.red_light_conflict_nominal_speed_cm_s
        )
        agent_speed_cm_s = max(
            float(event.get('agent_speed_cm_s') or 200.0),
            1.0,
        )
        # Model decisions, turns and waits make a one-shot ETA at violation
        # time unreliable.  Keep the staged car braked on its real lane until
        # the pedestrian itself has entered the conflict envelope around the
        # real lane/crosswalk intersection.  Releasing earlier by adding the
        # vehicle ETA allowed a 100 cm pedestrian step to land at 300 cm from
        # the target while the collision radius was only 100 cm; the car then
        # reached the target and was retired with a deterministic 50 cm miss.
        # The envelope-only threshold remains lane aligned, works with swept
        # collision checks, and guarantees that reaching the target actually
        # represents the requested collision consequence.
        release_trigger_distance_cm = (
            self.red_light_conflict_collision_radius_cm
        )
        initial_agent_distance_to_target_cm = agent_position.distance(
            target_position
        )
        initially_released = (
            lane_metadata['target_behind_agent']
            or initial_agent_distance_to_target_cm
            <= release_trigger_distance_cm
        )
        yaw = math.degrees(math.atan2(road_axis.y, road_axis.x))
        name = self.signal_traffic_communicator.get_vehicle_name(vehicle.id)
        unrealcv = self.signal_traffic_communicator.unrealcv
        actor_respawned = False
        try:
            vehicle_reference = getattr(
                vehicle,
                'vehicle_reference',
                None,
            )
            get_objects = getattr(unrealcv, 'get_objects', None)
            if callable(get_objects):
                existing_objects = {str(item) for item in get_objects()}
                if name not in existing_objects:
                    if not vehicle_reference:
                        raise RuntimeError(
                            f'Conflict vehicle {vehicle.id} has no live UE '
                            'actor and no vehicle_reference for respawn'
                        )
                    unrealcv.spawn_bp_asset(vehicle_reference, name)
                    respawned_objects = {str(item) for item in get_objects()}
                    if name not in respawned_objects:
                        raise RuntimeError(
                            f'Conflict vehicle actor {name} did not appear '
                            f'after respawning {vehicle_reference}'
                        )
                    actor_respawned = True
                    self.logger.warning(
                        'Respawned missing conflict vehicle actor %s from %s',
                        name,
                        vehicle_reference,
                    )
            # A managed car is already physics-enabled in the scene.  Moving
            # that live rigid body directly to Z=0 can overlap the road or a
            # nearby actor for one frame and inject a large contact impulse --
            # the visible result is the reported "vehicles raining" failure.
            # Quiesce exactly the one selected consequence car, preserve its
            # settled road height, assemble the new lane pose, and only then
            # re-enable collision/physics.  Other traffic actors are untouched.
            try:
                live_location = unrealcv.get_location(name)
            except ValueError:
                # UE can retain a destroyed actor name in ``vget /objects``
                # for one command cycle.  In that state the object-list check
                # above succeeds but ``vget /object/<name>/location`` returns
                # the literal string ``error``.  Recover only this explicit
                # stale-handle response; socket timeouts and other UE failures
                # must still propagate as infrastructure errors.
                if not vehicle_reference:
                    raise RuntimeError(
                        f'Conflict vehicle {vehicle.id} has a stale UE actor '
                        'handle and no vehicle_reference for respawn'
                    )
                destroy = getattr(unrealcv, 'destroy', None)
                if callable(destroy):
                    destroy(name)
                # Actor destruction is deferred until a later UE tick. A
                # same-name spawn in the next command can therefore attempt
                # to rename the new actor on top of the pending-kill object,
                # which is a fatal CoreUObject error. Wait for the name to
                # disappear from UnrealCV's object list; if it does not,
                # surface a retryable infrastructure error instead of
                # crashing UE.
                if callable(get_objects):
                    destroy_deadline = time.monotonic() + 2.0
                    while name in {str(item) for item in get_objects()}:
                        if time.monotonic() >= destroy_deadline:
                            raise RuntimeError(
                                f'Conflict vehicle actor {name} remained '
                                'visible after destroy; refusing unsafe '
                                'same-name respawn'
                            )
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)
                unrealcv.spawn_bp_asset(vehicle_reference, name)
                live_location = unrealcv.get_location(name)
                actor_respawned = True
                self.logger.warning(
                    'Respawned stale conflict vehicle actor %s from %s',
                    name,
                    vehicle_reference,
                )
            launch_z_cm = float(live_location[2])
            mutation_delay = max(
                0.0,
                float(
                    os.environ.get(
                        'SIMWORLD_ACTOR_MUTATION_SETTLE_SECONDS',
                        '0',
                    )
                ),
            )

            def settle():
                if mutation_delay:
                    time.sleep(mutation_delay)

            unrealcv.set_physics(name, False)
            unrealcv.set_collision(name, False)
            settle()
            vehicle.position = launch_position
            vehicle.direction = yaw
            # The packaged vehicle controller is not reliable enough for a
            # consequence shot, so advance this one actor kinematically. UE
            # collision remains enabled, while the 1 m swept-radius check
            # catches between-tick impacts that UE counters can miss.
            vehicle.state = VehicleState.STOPPED
            vehicle.set_attributes(0.0, 1.0, 0.0)
            unrealcv.set_location(
                (launch_position.x, launch_position.y, launch_z_cm),
                name,
            )
            settle()
            unrealcv.set_orientation((0, yaw, 0), name)
            settle()
            unrealcv.set_movable(name, True)
            settle()
            unrealcv.set_collision(name, True)
            settle()
            self.signal_traffic_communicator.update_vehicle(
                vehicle.id,
                0.0,
                1.0,
                0.0,
            )
        except Exception as exc:
            record.update(
                {
                    'disposition': 'failed',
                    'status': 'failed',
                    'vehicle_id': vehicle.id,
                    'vehicle_name': name,
                    'control_mode': control_mode,
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                    **lane_metadata,
                }
            )
            event_records.append(record)
            self.logger.exception(
                'Failed to launch conflict vehicle %s for violation %s',
                name,
                event.get('event_id'),
            )
            raise

        pooled_vehicles.pop(vehicle.id, None)
        self._retired_conflict_vehicles = pooled_vehicles
        record.update(
            {
                # A relocated, braked vehicle is staged, not launched. It
                # becomes a valid launch only when the pedestrian reaches the
                # conflict envelope and the vehicle begins its approach.
                'disposition': (
                    'launched' if initially_released else 'staged'
                ),
                'status': 'launched' if initially_released else 'staged',
                'vehicle_id': vehicle.id,
                'vehicle_name': name,
                'vehicle_actor_respawned': actor_respawned,
                'launch_position': {
                    'x': launch_position.x,
                    'y': launch_position.y,
                },
                'launch_z_cm': round(launch_z_cm, 2),
                'physics_quiesced_during_relocation': True,
                'physics_during_approach': False,
                'collision_during_approach': True,
                'collision_authority': (
                    'ue_counter_or_launched_vehicle_swept_radius'
                ),
                'motion_mode': 'lane_aligned_kinematic_intercept',
                'target_position': {
                    'x': target_position.x,
                    'y': target_position.y,
                },
                'road_direction': {'x': road_axis.x, 'y': road_axis.y},
                'launch_distance_cm': round(launch_distance_cm, 2),
                'requested_launch_distance_cm': round(
                    requested_launch_distance_cm,
                    2,
                ),
                'launch_distance_range_cm': [
                    self.red_light_conflict_launch_distance_min_cm,
                    self.red_light_conflict_launch_distance_max_cm,
                ],
                'collision_radius_cm': (
                    self.red_light_conflict_collision_radius_cm
                ),
                'minimum_agent_distance_cm': round(
                    launch_position.distance(agent_position), 2
                ),
                'agent_distance_to_target_at_launch_cm': round(
                    initial_agent_distance_to_target_cm,
                    2,
                ),
                'impact_zone_reached': False,
                'collision_triggered': False,
                'control_mode': control_mode,
                **lane_metadata,
            }
        )
        record['nominal_vehicle_speed_cm_s'] = (
            self.red_light_conflict_nominal_speed_cm_s
        )
        record['vehicle_arrival_time_s'] = round(vehicle_arrival_time_s, 3)
        record['release_mode'] = 'collision_envelope_intercept'
        record['release_trigger_distance_cm'] = round(
            release_trigger_distance_cm,
            2,
        )
        record['release_delay_s'] = 0.0 if initially_released else None
        record['release_wait_timeout_s'] = (
            getattr(
                self,
                'red_light_conflict_release_wait_timeout_s',
                12.0,
            )
        )
        record['released'] = initially_released
        if initially_released:
            record['released_sim_time_s'] = round(
                float(event.get('sim_time_s') or 0.0),
                3,
            )
        event_records.append(record)
        active = {
            'vehicle': vehicle,
            'record': record,
            'launch_position': launch_position,
            # The lane/crosswalk conflict point is the release and retirement plane.
            'target_position': target_position,
            'direction': road_axis,
            'control_mode': control_mode,
            'launch_sim_time_s': float(event.get('sim_time_s') or 0.0),
            'release_sim_time_s': (
                float(event.get('sim_time_s') or 0.0)
                if initially_released
                else None
            ),
            'last_vehicle_position': Vector(
                vehicle.position.x,
                vehicle.position.y,
            ),
            'last_agent_position': Vector(
                agent_position.x,
                agent_position.y,
            ),
            'vehicle_name': name,
            'launch_z_cm': launch_z_cm,
            'kinematic_motion': True,
        }
        # ``v_set_state`` is the final Blueprint command issued while staging
        # and can reactivate chassis simulation.  Reassert the kinematic pose
        # after that command so no unstable frame occurs before the next
        # controller update.
        self._stabilize_active_kinematic_conflict_vehicle(active)
        active_conflicts.append(active)
        self._active_vehicle_conflicts = active_conflicts
        self._active_red_light_conflict = active
        # Only consequence-active simulation is advanced in short chunks.
        # Ordinary navigation keeps its established timing behavior.
        agent = getattr(self, 'agent', None)
        if agent is not None:
            agent.red_light_conflict_active = True
        self.logger.error(
            '%s conflict vehicle %s for %s on crossing %s',
            'Launched' if initially_released else 'Staged',
            name,
            trigger_type,
            event.get('crosswalk_id'),
        )
        return record

    def _retire_conflict_vehicle(self, active, status):
        """Remove the one-shot consequence car after impact/target passage."""
        vehicle = active['vehicle']
        record = active['record']
        record['status'] = status
        record['retired_after_consequence'] = True
        record['retired_sim_time_s'] = round(
            float(getattr(self.agent, 'sim_time_elapsed', 0.0)),
            3,
        )
        communicator = getattr(self, 'signal_traffic_communicator', None)
        unrealcv = getattr(communicator, 'unrealcv', None)
        vehicle_id = getattr(vehicle, 'id', None)
        name = (
            communicator.get_vehicle_name(vehicle_id)
            if communicator is not None and vehicle_id is not None
            else None
        )
        if communicator is not None and hasattr(communicator, 'update_vehicle'):
            if vehicle_id is not None:
                communicator.update_vehicle(vehicle_id, 0.0, 1.0, 0.0)
        if unrealcv is not None and name is not None:
            if hasattr(unrealcv, 'set_physics'):
                unrealcv.set_physics(name, False)
            if hasattr(unrealcv, 'set_collision'):
                unrealcv.set_collision(name, False)
            if hasattr(unrealcv, 'set_movable'):
                unrealcv.set_movable(name, False)
            if hasattr(unrealcv, 'destroy'):
                unrealcv.destroy(name)
        current_lane = getattr(vehicle, 'current_lane', None)
        if vehicle in getattr(current_lane, 'vehicles', []):
            current_lane.remove_vehicle(vehicle)
        vehicle_manager = getattr(
            getattr(self, 'traffic_controller', None),
            'vehicle_manager',
            None,
        )
        if vehicle_manager is not None and vehicle in vehicle_manager.vehicles:
            vehicle_manager.vehicles.remove(vehicle)
        if not hasattr(self, '_retired_conflict_vehicles'):
            self._retired_conflict_vehicles = {}
        if vehicle_id is not None:
            self._retired_conflict_vehicles[vehicle_id] = vehicle
        if not hasattr(self, '_retired_signal_vehicle_ids'):
            self._retired_signal_vehicle_ids = set()
        if vehicle_id is not None:
            self._retired_signal_vehicle_ids.add(vehicle_id)
        remaining = [
            item
            for item in list(
                getattr(self, '_active_vehicle_conflicts', []) or []
            )
            if item is not active
        ]
        self._active_vehicle_conflicts = remaining
        self._active_red_light_conflict = remaining[-1] if remaining else None
        if getattr(self, 'agent', None) is not None:
            self.agent.red_light_conflict_active = bool(remaining)

    def _record_ue_vehicle_collision_outcome(self):
        """Attach an agent UE collision counter to its active vehicle event."""
        agent = getattr(self, 'agent', None)
        if (
            agent is None
            or getattr(agent, 'failure_reason', None) != 'vehicle_collision'
            or int(getattr(agent, 'vehicle_collision_count', 0) or 0) <= 0
        ):
            return False

        active = getattr(self, '_active_red_light_conflict', None)
        if active is None:
            return False
        record = active.get('record', {})
        if record.get('collision_triggered') is True:
            return True

        record.update({
            'collision_triggered': True,
            'collision_source': 'unreal_engine_counter',
            'collision_authority': 'unreal_engine_counter',
            'ue_vehicle_collision_count': int(agent.vehicle_collision_count),
            'collision_sim_time_s': round(
                float(getattr(agent, 'sim_time_elapsed', 0.0)),
                3,
            ),
        })
        self._retire_conflict_vehicle(active, 'collision')
        remaining = [
            item
            for item in list(
                getattr(self, '_active_vehicle_conflicts', []) or []
            )
            if item is not active
        ]
        self._active_vehicle_conflicts = remaining
        self._active_red_light_conflict = remaining[-1] if remaining else None
        return True

    @staticmethod
    def _swept_minimum_separation_cm(
        vehicle_start,
        vehicle_end,
        agent_start,
        agent_end,
    ):
        """Minimum separation of two linearly moving actors over one update.

        The simulator may advance for an entire model-inference interval before
        Python regains control. Sampling only the two endpoints can therefore
        miss a vehicle that crossed directly through the agent between them.
        """
        relative_start = vehicle_start - agent_start
        relative_motion = (
            (vehicle_end - vehicle_start) - (agent_end - agent_start)
        )
        denominator = relative_motion.dot(relative_motion)
        if denominator <= 1e-12:
            return relative_start.length()
        closest_fraction = max(
            0.0,
            min(1.0, -relative_start.dot(relative_motion) / denominator),
        )
        return (relative_start + relative_motion * closest_fraction).length()

    def _update_one_conflict_vehicle(self, active):
        """Advance one launched vehicle and retain it until terminal cleanup."""
        record = active.get('record', {})
        if (
            'release_trigger_distance_cm' not in record
            and 'collision_triggered' not in record
        ):
            return self._update_legacy_conflict_vehicle(active)
        vehicle = active['vehicle']
        current_sim_time_s = float(
            getattr(self.agent, 'sim_time_elapsed', 0.0)
        )
        elapsed = current_sim_time_s - active['launch_sim_time_s']
        if active.get('kinematic_motion') and active['record'].get(
            'released', True
        ):
            release_sim_time_s = active.get('release_sim_time_s')
            if release_sim_time_s is not None:
                elapsed_since_release = max(
                    0.0,
                    current_sim_time_s - float(release_sim_time_s),
                )
                commanded_travel_cm = (
                    self.red_light_conflict_nominal_speed_cm_s
                    * elapsed_since_release
                )
                commanded_position = (
                    active['launch_position']
                    + active['direction'] * commanded_travel_cm
                )
                self.signal_traffic_communicator.unrealcv.set_location(
                    (
                        commanded_position.x,
                        commanded_position.y,
                        active['launch_z_cm'],
                    ),
                    active['vehicle_name'],
                )
                vehicle.position = commanded_position
                active['record']['commanded_travel_cm'] = round(
                    commanded_travel_cm,
                    2,
                )
        travelled = (
            vehicle.position - active['launch_position']
        ).dot(active['direction'])
        best_travelled = float(active.get('best_travelled_cm', 0.0))
        if travelled > best_travelled + 1.0:
            active['best_travelled_cm'] = travelled
            active['stalled_update_count'] = 0
        else:
            active['stalled_update_count'] = int(
                active.get('stalled_update_count', 0)
            ) + 1
        target_position = active.get(
            'target_position',
            active.get('previous_agent_position', self.agent.position),
        )
        target_progress = (
            vehicle.position - target_position
        ).dot(active['direction'])
        agent_distance = vehicle.position.distance(self.agent.position)
        collision_distance = agent_distance
        collision_source = 'conflict_vehicle_distance_envelope'
        if active['record'].get('released', True):
            previous_vehicle = active.get(
                'last_vehicle_position',
                vehicle.position,
            )
            previous_agent = active.get(
                'last_agent_position',
                self.agent.position,
            )
            relative_start = previous_vehicle - previous_agent
            relative_end = vehicle.position - self.agent.position
            swept_distance = self._point_segment_distance(
                Vector(0.0, 0.0),
                relative_start,
                relative_end,
            )
            active['record']['minimum_swept_agent_distance_cm'] = round(
                min(
                    float(
                        active['record'].get(
                            'minimum_swept_agent_distance_cm',
                            float('inf'),
                        )
                    ),
                    swept_distance,
                ),
                2,
            )
            if swept_distance < collision_distance:
                collision_distance = swept_distance
                collision_source = 'conflict_vehicle_swept_envelope'
        active['record']['minimum_agent_distance_cm'] = round(
            min(
                float(active['record']['minimum_agent_distance_cm']),
                collision_distance,
            ),
            2,
        )
        collision_radius = self.red_light_conflict_collision_radius_cm
        if (
            collision_radius > 0.0
            and active['record'].get('released', True)
            and collision_distance <= collision_radius
            and not active['record']['impact_zone_reached']
        ):
            active['record']['impact_zone_reached'] = True
            active['record']['impact_zone_sim_time_s'] = round(
                float(getattr(self.agent, 'sim_time_elapsed', 0.0)),
                3,
            )
            active['record']['impact_zone_detection_source'] = collision_source
            active['record']['impact_zone_distance_cm'] = round(
                collision_distance,
                2,
            )
            active['record']['collision_triggered'] = True
            active['record']['collision_source'] = collision_source
            active['record']['collision_authority'] = (
                'launched_vehicle_swept_radius'
            )
            active['record']['collision_distance_cm'] = round(
                collision_distance,
                2,
            )
            active['record']['status'] = 'collision'
            queue_collision = getattr(
                self.agent,
                'queue_swept_vehicle_collision',
                None,
            )
            if callable(queue_collision):
                queue_collision()
            else:
                if not (
                    getattr(self.agent, 'failed', False)
                    and getattr(self.agent, 'failure_reason', None)
                    == 'vehicle_collision'
                ):
                    self.agent.vehicle_collision_count = int(
                        getattr(self.agent, 'vehicle_collision_count', 0)
                    ) + 1
                    self.agent.collision_count = int(
                        getattr(self.agent, 'collision_count', 0)
                    ) + 1
                self.agent.failed = True
                self.agent.success = False
                self.agent.failure_reason = 'vehicle_collision'
            self.vehicle_impact_zone_count = int(
                getattr(self, 'vehicle_impact_zone_count', 0)
            ) + 1
            self._retire_conflict_vehicle(active, 'collision')
            self.logger.error(
                'Conflict vehicle entered the %.1fcm swept collision radius '
                'at distance %.1fcm; terminating as vehicle_collision',
                collision_radius,
                collision_distance,
            )
            return False
        target_progress = (
            vehicle.position - target_position
        ).dot(active['direction'])
        active['record']['target_progress_cm'] = round(target_progress, 2)
        if (
            active['record'].get('released', True)
            and target_progress >= 0.0
            and not active['record'].get('target_reached', False)
        ):
            active['record']['target_reached'] = True
            active['record']['target_reached_sim_time_s'] = round(
                current_sim_time_s,
                3,
            )
            active['record']['status'] = 'target_reached_awaiting_ue_collision'
        if not active['record'].get('released', True):
            target_distance = self.agent.position.distance(
                active['target_position']
            )
            active['record']['agent_distance_to_target_cm'] = round(
                target_distance,
                2,
            )
            release_trigger_distance = float(
                active['record'].get('release_trigger_distance_cm', 0.0)
            )
            if target_distance > release_trigger_distance:
                vehicle.state = VehicleState.STOPPED
                vehicle.set_attributes(0.0, 1.0, 0.0)
                release_wait_timeout_s = float(
                    active['record'].get(
                        'release_wait_timeout_s',
                        getattr(
                            self,
                            'red_light_conflict_release_wait_timeout_s',
                            12.0,
                        ),
                    )
                )
                if elapsed >= release_wait_timeout_s:
                    active['record']['release_wait_elapsed_s'] = round(
                        elapsed,
                        3,
                    )
                    active['record']['release_reason'] = (
                        'release_wait_timeout'
                    )
                    # Every detected violation must produce a real launch.
                    # The envelope remains the preferred intercept timing, but
                    # a diagonal pedestrian path must not turn the consequence
                    # into an expired staged actor.
                else:
                    active['record']['release_wait_elapsed_s'] = round(
                        elapsed,
                        3,
                    )
                    self.signal_traffic_communicator.update_vehicle(
                        vehicle.id,
                        0.0,
                        1.0,
                        0.0,
                    )
                    self._stabilize_active_kinematic_conflict_vehicle(active)
                    active['last_vehicle_position'] = Vector(
                        vehicle.position.x,
                        vehicle.position.y,
                    )
                    active['last_agent_position'] = Vector(
                        self.agent.position.x,
                        self.agent.position.y,
                    )
                    return True

            active['record']['released'] = True
            active['record']['disposition'] = 'launched'
            active['record']['status'] = 'launched'
            active['record']['released_sim_time_s'] = round(
                current_sim_time_s,
                3,
            )
            active['record']['release_delay_s'] = round(elapsed, 3)
            active['release_sim_time_s'] = current_sim_time_s
            active['record']['agent_distance_to_target_at_release_cm'] = round(
                target_distance,
                2,
            )

        release_sim_time_s = active.get('release_sim_time_s')
        if release_sim_time_s is None:
            vehicle.state = VehicleState.STOPPED
            vehicle.set_attributes(0.0, 1.0, 0.0)
            self.signal_traffic_communicator.update_vehicle(
                vehicle.id,
                0.0,
                1.0,
                0.0,
            )
            self._stabilize_active_kinematic_conflict_vehicle(active)
            active['last_vehicle_position'] = Vector(
                vehicle.position.x,
                vehicle.position.y,
            )
            active['last_agent_position'] = Vector(
                self.agent.position.x,
                self.agent.position.y,
            )
            return True
        elapsed_since_release = max(
            0.0,
            current_sim_time_s - float(release_sim_time_s),
        )
        if elapsed_since_release >= 12.0 or travelled >= 2200.0:
            active['record']['travelled_cm'] = round(travelled, 2)
            active['record']['final_position'] = {
                'x': vehicle.position.x,
                'y': vehicle.position.y,
            }
            active['record']['completed_sim_time_s'] = round(
                float(getattr(self.agent, 'sim_time_elapsed', 0.0)),
                3,
            )
            self._retire_conflict_vehicle(active, 'completed')
            return False

        if active.get('kinematic_motion'):
            vehicle.state = VehicleState.STOPPED
            vehicle.set_attributes(0.0, 1.0, 0.0)
            self.signal_traffic_communicator.update_vehicle(
                vehicle.id,
                0.0,
                1.0,
                0.0,
            )
            self._stabilize_active_kinematic_conflict_vehicle(active)
        else:
            vehicle.state = VehicleState.MOVING
            vehicle.set_attributes(1.0, 0.0, 0.0)
            self.signal_traffic_communicator.update_vehicle(
                vehicle.id,
                1.0,
                0.0,
                0.0,
            )
        active['last_vehicle_position'] = Vector(
            vehicle.position.x,
            vehicle.position.y,
        )
        active['last_agent_position'] = Vector(
            self.agent.position.x,
            self.agent.position.y,
        )
        return True

    def _update_legacy_conflict_vehicle(self, active):
        """Advance a pre-lane-alignment conflict record without schema drift."""
        vehicle = active['vehicle']
        active['update_count'] = int(active.get('update_count', 0)) + 1
        elapsed = (
            float(getattr(self.agent, 'sim_time_elapsed', 0.0))
            - active['launch_sim_time_s']
        )
        travelled = (
            vehicle.position - active['launch_position']
        ).dot(active['direction'])
        best_travelled = float(active.get('best_travelled_cm', 0.0))
        if travelled > best_travelled + 1.0:
            active['best_travelled_cm'] = travelled
            active['stalled_update_count'] = 0
        else:
            active['stalled_update_count'] = int(
                active.get('stalled_update_count', 0)
            ) + 1
        target_position = active.get(
            'target_position',
            active.get('previous_agent_position', self.agent.position),
        )
        target_progress = (
            vehicle.position - target_position
        ).dot(active['direction'])
        previous_vehicle = active.get(
            'previous_vehicle_position', active['launch_position']
        )
        previous_agent = active.get(
            'previous_agent_position', self.agent.position
        )
        swept_distance = self._swept_minimum_separation_cm(
            previous_vehicle,
            vehicle.position,
            previous_agent,
            self.agent.position,
        )
        active['previous_vehicle_position'] = Vector(
            vehicle.position.x, vehicle.position.y
        )
        active['previous_agent_position'] = Vector(
            self.agent.position.x, self.agent.position.y
        )
        record = active['record']
        record['swept_minimum_agent_distance_cm'] = round(
            min(
                float(record.get(
                    'swept_minimum_agent_distance_cm', float('inf')
                )),
                swept_distance,
            ),
            2,
        )
        record['minimum_agent_distance_cm'] = round(
            min(
                float(record.get('minimum_agent_distance_cm', float('inf'))),
                vehicle.position.distance(self.agent.position),
                swept_distance,
            ),
            2,
        )
        impact_radius = float(
            getattr(self, 'conflict_vehicle_impact_radius_cm', 100.0)
        )
        if (
            swept_distance <= impact_radius
            and not record.get('impact_zone_reached', False)
        ):
            record['impact_zone_reached'] = True
            record['impact_zone_sim_time_s'] = round(
                float(getattr(self.agent, 'sim_time_elapsed', 0.0)), 3
            )
            record['impact_zone_detection_source'] = 'swept_actor_trajectory'
            record['collision_triggered'] = True
            record['collision_source'] = 'swept_actor_trajectory'
            record['collision_authority'] = 'launched_vehicle_swept_radius'
            record['collision_distance_cm'] = round(swept_distance, 2)
            record['status'] = 'collision'
            record['travelled_cm'] = round(travelled, 2)
            record['final_position'] = {
                'x': vehicle.position.x,
                'y': vehicle.position.y,
            }
            record['completed_sim_time_s'] = round(
                float(getattr(self.agent, 'sim_time_elapsed', 0.0)), 3
            )
            self.vehicle_impact_zone_count = int(
                getattr(self, 'vehicle_impact_zone_count', 0)
            ) + 1
            queue_collision = getattr(
                self.agent,
                'queue_swept_vehicle_collision',
                None,
            )
            if callable(queue_collision):
                queue_collision()
            else:
                self.agent.failed = True
                self.agent.success = False
                self.agent.failure_reason = 'vehicle_collision'
                self.agent.vehicle_collision_count = int(
                    getattr(self.agent, 'vehicle_collision_count', 0)
                ) + 1
                self.agent.collision_count = int(
                    getattr(self.agent, 'collision_count', 0)
                ) + 1
            self._retire_conflict_vehicle(active, 'collision')
            return False

        if target_progress >= 0.0:
            record['travelled_cm'] = round(travelled, 2)
            record['target_progress_cm'] = round(target_progress, 2)
            record['passed_target_position'] = True
            record['final_position'] = {
                'x': vehicle.position.x,
                'y': vehicle.position.y,
            }
            record['completed_sim_time_s'] = round(
                float(getattr(self.agent, 'sim_time_elapsed', 0.0)), 3
            )
            self._retire_conflict_vehicle(active, 'passed_launch_target')
            record['status'] = 'completed'
            record['completion_reason'] = 'passed_launch_target'
            record['despawned'] = True
            return False

        if (
            elapsed >= 120.0
            or active['stalled_update_count'] >= 120
            or travelled >= 2200.0
        ):
            record['passed_target_position'] = False
            self._retire_conflict_vehicle(active, 'stalled_or_distance_limit')
            record['status'] = 'completed'
            record['completion_reason'] = 'stalled_or_distance_limit'
            record['despawned'] = True
            return False

        vehicle.state = VehicleState.MOVING
        vehicle.set_attributes(1.0, 0.0, 0.0)
        self.signal_traffic_communicator.update_vehicle(
            vehicle.id, 1.0, 0.0, 0.0
        )
        return True

    def _update_red_light_conflict_vehicle(self):
        """Keep all launched vehicles moving until impact or target passage."""
        active_conflicts = list(
            getattr(self, '_active_vehicle_conflicts', []) or []
        )
        legacy_active = getattr(self, '_active_red_light_conflict', None)
        if legacy_active and not active_conflicts:
            active_conflicts = [legacy_active]
        if not active_conflicts:
            self._active_red_light_conflict = None
            return

        remaining = [
            active
            for active in active_conflicts
            if self._update_one_conflict_vehicle(active)
        ]
        self._active_vehicle_conflicts = remaining
        self._active_red_light_conflict = remaining[-1] if remaining else None
    
    def cleanup(self):
        """Clean up agents and pedestrians from the scene"""
        try:
            # Clear all spawned entities (agents, pedestrians, traffic signals, etc.)
            # self.communicator.unrealcv.set_mode('async')
            self.communicator.clear_agents()
            self.logger.info("Cleaned up agents, pedestrians, irregular NPCs, traffic signals and intersections")
            
            # Reset pedestrian and irregular pedestrian lists
            self.pedestrians = []
            self.irregular_pedestrians = []

            time.sleep(5)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.logger.warning(f"Failed to cleanup scene: {e}")

    def _get_model_configs(self, agent_path: str):
        """Load model configs from agent_path.
        
        agent_path is a JSON file. Expected format:
        - {"llm": [{"model": "...", "provider": "...", ...}, ...]}  # list of model configs
        - {"llm": {"model": "...", ...}}  # single config, wrapped as [config]
        
        Returns:
            List of config dicts for _create_llm_from_config
        """
        with open(agent_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        llm = data.get('llm', data)
        if isinstance(llm, list):
            return llm
        return [llm]

    def _create_llm_from_config(self, config: dict):
        """Create RTLLM from a config dict (supports {'llm': {...}} or direct llm fields)."""
        llm_config = config.get('llm', config)
        llm_model = llm_config.get('model', 'gpt-4o')
        llm_provider = llm_config.get('provider', 'openai')
        llm_url = llm_config.get('url', '')
        llm_reasoning = llm_config.get('reasoning', False)
        llm_reasoning_effort = llm_config.get('reasoning_effort')
        llm_max_tokens = llm_config.get('max_tokens')
        llm_extra_body = llm_config.get('extra_body')
        llm_prompt_suffix = llm_config.get('prompt_suffix')
        llm_api_mode = llm_config.get('api_mode', 'chat_completions')
        llm_image_detail = llm_config.get('image_detail')
        llm_text_verbosity = llm_config.get('text_verbosity')
        llm_service_tier = llm_config.get('service_tier')
        llm_store = llm_config.get('store', False)
        llm_request_timeout = llm_config.get('request_timeout')
        llm_raise_on_api_error = llm_config.get('raise_on_api_error', False)
        llm_temperature = llm_config.get('temperature', 0.7)
        llm_top_p = llm_config.get('top_p', 1.0)
        llm_seed = llm_config.get('seed')
        common_kwargs = {
            'provider': llm_provider,
            'reasoning': llm_reasoning,
            'reasoning_effort': llm_reasoning_effort,
            'max_tokens': llm_max_tokens,
            'extra_body': llm_extra_body,
            'prompt_suffix': llm_prompt_suffix,
            'api_mode': llm_api_mode,
            'image_detail': llm_image_detail,
            'text_verbosity': llm_text_verbosity,
            'service_tier': llm_service_tier,
            'store': llm_store,
            'request_timeout': llm_request_timeout,
            'raise_on_api_error': llm_raise_on_api_error,
            'temperature': llm_temperature,
            'top_p': llm_top_p,
            'seed': llm_seed,
        }
        if llm_url:
            return RTLLM(
                llm_model,
                url=llm_url,
                **common_kwargs,
            )
        return RTLLM(
            llm_model,
            **common_kwargs,
        )

    def run_multiple_tasks(self, repeats_per_task: int = 3):
        """Run all combinations: tasks x difficulties x models x repeats_per_task.
        
        Iterates over all tasks in task_file, all 3 difficulty levels, and all models
        in agent_path (JSON with "llm" list). Each task is repeated repeats_per_task
        times and results are saved per task/difficulty/model/repeat.
        """
        if not self.all_scenarios:
            self.logger.warning("No scenarios loaded. Please provide a valid task file.")
            return []
        if repeats_per_task < 1:
            raise ValueError("repeats_per_task must be >= 1")
        
        difficulty_levels = ["default", "medium", "easy"]
        total = len(self.all_scenarios) * len(difficulty_levels) * len(self.model_configs) * repeats_per_task
        self.logger.info(
            f"Starting batch: {len(self.all_scenarios)} tasks x {len(difficulty_levels)} difficulties x "
            f"{len(self.model_configs)} models x {repeats_per_task} repeats = {total} runs"
        )
        
        results = []
        for model_idx, model_config in enumerate(self.model_configs):
            self.config = model_config
            self.llm = self._create_llm_from_config(model_config)
            llm_cfg = model_config.get('llm', model_config)
            model_name = llm_cfg.get('model', f'model_{model_idx}')
            self.logger.info(f"Model {model_idx + 1}/{len(self.model_configs)}: {model_name}")
            
            # for task_idx in range(len(self.all_scenarios)):
            for task_idx in [0, 1, 3]:
                for difficulty in difficulty_levels:
                # for difficulty in ["medium"]:
                    for repeat_idx in range(repeats_per_task):
                        repeat_id = repeat_idx + 1
                        self._batch_suffix = f"d{difficulty}_{model_name.replace('/', '_')}_r{repeat_id}"
                        self.logger.info(f"\n{'='*60}")
                        self.logger.info(
                            f"Task {task_idx + 1}/{len(self.all_scenarios)} | Difficulty {difficulty} | "
                            f"Model: {model_name} | Repeat {repeat_id}/{repeats_per_task}"
                        )
                        self.logger.info(f"{'='*60}")
                        try:
                            self.run_single_task(task_idx, difficulty=difficulty)
                            result = {
                                'task_id': self.current_task_id,
                                'task_index': task_idx,
                                'difficulty': difficulty,
                                'model': model_name,
                                'repeat': repeat_id,
                                'success': self.agent.success,
                                'steps': self.agent.step_num,
                                'decision_count': self.agent.decision_count
                            }
                            results.append(result)
                            self.logger.info(f"Completed: {'SUCCESS' if self.agent.success else 'FAILED'}")
                        except Exception as e:
                            self.logger.error(f"Error: {e}")
                            self.logger.error(traceback.format_exc())
                            results.append({
                                'task_id': self.current_task_id,
                                'task_index': task_idx,
                                'difficulty': difficulty,
                                'model': model_name,
                                'repeat': repeat_id,
                                'success': False,
                                'error': str(e)
                            })
                        self.logger.info("Cleaning up...")
                        self.cleanup()
                        time.sleep(20)
        
        self._batch_suffix = None
        success_count = sum(1 for r in results if r.get('success', False))
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Batch Summary: {success_count}/{len(results)} successful ({success_count/len(results)*100:.1f}%)")
        self.logger.info(f"{'='*60}")
        return results

    def _load_background_agents_from_scenario(self, scenario_data, num_agents: int):
        """
        Load pedestrians from pre-generated scenario data with difficulty-based variations
        
        Args:
            scenario_data: Dictionary containing pedestrian route data
            num_agents: Number of pedestrian agents to create
            
        Returns:
            Tuple of (pedestrians_list, irregular_pedestrians_list) based on difficulty
            
        Difficulty levels:
            - level0: traffic lights/crosswalk only
            - level1: uniform-speed pedestrians
            - level2: variable-speed pedestrians plus moving objects
            - level3: level2 plus irregular crossings
            - level4: level3 plus falling hazards
            - default/medium/easy: legacy ratio-based dynamic activation
        """
        pedestrians = []
        irregular_pedestrians = []
        
        pedestrian_routes = scenario_data.get('pedestrian_routes', [])
        
        if not pedestrian_routes:
            return pedestrians, irregular_pedestrians
        
        # Get agent start position for distance checking
        task_data = scenario_data.get('task', {})
        agent_start_point = task_data.get('start_point', None)
        agent_start_pos = None
        if agent_start_point:
            agent_start_pos = Vector(agent_start_point[0], agent_start_point[1])
        else:
            raise ValueError("Agent start position not found in scenario data")
        
        speed_options = self._get_pedestrian_speed_options()
        
        # Initialize falling object IDs list (populated for highest difficulty)
        if not hasattr(self, 'falling_object_ids'):
            self.falling_object_ids = []
        
        pedestrian_ratio = self._get_pedestrian_activation_ratio()
        num_pedestrians = int(round(num_agents * pedestrian_ratio))
        self._requested_scripted_pedestrian_count = num_pedestrians
        
        # Create pedestrians
        for i in range(num_pedestrians):
            # Equal probability selection for all routes
            route_data = random.choice(pedestrian_routes)
            route_points = route_data['route']
            if len(route_points) < 2:
                continue
                
            def lane_separated_route(selected_route):
                selected_points = [
                    Vector(point[0], point[1])
                    for point in selected_route['route']
                ]
                if random.random() < 0.5:
                    selected_points = list(reversed(selected_points))
                lane_offset = (
                    SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                        'right_hand_lane_offset_cm'
                    ]
                    + random.uniform(
                        -SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                            'lane_offset_jitter_cm'
                        ],
                        SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                            'lane_offset_jitter_cm'
                        ],
                    )
                )
                return build_scripted_pedestrian_patrol_loop(
                    selected_points,
                    lane_offset,
                    is_loop=selected_route.get('is_loop', False),
                )

            route_nodes, placement_indices = lane_separated_route(route_data)
            
            # Build occupied points from existing pedestrians
            occupied_positions = [p.position for p in pedestrians]

            # Try to find a safe position by sampling uniformly along segments, switching segment/route if needed
            max_attempts_per_segment = 30
            position = None
            chosen_idx = None

            # Helper to attempt placement on the given polyline segments
            def try_place_on_route(nodes, allowed_indices):
                indices = list(allowed_indices)
                random.shuffle(indices)
                for idx in indices:
                    s = nodes[idx]
                    e = nodes[idx + 1]
                    for _ in range(max_attempts_per_segment):
                        t = random.random()
                        candidate = Vector(
                            s.x + (e.x - s.x) * t,
                            s.y + (e.y - s.y) * t
                        )
                        # Check distance to endpoints (at least 500), to other agents (at least 200),
                        # and to agent start position (at least 200)
                        safe_from_endpoints = candidate.distance(s) >= 500 and candidate.distance(e) >= 500
                        safe_from_others = all(
                            candidate.distance(op) >= (
                                SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                    'minimum_spawn_separation_cm'
                                ]
                            )
                            for op in occupied_positions
                        )
                        safe_from_agent = True
                        if agent_start_pos is not None:
                            safe_from_agent = candidate.distance(
                                agent_start_pos
                            ) >= SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                'minimum_spawn_separation_cm'
                            ]
                        safe_from_obstacles = all(
                            candidate.distance(obstacle_position) >= (
                                SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                                    'minimum_static_obstacle_separation_cm'
                                ]
                            )
                            for obstacle_position in getattr(
                                self,
                                'static_obstacle_positions',
                                [],
                            )
                        )
                        
                        if (
                            safe_from_endpoints
                            and safe_from_others
                            and safe_from_agent
                            and safe_from_obstacles
                        ):
                            return candidate, idx
                return None, None

            # First try the currently selected route
            position, chosen_idx = try_place_on_route(
                route_nodes,
                placement_indices,
            )

            # If not found, try other pedestrian routes from scenario
            if position is None:
                other_routes = list(pedestrian_routes)
                if route_data in other_routes:
                    other_routes.remove(route_data)
                random.shuffle(other_routes)
                for alt_route in other_routes:
                    if len(alt_route['route']) < 2:
                        continue
                    alt_nodes, alt_placement_indices = lane_separated_route(
                        alt_route
                    )
                    position, chosen_idx = try_place_on_route(
                        alt_nodes,
                        alt_placement_indices,
                    )
                    if position is not None:
                        route_nodes = alt_nodes
                        placement_indices = alt_placement_indices
                        break

            # Never force an actor onto an occupied midpoint. A smaller safely
            # placed population preserves physical collision semantics; an
            # overlapping fallback creates a deterministic crowd deadlock.
            if position is None:
                self.logger.warning(
                    'Skipping scripted pedestrian %s/%s: no route position '
                    'satisfies %.1fcm spawn separation',
                    i + 1,
                    num_pedestrians,
                    SCRIPTED_PEDESTRIAN_MOTION_POLICY[
                        'minimum_spawn_separation_cm'
                    ],
                )
                continue

            # Use the chosen segment for direction and downstream waypoint construction
            start_idx = chosen_idx
            start_pos = route_nodes[start_idx]
            end_pos = route_nodes[start_idx + 1]
            
            # Calculate direction from first to second point
            direction = Vector(
                end_pos.x - start_pos.x,
                end_pos.y - start_pos.y
            ).normalize()
            
            # Every regular route is a cycle. Authored non-loop routes use a
            # separately offset return lane instead of reversing onto the
            # outbound lane and creating deterministic head-on deadlocks.
            waypoints = (
                route_nodes[start_idx + 1:]
                + route_nodes[:start_idx + 1]
            )
            
            # Interpolate waypoints to make trajectory denser
            # Insert intermediate points if distance between consecutive points > 2500
            interpolated_waypoints = []
            for j in range(len(waypoints)):
                interpolated_waypoints.append(waypoints[j])
                if j < len(waypoints) - 1:
                    p1 = waypoints[j]
                    p2 = waypoints[j + 1]
                    # Calculate distance between consecutive points
                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    # If distance > 2500, insert intermediate point(s)
                    if distance > 2500:
                        num_segments = int(distance / 2500)
                        for k in range(1, num_segments + 1):
                            t = k / (num_segments + 1)
                            interp_x = p1.x + t * dx
                            interp_y = p1.y + t * dy
                            interpolated_waypoints.append(Vector(interp_x, interp_y))
            waypoints = interpolated_waypoints
            
            # Create pedestrian
            agent = RTPedestrian(position, direction, waypoints)
            ped_speed = random.choice(speed_options) if speed_options else 100
            agent.set_speed(ped_speed)
            pedestrians.append(agent)
        
        movable_ratio = self._get_movable_obstacle_activation_ratio()
        self.logger.info(
            f"[Difficulty {self.difficulty}] Moving objects will be activated during spawn "
            f"with ratio={movable_ratio:.0%} ({len(self.movable_obstacle_ids)} movable obstacles found)"
        )
        
        # Irregular NPCs (robot_dog/child-style crossings) are created from a sampled subset of irregular routes.
        irregular_routes = scenario_data.get('irregular_routes', [])
        irregular_ratio = self._get_irregular_activation_ratio()
        if irregular_routes:
            target_irregular_count = int(round(len(irregular_routes) * irregular_ratio))
            target_irregular_count = max(0, min(len(irregular_routes), target_irregular_count))
            if self.irregular_npc_max is not None:
                target_irregular_count = min(
                    target_irregular_count,
                    self.irregular_npc_max,
                )
            sampled_irregular_routes = random.sample(irregular_routes, target_irregular_count) if target_irregular_count > 0 else []
            irregular_pedestrians = self._create_irregular_npcs(sampled_irregular_routes, agent_start_pos, pedestrians, self.static_obstacle_positions)
            self.logger.info(
                f"[Difficulty {self.difficulty}] Created {len(irregular_pedestrians)}/{len(irregular_routes)} irregular NPCs "
                f"(robot_dog) with ratio={irregular_ratio:.0%}"
            )
        else:
            self.logger.warning(f"[Difficulty {self.difficulty}] No irregular_routes found in scenario data")
        
        # Falling objects are loaded for all difficulties and sampled during spawn by configured ratio.
        self.falling_object_ids = scenario_data.get('falling_objects', [])
        falling_ratio = self._get_falling_object_activation_ratio()
        if self.falling_object_ids:
            self.logger.info(
                f"[Difficulty {self.difficulty}] Found {len(self.falling_object_ids)} falling objects; "
                f"activation ratio={falling_ratio:.0%}"
            )
        else:
            self.logger.warning(f"[Difficulty {self.difficulty}] No falling_objects found in scenario data")
        
        # Log all background agents information
        self.logger.info(f"=== Task Background Agents Summary ===")
        self.logger.info(f"Difficulty Level: {self.difficulty}")
        self.logger.info(f"Total Pedestrians: {len(pedestrians)}")
        self.logger.info(f"Total Irregular NPCs (robot_dog): {len(irregular_pedestrians)}")
        
        # Log pedestrian details
        if pedestrians:
            self.logger.info(f"\n--- Pedestrian Details ---")
            for i, ped in enumerate(pedestrians):
                self.logger.info(f"Pedestrian {i+1}:")
                self.logger.info(f"  Initial Position: ({ped.position.x:.2f}, {ped.position.y:.2f})")
                self.logger.info(f"  Direction: ({ped.direction.x:.2f}, {ped.direction.y:.2f})")
                self.logger.info(f"  Speed: {ped.speed}")
                self.logger.info(f"  Waypoints ({len(ped.waypoints)}):")
                for j, wp in enumerate(ped.waypoints[:5]):  # Show first 5 waypoints
                    self.logger.info(f"    [{j+1}] ({wp.x:.2f}, {wp.y:.2f})")
                if len(ped.waypoints) > 5:
                    self.logger.info(f"    ... and {len(ped.waypoints) - 5} more waypoints")
        
        # Log irregular NPC details
        if irregular_pedestrians:
            self.logger.info(f"\n--- Irregular NPC (robot_dog) Details ---")
            for i, ped in enumerate(irregular_pedestrians):
                self.logger.info(f"Irregular NPC {i+1}:")
                self.logger.info(f"  Initial Position: ({ped.position.x:.2f}, {ped.position.y:.2f})")
                self.logger.info(f"  Direction: ({ped.direction.x:.2f}, {ped.direction.y:.2f})")
                self.logger.info(f"  Speed: {ped.speed}")
                self.logger.info(f"  Waypoints ({len(ped.waypoints)}):")
                for j, wp in enumerate(ped.waypoints):
                    self.logger.info(f"    [{j+1}] ({wp.x:.2f}, {wp.y:.2f})")
        
        self.logger.info(f"=== End of Background Agents Summary ===\n")
        
        return pedestrians, irregular_pedestrians

    def _create_irregular_npcs(self, irregular_routes, agent_start_pos, existing_pedestrians, static_obstacle_positions=None):
        """
        Create irregular NPCs (robot_dog) that follow non-standard routes (e.g. jaywalking, crossing sidewalks diagonally).
        
        Each irregular route defines a short path. An NPC is spawned at a valid route node and 
        walks back and forth along it repeatedly. These are spawned as robot_dog models (type=0).
        
        Position selection: tries each node in the route sequentially (node 0, 1, 2, ...),
        picks the first one that passes all collision checks:
            - >= 200 distance from existing pedestrians and other irregular NPCs
            - >= 200 distance from agent start position
            - >= 100 distance from all static obstacles
        
        Args:
            irregular_routes: List of irregular route dicts, each with 'route_id', 'sidewalk_id', and 'route' (list of [x,y])
            agent_start_pos: Agent's start position (Vector) for collision avoidance
            existing_pedestrians: List of already-created pedestrians for collision avoidance
            static_obstacle_positions: List of Vector positions of static obstacles for collision avoidance
            
        Returns:
            List of RTPedestrian objects following irregular routes
        """
        irregular_pedestrians = []
        if static_obstacle_positions is None:
            static_obstacle_positions = []
        
        for route_data in irregular_routes:
            route_points = route_data.get('route', [])
            if len(route_points) < 2:
                continue
            
            # Convert route to Vector objects
            route_nodes = [Vector(point[0], point[1]) for point in route_points]
            
            # Try each node sequentially (0, 1, 2, ...) as starting position
            occupied_positions = [p.position for p in existing_pedestrians] + \
                                 [p.position for p in irregular_pedestrians]
            
            chosen_idx = None
            for idx, candidate in enumerate(route_nodes):
                safe_from_others = all(candidate.distance(op) >= 200 for op in occupied_positions)
                safe_from_agent = agent_start_pos is None or candidate.distance(agent_start_pos) >= 200
                safe_from_obstacles = all(candidate.distance(obs) >= 100 for obs in static_obstacle_positions)
                
                if safe_from_others and safe_from_agent and safe_from_obstacles:
                    chosen_idx = idx
                    break
            
            if chosen_idx is None:
                self.logger.warning(f"Skipping irregular route {route_data.get('route_id', '?')} - no valid position at any node")
                continue
            
            position = route_nodes[chosen_idx]
            
            # Build back-and-forth waypoints starting from chosen_idx:
            #   Phase 1: forward to end         (chosen_idx+1 -> last)
            #   Phase 2: reverse to start        (last-1 -> first)
            #   Phase 3: forward back to start   (first+1 -> chosen_idx)
            part1 = route_nodes[chosen_idx + 1:]                          # forward to end
            part2 = list(reversed(route_nodes[:-1]))                      # reverse to start (skip last, already there)
            part3 = route_nodes[1:chosen_idx + 1] if chosen_idx > 0 else []  # forward back to start pos (skip first)
            waypoints = part1 + part2 + part3
            
            # Determine initial direction toward the first waypoint
            first_wp = waypoints[0]
            direction = Vector(
                first_wp.x - position.x,
                first_wp.y - position.y
            ).normalize()
            
            # Create irregular pedestrian with fixed speed
            agent = RTPedestrian(position, direction, waypoints)
            agent.set_speed(100)
            irregular_pedestrians.append(agent)
        
        return irregular_pedestrians

    def _convert_tasks_to_scenarios(self, tasks):
        """Convert all_tasks.json format to scenario format
        
        Args:
            tasks: List of task objects from all_tasks.json
            
        Returns:
            List of scenario objects compatible with internal format
        """
        scenarios = []
        map_config = Config(CONFIG_PATH)
        rendered_crosswalk_offset_cm = map_config[
            'traffic.crosswalk_offset'
        ]
        rendered_geometry_cache = {}
        for source_task in tasks:
            map_path = source_task.get('map_path', 'data/roads.json')
            if map_path not in rendered_geometry_cache:
                with open(map_path, 'r', encoding='utf-8') as roads_handle:
                    roads_payload = json.load(roads_handle)
                centers = set()
                for road in roads_payload.get('roads', []):
                    for endpoint_name in ('start', 'end'):
                        endpoint = road[endpoint_name]
                        centers.add((
                            float(endpoint['x']) * 100.0,
                            float(endpoint['y']) * 100.0,
                        ))
                rendered_geometry_cache[map_path] = {
                    'intersection_centers': [
                        list(center) for center in sorted(centers)
                    ],
                    'crosswalk_segments': rendered_native_crosswalk_segments(
                        roads_payload,
                        rendered_crosswalk_offset_cm,
                        map_config['traffic.sidewalk_offset'],
                    ),
                }
            rendered_geometry = rendered_geometry_cache[map_path]
            task = align_task_crosswalks_to_rendered_geometry(
                source_task,
                rendered_crosswalk_offset_cm,
                intersection_centers=(
                    rendered_geometry['intersection_centers']
                ),
                rendered_crosswalk_segments=(
                    rendered_geometry['crosswalk_segments']
                ),
            )
            route_info = dict(task.get('route_info', {}))
            shortest_path = reconstruct_route_points_from_task_edges(task)
            route_info['shortest_path'] = shortest_path
            if len(shortest_path) >= 2:
                total_distance = 0
                for i in range(len(shortest_path) - 1):
                    p1 = shortest_path[i]
                    p2 = shortest_path[i + 1]
                    distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                    total_distance += distance
                required_time = total_distance / 100 * 1.2  # 100 is the minimum speed

                # Count crosswalks from edges and add 60s for each
                edges = task.get('edges', [])
                crosswalk_count = sum(1 for edge in edges if edge.get('type') == 'crosswalk')
                required_time += crosswalk_count * 60
            
            scenario = {
                'task_id': task.get('task_id'),
                'task': {
                    'start_point': task['start_point'],
                    'end_point': task['end_point'],
                    'start_edge': task.get('start_edge'),
                    'end_edge': task.get('end_edge'),
                    'edges': task.get('edges', []),
                    'total_hops': task.get('total_hops', 0),
                    'crosswalk_hops': task.get('crosswalk_hops', 0),
                    'required_time': required_time
                },
                'route_info': route_info,
                'pedestrian_routes': task.get('pedestrian_routes', []),
                'irregular_routes': task.get('irregular_routes', []),
                'falling_objects': task.get('falling_objects', []),
                'render_geometry_calibration': task.get(
                    'render_geometry_calibration', {}
                ),
                'map_path': map_path,
            }
            scenarios.append(scenario)
        return scenarios
    
    def _initialize_world(self):
        """Initialize the world with scenario data"""
        # Load map configuration
        map_config = Config(CONFIG_PATH)
        map_config.config.setdefault('traffic', {}).setdefault(
            'pedestrian', {}
        )['signal_compliance_probability'] = (
            self.pedestrian_signal_compliance_probability
        )
        self.map = Map(map_config, None)
        
        # Use map_path from scenario data if available
        map_path = self.scenario_data.get('map_path', None)
        if map_path is None:
            raise ValueError("Map path is not provided")
        self.map.initialize_map_from_file(
            map_path,
            map_config['traffic.sidewalk_offset'],
            False,
            crosswalk_offset=map_config['traffic.crosswalk_offset'],
        )
        def signal_traffic_count(env_name, config_key):
            raw_value = os.environ.get(
                env_name,
                map_config.get(config_key, 0),
            )
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'{env_name} must be a non-negative integer, got '
                    f'{raw_value!r}'
                ) from exc
            if value < 0:
                raise ValueError(
                    f'{env_name} must be a non-negative integer, got {value}'
                )
            return value

        self.signal_traffic_vehicle_count = signal_traffic_count(
            'SIMWORLD_SIGNAL_TRAFFIC_VEHICLES',
            'traffic.rt_signal_controlled_vehicles',
        )
        self.signal_traffic_pedestrian_count = signal_traffic_count(
            'SIMWORLD_SIGNAL_TRAFFIC_PEDESTRIANS',
            'traffic.rt_signal_controlled_pedestrians',
        )
        self.signal_traffic_enabled = bool(
            self.signal_traffic_vehicle_count
            or self.signal_traffic_pedestrian_count
        )
        self.traffic_controller = RTTrafficController(
            map_config,
            map_path,
            seed=self.seed,
            num_vehicles=self.signal_traffic_vehicle_count,
            num_pedestrians=self.signal_traffic_pedestrian_count,
        )
        if self.signal_traffic_enabled:
            self.signal_traffic_communicator = RTSignalTrafficCommunicator(
                self.communicator.unrealcv
            )
            self.traffic_controller.init_communicator(
                self.signal_traffic_communicator
            )
        else:
            self.signal_traffic_communicator = None
        self.logger.info(
            'Signal-following background traffic: vehicles=%s pedestrians=%s',
            self.signal_traffic_vehicle_count,
            self.signal_traffic_pedestrian_count,
        )

        self.traffic_phase_timing = TrafficPhaseTiming.from_config(map_config)
        if bool(
            map_config.get(
                'traffic.traffic_signal.auto_calibrate_pedestrian_clearance',
                True,
            )
        ):
            self.traffic_phase_timing = (
                self.traffic_phase_timing.calibrated_for_crosswalks(
                    self.traffic_controller.crosswalks
                )
            )
        self.traffic_phase_timing = apply_timing_environment_overrides(
            self.traffic_phase_timing
        )
        self.use_intersection_phase_api = bool(
            map_config.get(
                'traffic.traffic_signal.use_intersection_phase_api',
                False,
            )
        )
        self.intersection_controller_model_path = map_config.get(
            'traffic.traffic_signal.intersection_controller_model_path',
            self.intersection_controller_model_path,
        )
        self.traffic_light_model_path = map_config.get(
            'traffic.traffic_signal.traffic_light_model_path',
            self.traffic_light_model_path,
        )
        self.pedestrian_light_model_path = map_config.get(
            'traffic.traffic_signal.pedestrian_light_model_path',
            self.pedestrian_light_model_path,
        )
        self.traffic_light_scale = float(
            map_config.get(
                'traffic.traffic_signal.traffic_light_scale',
                self.traffic_light_scale,
            )
        )
        self.pedestrian_light_scale = float(
            map_config.get(
                'traffic.traffic_signal.pedestrian_light_scale',
                self.pedestrian_light_scale,
            )
        )
        conflict_geometry_overrides = getattr(
            self,
            '_conflict_vehicle_geometry_overrides_m',
            {},
        )
        self.conflict_vehicle_min_launch_distance_cm = 100.0 * float(
            conflict_geometry_overrides.get('minimum')
            if conflict_geometry_overrides.get('minimum') is not None
            else map_config.get(
                'traffic.traffic_signal.conflict_vehicle_min_launch_distance_m',
                self.conflict_vehicle_min_launch_distance_cm / 100.0,
            )
        )
        self.conflict_vehicle_max_launch_distance_cm = 100.0 * float(
            conflict_geometry_overrides.get('maximum')
            if conflict_geometry_overrides.get('maximum') is not None
            else map_config.get(
                'traffic.traffic_signal.conflict_vehicle_max_launch_distance_m',
                self.conflict_vehicle_max_launch_distance_cm / 100.0,
            )
        )
        if (
            self.conflict_vehicle_min_launch_distance_cm < 0.0
            or self.conflict_vehicle_max_launch_distance_cm < 0.0
        ):
            raise ValueError(
                'Conflict-vehicle launch distances must be non-negative'
            )
        if (
            self.conflict_vehicle_min_launch_distance_cm
            > self.conflict_vehicle_max_launch_distance_cm
        ):
            raise ValueError(
                'Conflict-vehicle minimum launch distance cannot exceed maximum'
            )
        self.conflict_vehicle_impact_radius_cm = 100.0 * float(
            conflict_geometry_overrides.get('impact_radius')
            if conflict_geometry_overrides.get('impact_radius') is not None
            else map_config.get(
                'traffic.traffic_signal.conflict_vehicle_impact_radius_m',
                self.conflict_vehicle_impact_radius_cm / 100.0,
            )
        )
        if self.conflict_vehicle_impact_radius_cm <= 0.0:
            raise ValueError(
                'Conflict-vehicle impact radius must be positive'
            )

        traffic_controller_mode = os.environ.get(
            'SIMWORLD_TRAFFIC_CONTROLLER_MODE',
            'grouped_legacy',
        ).strip().lower() or 'grouped_legacy'
        self.traffic_controller_mode = traffic_controller_mode
        self.python_grouped_signal_control = False
        if traffic_controller_mode == 'legacy_blueprint':
            self.use_intersection_phase_api = False
            self.intersection_controller_model_path = (
                '/Game/RealTimeBench/Traffic/'
                'RT_BP_Intersection.RT_BP_Intersection_C'
            )
            self.traffic_light_model_path = (
                '/Game/RealTimeBench/Traffic/'
                'RT_BP_street_light.RT_BP_street_light_C'
            )
            self.pedestrian_light_model_path = (
                '/Game/RealTimeBench/Traffic/'
                'RT_BP_street_light_ped.RT_BP_street_light_ped_C'
            )
        elif traffic_controller_mode == 'grouped_legacy':
            self.use_intersection_phase_api = False
            self.python_grouped_signal_control = True
            self.intersection_controller_model_path = None
            self.traffic_light_model_path = (
                '/Game/RealTimeBench/Traffic/'
                'RT_BP_street_light.RT_BP_street_light_C'
            )
            self.pedestrian_light_model_path = (
                '/Game/RealTimeBench/Traffic/'
                'RT_BP_street_light_ped.RT_BP_street_light_ped_C'
            )
        elif traffic_controller_mode == 'native':
            self.use_intersection_phase_api = True
            self.intersection_controller_model_path = (
                self._native_intersection_controller_path()
            )
        elif traffic_controller_mode:
            raise ValueError(
                'SIMWORLD_TRAFFIC_CONTROLLER_MODE must be '
                "'legacy_blueprint', 'grouped_legacy', or 'native', got "
                f'{traffic_controller_mode!r}'
            )
        self.logger.info(
            'Configured real-world traffic phase timing: %s',
            self.traffic_phase_timing.to_dict(),
        )
        self.logger.info(
            'Traffic intersection controller: mode=%s path=%s',
            self.traffic_controller_mode,
            self.intersection_controller_model_path,
        )
        self._generate_ue_world(map_path)
        
        # Store map_path for loading obstacles later
        self.current_map_path = map_path
        
        # Traffic-light alignment runs can disable background motion while
        # keeping crossing geometry and UE traffic signals intact.
        self.movable_obstacle_ids = (
            [] if self.disable_background_agents
            else self._load_movable_obstacle_ids()
        )
        
        # Keep the raw map list for NPC placement. The filtered list is
        # available to internal collision/traffic geometry only; it is not
        # appended to the ordinary VLM prompt.
        self.map_static_obstacles = self._load_static_obstacles()
        if self.difficulty_config.get(
            "load_static_obstacles_for_internal_geometry",
            True,
        ):
            self.static_obstacles = self.map_static_obstacles
        else:
            self.static_obstacles = []
        self.static_obstacle_positions = [Vector(o['x'], o['y']) for o in self.map_static_obstacles]
        
        # Get task-relevant intersections from route_info
        self.task_intersections = []
        route_info = self.scenario_data.get('route_info', {})
        if 'intersections' in route_info:
            task_intersection_coords = route_info['intersections']
            self.task_intersections = self.traffic_controller.get_intersections_from_task(task_intersection_coords)
            self.logger.info(f"Found {len(self.task_intersections)} task-relevant intersections")
            for intersection in self.task_intersections:
                self.logger.info(f"  - Intersection {intersection.id} at ({intersection.center.x}, {intersection.center.y})")
        else:
            self.logger.warning("No intersection information in task route_info")
        
        # Calculate actual pedestrian number based on task length (total_hops)
        task_data = self.scenario_data.get('task', {})
        total_hops = task_data.get('total_hops', 0)
        assert total_hops > 0, "Total hops must be greater than 0"
        actual_ped_num = self.ped_num_per_edge * total_hops

        
        # Load pedestrians and irregular NPCs from scenario data unless this is
        # an isolated traffic-light alignment benchmark.
        if self.disable_background_agents:
            self.pedestrians = []
            self.irregular_pedestrians = []
            self.falling_object_ids = []
            self.logger.info(
                "Background pedestrians, irregular NPCs, movable obstacles, "
                "and falling objects disabled for traffic-light alignment"
            )
        else:
            self.pedestrians, self.irregular_pedestrians = self._load_background_agents_from_scenario(
                self.scenario_data,
                actual_ped_num,
            )
        num_ped_routes = len(self.scenario_data.get('pedestrian_routes', []))
        num_irregular_routes = len(self.scenario_data.get('irregular_routes', []))
        self.logger.info(
            f"Task has {total_hops} edges, using {actual_ped_num} background "
            f"pedestrian candidates ({self.ped_num_per_edge} per edge); "
            f"irregular NPC max={self.irregular_npc_max}"
        )
        self.logger.info(f"Loaded {len(self.pedestrians)} pedestrians, {len(self.irregular_pedestrians)} irregular NPCs (robot_dog)")
        self.logger.info(f"Available routes: {num_ped_routes} pedestrian, {num_irregular_routes} irregular")
        self._prepare_dynamic_obstacle_selection()

    def _generate_ue_world(self, map_path: str):
        """Load static map geometry into UE before spawning realtime actors."""
        map_dir = os.path.dirname(map_path)
        world_json = map_path
        if os.path.basename(map_path) != 'progen_world.json':
            candidate_world_json = os.path.join(map_dir, 'progen_world.json')
            if os.path.exists(candidate_world_json):
                world_json = candidate_world_json

        if not os.path.exists(world_json):
            self.logger.warning(f"World JSON not found, static UE map will not be generated: {world_json}")
            return
        if not os.path.exists(UE_ASSET_PATH):
            self.logger.warning(f"UE asset library not found, static UE map will not be generated: {UE_ASSET_PATH}")
            return
        if not hasattr(self.communicator, 'generate_world'):
            self.logger.warning("Communicator does not support generate_world; static UE map will not be generated")
            return

        try:
            # A previous rollout may have crashed before ``cleanup``.  Remove
            # all runtime actors before rebuilding the map so signal vehicles
            # and pedestrians cannot accumulate across runs.
            if hasattr(self.communicator, 'clear_agents'):
                self.communicator.clear_agents()
            if hasattr(self.communicator, 'clear_env'):
                self.communicator.clear_env(keep_roads=False)
            generated_ids = self.communicator.generate_world(world_json, UE_ASSET_PATH, run_time=True)
            self.logger.info(f"Generated {len(generated_ids)} static UE map objects from {world_json}")
        except Exception as exc:
            self.logger.error(f"Failed to generate static UE map from {world_json}: {exc}")
            self.logger.error(traceback.format_exc())
            return

        # Parked vehicles embedded in procedural road blueprints are scenery,
        # not traffic participants.  Freeze them before simulation can resume.
        # This call intentionally sits outside the generation catch so a
        # freeze failure aborts the rollout instead of producing invalid video.
        self._freeze_generated_decorative_vehicles('post_world_generation')

    def _freeze_generated_decorative_vehicles(self, stage: str):
        """Freeze road-blueprint parked cars and record auditable metadata."""
        if not hasattr(
            self.communicator,
            'freeze_generated_decorative_vehicles',
        ):
            raise RuntimeError(
                'Communicator cannot freeze generated decorative vehicles'
            )

        frozen_ids = (
            self.communicator.freeze_generated_decorative_vehicles()
        )
        self.frozen_decorative_vehicle_ids = sorted(
            set(self.frozen_decorative_vehicle_ids).union(frozen_ids)
        )
        freeze_record = {
            'stage': str(stage),
            'matched_count': len(frozen_ids),
        }
        self.decorative_vehicle_freeze_passes.append(freeze_record)
        self.logger.info(
            "Froze %d generated decorative vehicles at %s",
            len(frozen_ids),
            stage,
        )
        return frozen_ids

    @staticmethod
    def _route_crosswalk_marking_specs(
        crosswalk,
        stripe_count=ROUTE_CROSSWALK_MARKING_STRIPE_COUNT,
        stripe_width_cm=ROUTE_CROSSWALK_MARKING_STRIPE_WIDTH_CM,
        stripe_length_cm=ROUTE_CROSSWALK_MARKING_LENGTH_CM,
        stripe_height_cm=ROUTE_CROSSWALK_MARKING_HEIGHT_CM,
        base_size_cm=ROUTE_CROSSWALK_MARKING_BASE_SIZE_CM,
    ):
        """Return render-only zebra-stripe transforms for one logical crossing.

        Stripes are distributed along the existing crosswalk centerline and
        extend perpendicular to pedestrian travel.  The graph geometry remains
        the single source of truth; this method merely makes that geometry
        visible when the packaged road asset omitted paint on this arm.
        """
        start = crosswalk.start
        end = crosswalk.end
        delta = end - start
        length = start.distance(end)
        if length <= 1e-6 or stripe_count <= 0 or base_size_cm <= 0:
            return []

        axis = delta.normalize()
        yaw_degrees = math.degrees(math.atan2(axis.y, axis.x))
        along_scale = stripe_width_cm / base_size_cm
        across_scale = stripe_length_cm / base_size_cm
        height_scale = stripe_height_cm / base_size_cm
        specs = []
        for index in range(stripe_count):
            fraction = (index + 1.0) / (stripe_count + 1.0)
            center = start + delta * fraction
            specs.append({
                'stripe_index': index,
                'fraction': fraction,
                'location': (center.x, center.y, stripe_height_cm / 2.0),
                'orientation': (0.0, yaw_degrees, 0.0),
                # RT_Box's local X follows the crosswalk and local Y spans it.
                'scale': (along_scale, across_scale, height_scale),
            })
        return specs

    def _spawn_route_crosswalk_markings(self, route_crosswalks):
        """Make every route crosswalk visually match its logical corridor.

        This is deliberately independent of ``traffic_policy``.  A
        ``visual_only`` agent needs the physical paint, while a
        ``safety_assisted`` run must see the identical scene.  Markings are
        non-colliding, immovable, and omitted when the explicit environment
        opt-out is set for package compatibility checks.
        """
        enabled = os.environ.get(
            'SIMWORLD_RENDER_ROUTE_CROSSWALKS',
            '0',
        ).strip().lower() not in {'0', 'false', 'no', 'off'}
        self.route_crosswalk_marking_ids = []
        self.route_crosswalk_marking_records = []
        if not enabled or not route_crosswalks:
            return

        spawn_ue_actors = os.environ.get(
            'SIMWORLD_RENDER_ROUTE_CROSSWALK_UE_ACTORS',
            '1',
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        unrealcv = self.communicator.unrealcv
        for crosswalk in route_crosswalks:
            crosswalk_id = str(getattr(crosswalk, 'id', 'unknown'))
            for spec in self._route_crosswalk_marking_specs(crosswalk):
                record = dict(spec)
                record.update({
                    'actor_name': None,
                    'crosswalk_id': crosswalk_id,
                    'asset_path': None,
                    'collision': False,
                    'render_backend': (
                        'ue_actor_static'
                        if spawn_ue_actors
                        else 'perspective_rgb_overlay'
                    ),
                })
                self.route_crosswalk_marking_records.append(record)
                if not spawn_ue_actors:
                    continue
                name = (
                    'RT_ROUTE_CROSSWALK_'
                    f'{crosswalk_id}_{spec["stripe_index"]}'
                )
                try:
                    unrealcv.spawn_bp_asset(
                        ROUTE_CROSSWALK_MARKING_ASSET,
                        name,
                    )
                    unrealcv.set_location(spec['location'], name)
                    unrealcv.set_orientation(spec['orientation'], name)
                    unrealcv.set_scale(spec['scale'], name)
                    unrealcv.set_color(name, [245, 245, 235])
                    unrealcv.set_collision(name, False)
                    unrealcv.set_movable(name, False)
                except Exception as exc:
                    raise RuntimeError(
                        'Failed to render persistent route crosswalk '
                        f'{crosswalk_id} stripe {spec["stripe_index"]}: {exc}'
                    ) from exc

                record.update({
                    'actor_name': name,
                    'crosswalk_id': crosswalk_id,
                    'asset_path': ROUTE_CROSSWALK_MARKING_ASSET,
                    'collision': False,
                    'render_backend': 'ue_actor_static',
                })
                self.route_crosswalk_marking_ids.append(name)

        self.logger.info(
            'Prepared %d route zebra stripes for %d route '
            'crosswalks (persistent UE actors=%d)',
            len(self.route_crosswalk_marking_records),
            len(route_crosswalks),
            len(self.route_crosswalk_marking_ids),
        )

    def _normalize_difficulty_label(self, difficulty):
        """Normalize difficulty input to one of the level labels or legacy labels.

        Backward compatibility:
            1 -> default
            2 -> medium
            3 -> easy
        """
        if isinstance(difficulty, int):
            legacy_mapping = {1: "default", 2: "medium", 3: "easy"}
            if difficulty in legacy_mapping:
                return legacy_mapping[difficulty]
            if 0 <= difficulty <= 4:
                return f"level{difficulty}"
            raise ValueError(f"Invalid difficulty: {difficulty}. Supported integer difficulties are 0, 1, 2, 3, 4.")

        if isinstance(difficulty, str):
            difficulty_label = difficulty.strip().lower()
            difficulty_label = LEVEL_DIFFICULTY_ALIASES.get(difficulty_label, difficulty_label)
            if difficulty_label in LEVEL_DIFFICULTY_CONFIGS:
                return difficulty_label
            if difficulty_label in DIFFICULTY_DYNAMIC_ACTIVATION_RATIO:
                return difficulty_label
            raise ValueError(
                f"Invalid difficulty: {difficulty}. Supported string difficulties are "
                f"{', '.join(list(LEVEL_DIFFICULTY_CONFIGS.keys()) + list(DIFFICULTY_DYNAMIC_ACTIVATION_RATIO.keys()))}."
            )

        raise ValueError(f"Invalid difficulty type: {type(difficulty)}. Use str or int.")

    def _get_difficulty_config(self) -> dict:
        """Return the active difficulty/level configuration for result metadata."""
        if self.difficulty in LEVEL_DIFFICULTY_CONFIGS:
            return dict(LEVEL_DIFFICULTY_CONFIGS[self.difficulty])
        ratio = DIFFICULTY_DYNAMIC_ACTIVATION_RATIO[self.difficulty]
        return {
            "label": self.difficulty,
            "description": "Legacy ratio-based difficulty.",
            "pedestrian_ratio": ratio,
            "pedestrian_speeds": [100],
            "movable_obstacle_ratio": ratio,
            "irregular_ratio": ratio,
            "falling_ratio": ratio,
            "load_static_obstacles_for_internal_geometry": True,
        }

    def _get_dynamic_activation_ratio(self) -> float:
        """Return dynamic object activation ratio for current difficulty."""
        if self.difficulty in DIFFICULTY_DYNAMIC_ACTIVATION_RATIO:
            return DIFFICULTY_DYNAMIC_ACTIVATION_RATIO[self.difficulty]
        return 1.0

    def _get_conflict_vehicle_launch_probability(self) -> float:
        """Return the violation-consequence probability for this task."""
        override = getattr(
            self,
            '_red_light_conflict_vehicle_probability_override',
            None,
        )
        if override is not None:
            return float(override)
        return float(
            CONFLICT_VEHICLE_LAUNCH_PROBABILITY.get(self.difficulty, 1.0)
        )

    def _get_pedestrian_activation_ratio(self) -> float:
        if not getattr(self, "pedestrians_enabled", True):
            return 0.0
        if getattr(self, "load_all_unsafe_triggers", False):
            return 1.0
        return float(self.difficulty_config.get("pedestrian_ratio", self._get_dynamic_activation_ratio()))

    def _get_irregular_activation_ratio(self) -> float:
        if not getattr(self, "irregular_npcs_enabled", True):
            return 0.0
        if getattr(self, "load_all_unsafe_triggers", False):
            return 1.0
        return float(self.difficulty_config.get("irregular_ratio", self._get_dynamic_activation_ratio()))

    def _get_movable_obstacle_activation_ratio(self) -> float:
        if not getattr(self, "movable_obstacles_enabled", True):
            return 0.0
        if getattr(self, "load_all_unsafe_triggers", False):
            return 1.0
        return float(self.difficulty_config.get("movable_obstacle_ratio", self._get_dynamic_activation_ratio()))

    def _get_falling_object_activation_ratio(self) -> float:
        if not getattr(self, "falling_objects_enabled", True):
            return 0.0
        if getattr(self, "load_all_unsafe_triggers", False):
            return 1.0
        return float(self.difficulty_config.get("falling_ratio", self._get_dynamic_activation_ratio()))

    def _prepare_dynamic_obstacle_selection(self) -> None:
        """Select live dynamic actors and remove only those from static input.

        Selection uses a dedicated task-scoped RNG for each actor kind. This
        makes easy/medium/default nested slices of the same actor ordering and
        prevents unrelated random draws from changing selected identities. It
        runs before RTAgent/RTEvaluator construction so a selected actor cannot
        be evaluated at both its authored JSON position and its live UE position.
        """
        movable_ids = list(getattr(self, 'movable_obstacle_ids', []) or [])
        falling_ids = list(getattr(self, 'falling_object_ids', []) or [])

        def select(actor_ids, ratio, actor_kind):
            count = int(round(len(actor_ids) * float(ratio)))
            count = max(0, min(len(actor_ids), count))
            selection_seed = (
                f"{getattr(self, 'seed', 0)}:"
                f"{getattr(self, 'current_task_id', 0)}:{actor_kind}"
            )
            ordered_ids = sorted(actor_ids, key=str)
            random.Random(selection_seed).shuffle(ordered_ids)
            return ordered_ids[:count]

        self.selected_movable_obstacle_ids = select(
            movable_ids,
            self._get_movable_obstacle_activation_ratio(),
            'movable_obstacle',
        )
        self.selected_falling_object_ids = select(
            falling_ids,
            self._get_falling_object_activation_ratio(),
            'falling_object',
        )

        obstacle_by_id = {
            str(obstacle.get('id')): obstacle
            for obstacle in getattr(self, 'map_static_obstacles', [])
            if obstacle.get('id') is not None
        }
        metadata = {}
        for kind, actor_ids in (
            ('movable_obstacle', self.selected_movable_obstacle_ids),
            ('falling_object', self.selected_falling_object_ids),
        ):
            for actor_id in actor_ids:
                source = obstacle_by_id.get(str(actor_id), {})
                metadata[str(actor_id)] = {
                    'id': str(actor_id),
                    'kind': kind,
                    'type': source.get('type', 'unknown'),
                    'initial_location_cm': [
                        float(source.get('x', 0.0)),
                        float(source.get('y', 0.0)),
                        float(source.get('z', 0.0)),
                    ],
                }
        self.dynamic_obstacle_metadata = metadata

        dynamic_ids = set(metadata)
        static_count_before = len(getattr(self, 'static_obstacles', []))
        self.static_obstacles = [
            obstacle
            for obstacle in getattr(self, 'static_obstacles', [])
            if str(obstacle.get('id')) not in dynamic_ids
        ]
        self.logger.info(
            'Prepared live dynamic obstacle tracking: movable=%s falling=%s; '
            'removed %s selected actors from static evaluator input',
            len(self.selected_movable_obstacle_ids),
            len(self.selected_falling_object_ids),
            static_count_before - len(self.static_obstacles),
        )
        # Scripted pedestrians are already live safety obstacles because their
        # RT_ names are discovered by RTEvaluator. Register them here as well
        # so result telemetry proves their displacement and heartbeat state;
        # their predicted collision geometry remains the same 2-D trajectory.
        for kind, actors in (
            ('scripted_pedestrian', getattr(self, 'pedestrians', [])),
            ('irregular_npc', getattr(self, 'irregular_pedestrians', [])),
        ):
            for actor in actors:
                name = self.communicator.get_pedestrian_name(actor.id)
                self.dynamic_obstacle_metadata[name] = {
                    'id': name,
                    'kind': kind,
                    'initial_location_cm': [
                        float(actor.position.x),
                        float(actor.position.y),
                        110.0,
                    ],
                }
        self._scripted_pedestrian_motion_state = {}
        self._scripted_pedestrian_motion_events = []
        self._next_scripted_pedestrian_motion_sample_s = 0.0
        self._scripted_pedestrians_started = False

    def _get_pedestrian_speed_options(self) -> list[float]:
        return list(self.difficulty_config.get("pedestrian_speeds", [100, 150, 200]))
        
    def _load_static_obstacles(self):
        """
        Load static obstacles from the map's obstacles.json file.
        
        Returns:
            List[Dict]: List of static obstacles with their positions
        """
        if not hasattr(self, 'current_map_path') or self.current_map_path is None:
            self.logger.warning("No map path available, cannot load static obstacles")
            return []
        
        # Derive obstacles.json path from map_path
        # map_path is like "data/map1_10roads/progen_world.json"
        # obstacles path should be "data/map1_10roads/obstacles.json"
        import os
        map_dir = os.path.dirname(self.current_map_path)
        obstacles_path = os.path.join(map_dir, 'obstacles.json')
        
        if not os.path.exists(obstacles_path):
            self.logger.warning(f"Obstacles file not found: {obstacles_path}")
            return []
        
        try:
            with open(obstacles_path, 'r', encoding='utf-8') as f:
                obstacles_data = json.load(f)
            
            # Extract obstacle positions
            static_obstacles = []
            nodes = obstacles_data.get('nodes', [])
            
            for node in nodes:
                properties = node.get('properties', {})
                location = properties.get('location', {})
                
                # Only include obstacles that have valid location data
                if 'x' in location and 'y' in location:
                    obstacle = {
                        'id': node.get('id', 'unknown'),
                        'type': node.get('instance_name', 'unknown'),
                        'x': location['x'],
                        'y': location['y'],
                        'z': location.get('z', 0)
                    }
                    static_obstacles.append(obstacle)
            
            buildings_path = os.path.join(map_dir, 'buildings.json')
            if os.path.exists(buildings_path):
                with open(buildings_path, 'r', encoding='utf-8') as f:
                    buildings_data = json.load(f)

                for idx, building in enumerate(buildings_data.get('buildings', [])):
                    bounds = building.get('bounds', {})
                    center = building.get('center', {})
                    required_bounds = ('x', 'y', 'width', 'height')
                    if not all(key in bounds for key in required_bounds):
                        continue

                    # Building exports are in map meters; the simulator state uses centimeters.
                    static_obstacles.append({
                        'id': f"building_{idx}",
                        'type': building.get('type', 'building'),
                        'kind': 'building',
                        'x': center.get('x', bounds['x'] + bounds['width'] / 2) * 100,
                        'y': center.get('y', bounds['y'] + bounds['height'] / 2) * 100,
                        'bounds': {
                            'min_x': bounds['x'] * 100,
                            'min_y': bounds['y'] * 100,
                            'max_x': (bounds['x'] + bounds['width']) * 100,
                            'max_y': (bounds['y'] + bounds['height']) * 100,
                        }
                    })

            self.logger.info(f"Loaded {len(static_obstacles)} static obstacles from {obstacles_path} to agent")
            return static_obstacles
            
        except Exception as e:
            self.logger.error(f"Error loading static obstacles from {obstacles_path}: {e}")
            return []

    def _load_movable_obstacle_ids(self):
        """
        Load IDs of movable obstacles from the map's obstacles.json file.
        Movable obstacles have IDs starting with 'GEN_RT_RT_'.
        
        Returns:
            List[str]: List of movable obstacle IDs
        """
        if not hasattr(self, 'current_map_path') or self.current_map_path is None:
            self.logger.warning("No map path available, cannot load movable obstacles")
            return []
        
        map_dir = os.path.dirname(self.current_map_path)
        obstacles_path = os.path.join(map_dir, 'obstacles.json')
        
        if not os.path.exists(obstacles_path):
            self.logger.warning(f"Obstacles file not found: {obstacles_path}")
            return []
        
        try:
            with open(obstacles_path, 'r', encoding='utf-8') as f:
                obstacles_data = json.load(f)
            
            movable_ids = []
            nodes = obstacles_data.get('nodes', [])
            for node in nodes:
                node_id = node.get('id', '')
                if node_id.startswith('GEN_RT_RT_'):
                    movable_ids.append(node_id)
            
            self.logger.info(f"Found {len(movable_ids)} movable obstacles from {obstacles_path}")
            return movable_ids
            
        except Exception as e:
            self.logger.error(f"Error loading movable obstacles from {obstacles_path}: {e}")
            return []

    def _setup_qt_application(self):
        """Setup Qt application for human control mode."""
        try:
            if QApplication is None or HumanControlInterface is None:
                raise ImportError("PyQt5 human-control dependencies unavailable")
            # Check if QApplication already exists
            app = QApplication.instance()
            if app is None:
                self.qt_app = QApplication(sys.argv)
                self.logger.info("Created new Qt application for human control mode")
            else:
                self.qt_app = app
                self.logger.info("Using existing Qt application for human control mode")
            
            # Create and show the human control interface
            self.agent.human_interface = HumanControlInterface()
            self.agent.human_interface.set_action_callback(self.agent.on_human_action_selected)
            self.agent.human_interface.show_interface()
            
            # Ensure the application is properly initialized
            self.qt_app.processEvents()
            self.logger.info("Qt application and human control interface initialized")
            
        except ImportError as e:
            self.logger.warning(f"PyQt5 not available for human control mode: {e}")
            self.logger.warning("Please install PyQt5: pip install PyQt5")
            self.control_mode = 'llm'  # Fallback to LLM mode

    def _require_ue_actors(self, required_names, context):
        """Fail fast when a required packaged actor did not actually spawn."""
        available = {
            str(name) for name in self.communicator.unrealcv.get_objects()
        }
        missing = sorted(set(required_names) - available)
        if missing:
            raise RuntimeError(
                f"{context} failed to spawn required UE actors: "
                + ", ".join(missing)
            )

    def _spawn_and_setup_traffic_signals(self):
        """
        Spawn all traffic signals and setup simulation for all intersections.
        
        This method:
        1. Spawns all traffic signals from the traffic controller
        2. Creates BP_Intersection blueprints for all intersections
        3. Adds traffic signals to these intersections
        4. Starts simulation for all intersections
        """
        if not hasattr(self, 'traffic_controller') or self.traffic_controller is None:
            self.logger.warning("Traffic controller not initialized, skipping traffic signal setup")
            return
        
        # Step 1: Spawn all traffic signals
        all_traffic_signals = self.traffic_controller.traffic_signals
        if all_traffic_signals:
            self.communicator.spawn_traffic_signals(
                all_traffic_signals,
                self.traffic_light_model_path,
                self.pedestrian_light_model_path,
                self.traffic_light_scale,
                self.pedestrian_light_scale,
            )
            self._require_ue_actors(
                [
                    f"RT_TRAFFIC_SIGNAL_{signal.id}"
                    for signal in all_traffic_signals
                ],
                "traffic-signal setup",
            )
            self.logger.info(f"Spawned {len(all_traffic_signals)} traffic signals")

            # Apply one shared timing plan before any intersection starts.  The
            # legacy Blueprint accepts vehicle green, vehicle yellow, and one
            # combined pedestrian interval; WALK and clearance remain separate
            # in Python so the upgraded Blueprint can expose them independently.
            timing = self.traffic_phase_timing or TrafficPhaseTiming()
            for light in all_traffic_signals:
                self.communicator.traffic_signal_set_duration(
                    light.id,
                    timing.vehicle_green_s,
                    timing.vehicle_yellow_s,
                    timing.blueprint_pedestrian_green_s,
                )
            self.logger.info(
                "Applied geometry-calibrated traffic phase timing to %d signals: %s",
                len(all_traffic_signals),
                timing.to_dict(),
            )
        else:
            self.logger.info("No traffic signals to spawn")
            return
        
        # Step 2: Only intersections with spawned signals need a controller
        # blueprint.  Spawning empty endpoint blueprints is unnecessary and can
        # destabilize the Vulkan runtime after many dynamic actors are loaded.
        all_intersections = [
            intersection
            for intersection in self.traffic_controller.intersections
            if intersection.traffic_lights or intersection.pedestrian_lights
        ]
        if not all_intersections:
            self.logger.info("No intersections in map, skipping intersection simulation setup")
            return
        
        self.logger.info(
            f"Setting up simulation for {len(all_intersections)} intersections with traffic signals"
        )
        
        for intersection in all_intersections:
            try:
                if self.python_grouped_signal_control:
                    grouped_signals = (
                        list(intersection.traffic_lights)
                        + list(intersection.pedestrian_lights)
                    )
                    for signal in grouped_signals:
                        self.communicator.traffic_signal_switch_to(
                            signal.id, 'all red'
                        )
                        self.traffic_controller.intersection_manager.python_states[
                            signal.id
                        ] = (
                            # Keep the Python state and the rendered material
                            # on the same all-red barrier before phase 0.
                            TrafficSignalState.VEHICLE_RED,
                            TrafficSignalState.PEDESTRIAN_RED,
                        )
                    intersection.cycle_count = 0
                    intersection.active_vehicle_group_index = None
                    intersection.active_vehicle_group_seen_green = False
                    intersection.active_vehicle_group_elapsed_s = 0.0
                    intersection.python_pedestrian_phase = None
                    intersection.python_pedestrian_phase_elapsed_s = 0.0
                    snapshot = self.communicator.get_intersection_traffic_state(
                        intersection,
                        timing=self.traffic_phase_timing,
                    )
                    self.traffic_system_snapshots[intersection.id] = snapshot
                    self.logger.info(
                        "Python grouped control owns intersection %s with "
                        "%s vehicle groups; legacy round-robin BP bypassed",
                        intersection.id,
                        len(intersection.vehicle_movement_groups),
                    )
                    continue

                # Create unique name for intersection BP
                intersection_bp_name = f"RT_Intersection_{intersection.id}"
                
                # Spawn BP_Intersection blueprint
                self.communicator.unrealcv.spawn_bp_asset(
                    self.intersection_controller_model_path,
                    intersection_bp_name
                )
                self._require_ue_actors(
                    [intersection_bp_name],
                    "intersection-controller setup",
                )
                self.logger.info(
                    "Spawned intersection controller %s from %s",
                    intersection_bp_name,
                    self.intersection_controller_model_path,
                )
                
                # Add all traffic signals (both vehicle and pedestrian) to the intersection
                signal_count = 0
                
                # Add vehicle traffic lights
                for light in intersection.traffic_lights:
                    signal_name = 'RT_TRAFFIC_SIGNAL_' + str(light.id)
                    self.communicator.unrealcv.add_vehicle_signal(intersection_bp_name, signal_name)
                    signal_count += 1
                    self.logger.debug(f"Added vehicle signal {signal_name} to {intersection_bp_name}")
                
                # Add pedestrian lights
                for light in intersection.pedestrian_lights:
                    signal_name = 'RT_TRAFFIC_SIGNAL_' + str(light.id)
                    self.communicator.unrealcv.add_pedestrian_signal(intersection_bp_name, signal_name)
                    signal_count += 1
                    self.logger.debug(f"Added pedestrian signal {signal_name} to {intersection_bp_name}")
                
                if signal_count > 0:
                    if self.use_intersection_phase_api:
                        self.communicator.configure_intersection_phase_plan(
                            intersection_bp_name,
                            self.traffic_phase_timing,
                        )
                        self.logger.info(
                            "Configured canonical UE phase plan for %s",
                            intersection_bp_name,
                        )

                    # Start simulation for this intersection
                    self.communicator.unrealcv.traffic_signal_start_simulation(intersection_bp_name)
                    self.logger.info(f"Started simulation for {intersection_bp_name} with {signal_count} signals")

                    # Record the whole intersection as one state.  This is the
                    # canonical environment snapshot used for consistency QA;
                    # it also exposes explicit capability gaps in the packaged
                    # Blueprint (yellow/all-red/clearance are not queryable yet).
                    try:
                        if self.use_intersection_phase_api:
                            snapshot = (
                                self.communicator.get_canonical_intersection_state(
                                    intersection_bp_name
                                )
                            )
                            snapshot['timing'] = self.traffic_phase_timing.to_dict()
                        else:
                            snapshot = self.communicator.get_intersection_traffic_state(
                                intersection,
                                timing=self.traffic_phase_timing,
                            )
                        self.traffic_system_snapshots[intersection.id] = snapshot
                        self.logger.info(
                            "Initial traffic-system state for intersection %s: "
                            "phase=%s safe=%s",
                            intersection.id,
                            snapshot['observed_phase'],
                            snapshot['safe'],
                        )
                    except Exception as state_error:
                        self.logger.warning(
                            "Could not read initial intersection state %s: %s",
                            intersection.id,
                            state_error,
                        )
                else:
                    self.logger.warning(f"No signals to start simulation for {intersection_bp_name}")
                
            except Exception as e:
                raise RuntimeError(
                    f"Error setting up intersection {intersection.id}: {e}"
                ) from e
        
        self.logger.info("Traffic signal setup completed")
