"""Engine-side native RT-SAFE pilot. Import only inside Unreal Editor Python.

All movement uses CharacterMovement and collision sweeps. Safety events come
from engine hit/overlap delegates or annotated traffic-zone entry. Python
slate ticks supervise commands and record time; they never teleport a moving
policy agent to a target. Editor-time placement is the only teleport step.
"""

from __future__ import annotations
import math
import time
import traceback
import unreal

from benchmark.map_transfer.contract import (
    Action,
    TURN_ANGLES,
    EventLedger,
    point_in_polygon,
    local_to_world,
    hit_blocks_move,
)
from benchmark.map_transfer.manifest import validate_manifest
from benchmark.map_transfer.hazards import HazardEffects
from benchmark.map_transfer.dynamics import (
    spawn_falling,
    release_falling_objects,
    activate_conflict_vehicles,
)
from benchmark.map_transfer.fixtures import (
    spawn_movable,
    spawn_robot,
    disable_robot_visual_collision,
    animate_robot,
)

PREFIX = "RTSAFE_NYC_PILOT:"
_state = None
_prepared = None
_keep = []
_anim_cache = {}


def _vec(values):
    return unreal.Vector(*map(float, values))


def _xyz(value):
    return [float(value.x), float(value.y), float(value.z)]


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _world(editor=False):
    sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    return sub.get_editor_world() if editor else sub.get_game_world()


def _ground(world, position, ignored=(), *, rise=150, drop=500):
    x, y, z = position
    hit = unreal.SystemLibrary.line_trace_single(
        world,
        _vec((x, y, z + rise)),
        _vec((x, y, z - drop)),
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
        True,
        list(ignored),
        unreal.DrawDebugTrace.NONE,
        True,
    )
    if not hit or not hit.to_tuple()[0] or hit.to_tuple()[1]:
        raise RuntimeError(f"No native ground at {position}")
    if hit.to_tuple()[6].z < 0.65:
        raise RuntimeError(f"Ground probe found a non-walkable surface at {position}")
    return float(hit.to_tuple()[4].z)


def _check_spawn_clearance(world, actor):
    p = actor.get_actor_location()
    cap = actor.capsule_component
    hits = (
        unreal.SystemLibrary.capsule_trace_multi(
            world,
            p,
            p + unreal.Vector(0, 0, 0.1),
            cap.get_scaled_capsule_radius() * 0.98,
            cap.get_scaled_capsule_half_height() * 0.98,
            unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
            False,
            [actor],
            unreal.DrawDebugTrace.NONE,
            True,
        )
        or []
    )
    if any(hit.to_tuple()[0] and hit.to_tuple()[1] for hit in hits):
        raise RuntimeError(
            "Initial capsule intersects blocking geometry: " + actor.get_actor_label()
        )
    return {
        "actor": actor.get_actor_label(),
        "position_cm": _xyz(p),
        "initial_penetration": False,
    }


def _fixture_ground(world, position, spec, yaw, ignored):
    """Support a box footprint above sloping native ground, not just its center."""
    defaults = {
        "robot": [72, 42, 50],
        "vehicle": [450, 185, 110],
        "movable": [80, 80, 80],
        "obstacle": [100, 100, 100],
    }
    size = spec.get("size_cm", defaults[spec["kind"]])
    heights = []
    for forward in (-size[0] / 2, 0, size[0] / 2):
        for right in (-size[1] / 2, 0, size[1] / 2):
            point = local_to_world(position, yaw, forward, right)
            heights.append(_ground(world, point, ignored))
    return max(heights)


def _tag(actor, id_):
    actor.tags = [unreal.Name(PREFIX + id_)]
    actor.set_actor_label(PREFIX + id_)


def _check_fixture_clearance(world, actor, spec):
    defaults = {
        "robot": [72, 42, 50],
        "vehicle": [450, 185, 110],
        "movable": [80, 80, 80],
        "falling": [80, 80, 80],
        "obstacle": [100, 100, 100],
    }
    extent = _vec([x * 0.49 for x in spec.get("size_cm", defaults[spec["kind"]])])
    center = actor.get_actor_location()
    hits = (
        unreal.SystemLibrary.box_trace_multi(
            world,
            center,
            center + unreal.Vector(0, 0, 0.1),
            extent,
            actor.get_actor_rotation(),
            unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
            False,
            [actor],
            unreal.DrawDebugTrace.NONE,
            True,
        )
        or []
    )
    blockers = [
        hit.to_tuple()[9].get_actor_label()
        for hit in hits
        if hit.to_tuple()[0] and hit.to_tuple()[1]
    ]
    if blockers:
        raise RuntimeError(
            f"Initial fixture {spec['id']} intersects blocking geometry: {blockers}"
        )
    return {
        "actor": actor.get_actor_label(),
        "position_cm": _xyz(center),
        "initial_penetration": False,
    }


def _mesh_actor(es, id_, position, size, *, color, collision=True, yaw=0):
    actor = es.spawn_actor_from_class(
        unreal.StaticMeshActor, _vec(position), unreal.Rotator(pitch=0, yaw=yaw, roll=0)
    )
    _tag(actor, id_)
    c = actor.static_mesh_component
    c.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Cube"))
    actor.set_actor_scale3d(_vec([x / 100 for x in size]))
    c.set_collision_profile_name("BlockAll" if collision else "NoCollision")
    c.set_notify_rigid_body_collision(collision)
    base = unreal.load_asset("/Engine/BasicShapes/BasicShapeMaterial")
    material = unreal.MaterialLibrary.create_dynamic_material_instance(
        _world(True), base
    )
    material.set_vector_parameter_value("Color", unreal.LinearColor(*color, 1))
    c.set_material(0, material)
    _keep.append(material)
    return actor


def _spawn_vehicle(es, spec, task, position):
    """Native vehicle visual on a swept box collider, with no teleport motion.

    This is explicitly a kinematic traffic fixture, not a Chaos drivetrain.
    The city MassTraffic visual has no blocking collider in this project.
    """
    path = spec.get("path_cm", [])
    yaw = (
        task["yaw_deg"]
        if len(path) < 2
        else math.degrees(math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0]))
    )
    size = spec.get("size_cm", [450, 185, 110])
    id_ = spec["id"]
    root = _mesh_actor(
        es,
        id_,
        [
            position[0],
            position[1],
            position[2] + size[2] / 2 + spec.get("ground_clearance_cm", 40),
        ],
        size,
        color=(0.1, 0.1, 0.1),
        yaw=yaw,
    )
    root.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    root.static_mesh_component.set_visibility(False, False)
    for suffix, mesh_path in [
        ("body", "/Game/Vehicle/vehCar_vehicle07/Mesh/SM_vehCar_vehicle07"),
        (
            "glass",
            "/Game/Vehicle/vehCar_vehicle07/Mesh/Transparent/SM_All_Trans_vehCar_vehicle07",
        ),
    ]:
        mesh = unreal.load_asset(mesh_path)
        if mesh is None:
            raise RuntimeError("Missing native vehicle mesh " + mesh_path)
        visual = es.spawn_actor_from_class(
            unreal.StaticMeshActor,
            _vec(position),
            unreal.Rotator(pitch=0, yaw=yaw, roll=0),
        )
        _tag(visual, id_ + ":" + suffix)
        component = visual.static_mesh_component
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
        component.set_static_mesh(mesh)
        component.set_collision_profile_name("NoCollision")
        visual.set_actor_enable_collision(False)
        if suffix == "body":
            center, extent = visual.get_actor_bounds(False)
            lift = position[2] - (center.z - extent.z)
        visual.set_actor_location(
            _vec([position[0], position[1], position[2] + lift]), False, False
        )
        visual.attach_to_actor(
            root,
            "",
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            False,
        )
    return root


def _play(actor, walking=False, pedestrian=False):
    if pedestrian:
        path = "/Game/Crowd/Character/Anims/Loco/" + (
            "MTN_N_Walk_RF_InPlace" if walking else "MTN_N_Idle"
        )
    else:
        path = (
            "/Game/Interactable_InteractionKitVol3/Demo/Characters/Mannequins/Animations/Manny/"
            + ("MM_Walk_InPlace" if walking else "MM_Idle")
        )
    if path not in _anim_cache:
        _anim_cache[path] = unreal.load_asset(path)
    animation = _anim_cache[path]
    if animation is None:
        raise RuntimeError("Missing native animation " + path)
    actor.mesh.play_animation(animation, True)


def _prepare_signals(es, task, agent):
    for zone in task.get("traffic_zones", []):
        p = list(zone["signal_position_cm"])
        p[2] = _ground(_world(True), p, [agent])
        yaw = task["yaw_deg"] + 180
        _mesh_actor(
            es,
            "signal:" + zone["id"] + ":pole",
            [p[0], p[1], p[2] + 100],
            [8, 8, 200],
            color=(0.06, 0.06, 0.06),
        )
        _mesh_actor(
            es,
            "signal:" + zone["id"] + ":board",
            [p[0], p[1], p[2] + 210],
            [8, 240, 100],
            color=(0.025, 0.025, 0.025),
            yaw=yaw,
        )
        angle = math.radians(yaw)
        text = es.spawn_actor_from_class(
            unreal.TextRenderActor,
            _vec((p[0] + 6 * math.cos(angle), p[1] + 6 * math.sin(angle), p[2] + 210)),
            unreal.Rotator(pitch=0, yaw=yaw, roll=0),
        )
        _tag(text, "signal:" + zone["id"] + ":text")
        component = text.get_component_by_class(unreal.TextRenderComponent)
        component.set_world_size(40)
        component.set_text("STOP")
        component.set_text_render_color(unreal.Color(r=255, g=70, b=60, a=255))


def _spawn_character(es, id_, position, yaw, agent_config, pedestrian=False):
    cls = unreal.load_class(None, agent_config["class_path"])
    if cls is None:
        raise RuntimeError("Native humanoid class unavailable")
    z = _ground(_world(True), position) + agent_config["capsule_half_height_cm"] + 4
    actor = es.spawn_actor_from_class(
        cls,
        _vec((position[0], position[1], z)),
        unreal.Rotator(pitch=0, yaw=yaw, roll=0),
    )
    _tag(actor, id_)
    actor.set_editor_property("AgentTag", unreal.Name(PREFIX + id_))
    cap = actor.capsule_component
    cap.set_capsule_size(
        agent_config["capsule_radius_cm"], agent_config["capsule_half_height_cm"]
    )
    cap.set_collision_response_to_all_channels(unreal.CollisionResponseType.ECR_BLOCK)
    cap.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
    cap.set_notify_rigid_body_collision(True)
    cap.set_editor_property("generate_overlap_events", True)
    mesh_path = (
        "/Game/Character/Player/Male/Meshes/SKM_PlayerMale"
        if pedestrian
        else agent_config["mesh_path"]
    )
    mesh = unreal.load_asset(mesh_path)
    if mesh is None:
        raise RuntimeError("Missing native visual asset " + mesh_path)
    actor.mesh.set_skeletal_mesh_asset(mesh)
    actor.mesh.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    if not pedestrian:
        # Keep the eye inside the blocking capsule. Moving it ahead of the
        # capsule hides nearby obstacles by clipping through their surface.
        # Hide only the robot's own visual mesh in scene captures instead.
        actor.mesh.set_hidden_in_scene_capture(True)
    _play(actor, False, pedestrian)
    actor.character_movement.set_walkable_floor_angle(45)
    # UE's default push factor can launch light props tens of meters. Make
    # this pilot's controlled-contact parameters explicit in the manifest.
    interaction = agent_config.get("physics_interaction", {})
    for key, default in [
        ("initial_push_force_factor", 2000.0),
        ("push_force_factor", 2500.0),
        ("touch_force_factor", 0.0),
    ]:
        actor.character_movement.set_editor_property(key, interaction.get(key, default))
    actor.agent_set_max_speed(agent_config["speed_cm_s"])
    # The evaluated view is first-person: head height, 100-degree FOV,
    # downward pitch matching the legacy observation convention.
    for arm in actor.get_components_by_class(unreal.SpringArmComponent):
        arm.target_arm_length = 0
        arm.set_relative_location(unreal.Vector(0, 0, 85), False, False)
        arm.set_relative_rotation(
            unreal.Rotator(pitch=agent_config["camera_pitch_deg"], yaw=0, roll=0),
            False,
            False,
        )
    return actor


def prepare(manifest, task_id, seed=0):
    """Place a task in the editor. Never saves or modifies source assets."""
    global _prepared
    validate_manifest(manifest)
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise RuntimeError("End PIE before preparing a fresh task")
    w = _world(True)
    # Imported district-scale mesh distance fields falsely occlude street-level
    # ambient light. Raster lighting plus SSAO avoids that map artifact.
    for command in (
        "r.DistanceFieldAO 0",
        "r.AmbientOcclusionLevels 2",
        "r.Lumen.DiffuseIndirect.Allow 0",
        "r.Lumen.Reflections.Allow 0",
    ):
        unreal.SystemLibrary.execute_console_command(w, command)
    expected = manifest["map_path"]
    if w.get_path_name().split(".")[0] != expected:
        raise RuntimeError(f"Expected {expected}, got {w.get_path_name()}")
    task = next(t for t in manifest["tasks"] if t["id"] == task_id)
    es = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in es.get_all_level_actors():
        if any(str(t).startswith((PREFIX, "RTSAFE_TRANSFER_")) for t in actor.tags):
            es.destroy_actor(actor)
        elif manifest.get("disable_ambient_spawners") and actor.get_actor_label() in (
            "MassTrafficVehicleSpawner",
            "MassCrowdSpawner",
        ):
            actor.set_editor_property("auto_spawn_on_begin_play", False)
    _keep.clear()
    agent = _spawn_character(
        es, "agent", task["start_cm"], task["yaw_deg"], manifest["agent"]
    )
    entities = {}
    for spec in task.get("entities", []):
        p = list(spec["position_cm"])
        kind = spec["kind"]
        id_ = spec["id"]
        ignored = [
            a
            for a in es.get_all_level_actors()
            if any(str(t).startswith(PREFIX) for t in a.tags)
        ]
        p[2] = _ground(w, p, ignored)
        if kind in ("obstacle", "movable", "robot", "vehicle"):
            path = spec.get("path_cm", [])
            yaw = task["yaw_deg"]
            if kind in ("robot", "vehicle") and len(path) >= 2:
                yaw = math.degrees(math.atan2(path[1][1] - p[1], path[1][0] - p[0]))
            p[2] = _fixture_ground(w, p, spec, yaw, ignored)
        if kind == "pedestrian":
            path = spec.get("path_cm", [p, p])
            yaw = math.degrees(math.atan2(path[-1][1] - p[1], path[-1][0] - p[0]))
            actor = _spawn_character(es, id_, p, yaw, manifest["agent"], True)
            actor.agent_set_max_speed(spec.get("speed_cm_s", 90))
        elif kind == "vehicle":
            actor = _spawn_vehicle(es, spec, task, p)
        elif kind == "robot":
            actor = spawn_robot(es, spec, task, p, _mesh_actor, _tag)
        elif kind == "movable":
            actor = spawn_movable(es, spec, task, p, _mesh_actor)
        elif kind == "falling":
            actor = spawn_falling(es, spec, task, p, spawn_movable, _mesh_actor)
        elif kind == "obstacle":
            size = spec.get("size_cm", [100, 100, 100])
            actor = _mesh_actor(
                es,
                id_,
                [p[0], p[1], p[2] + size[2] / 2 + 2],
                size,
                color=(0.55, 0.32, 0.12),
                yaw=task["yaw_deg"],
            )
        else:
            size = spec["size_cm"]
            color = {
                "water": (0.04, 0.5, 0.85),
                "oil": (0.08, 0.045, 0.02),
                "trip": (0.65, 0.25, 0.06),
            }[kind]
            visual = _mesh_actor(
                es,
                id_ + ":visual",
                [p[0], p[1], p[2] + 2],
                [size[0], size[1], 4],
                color=color,
                collision=False,
                yaw=task["yaw_deg"],
            )
            actor = es.spawn_actor_from_class(
                unreal.TriggerBox,
                _vec((p[0], p[1], p[2] + 45)),
                unreal.Rotator(pitch=0, yaw=task["yaw_deg"], roll=0),
            )
            _tag(actor, id_)
            box = actor.get_component_by_class(unreal.BoxComponent)
            box.set_box_extent(unreal.Vector(size[0] / 2, size[1] / 2, 50))
            box.set_collision_profile_name("Trigger")
            box.set_editor_property("generate_overlap_events", True)
        entities[id_] = actor.get_path_name()
    _prepare_signals(es, task, agent)
    goal = list(task["goal_cm"])
    goal[2] = _ground(w, goal, [agent])
    _mesh_actor(
        es,
        "goal",
        [goal[0], goal[1], goal[2] + 2],
        [150, 150, 4],
        color=(0.08, 0.75, 0.4),
        collision=False,
        yaw=task["yaw_deg"],
    )
    clearance = [_check_spawn_clearance(w, agent)]
    for spec in task.get("entities", []):
        if spec["kind"] == "pedestrian":
            clearance.append(
                _check_spawn_clearance(
                    w, unreal.load_object(None, entities[spec["id"]])
                )
            )
        elif spec["kind"] in ("obstacle", "movable", "robot", "vehicle", "falling"):
            clearance.append(
                _check_fixture_clearance(
                    w, unreal.load_object(None, entities[spec["id"]]), spec
                )
            )
    _prepared = {"manifest": manifest, "task": task, "seed": int(seed)}
    return {
        "task_id": task_id,
        "seed": int(seed),
        "agent": agent.get_path_name(),
        "entities": entities,
        "ground_goal_cm": goal,
        "spawn_clearance": clearance,
        "runtime": {
            "engine_version": unreal.SystemLibrary.get_engine_version(),
            "native_class": agent.get_class().get_path_name(),
            "map": w.get_path_name(),
            "ambient_spawners_disabled": manifest.get(
                "disable_ambient_spawners", False
            ),
        },
    }


def cleanup_scene():
    """Remove only owned fixtures, in a separate editor frame before placement."""
    global _prepared
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise RuntimeError("End PIE before clearing fixtures")
    es = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    count = 0
    for actor in es.get_all_level_actors():
        if any(str(tag).startswith((PREFIX, "RTSAFE_TRANSFER_")) for tag in actor.tags):
            actor.set_actor_enable_collision(False)
            es.destroy_actor(actor)
            count += 1
    _keep.clear()
    _prepared = None
    return {"removed_fixtures": count}


def begin_play():
    if not _prepared:
        raise RuntimeError("Prepare a task first")
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_begin_play()
    return {"requested": True}


def game_ready():
    w = _world()
    return bool(
        w and unreal.GameplayStatics.get_all_actors_with_tag(w, PREFIX + "agent")
    )


def attach(mode="realtime"):
    global _state
    if mode not in ("static", "realtime"):
        raise ValueError("Unknown evaluation mode")
    if _state:
        detach()
    w = _world()
    if not w:
        raise RuntimeError("PIE is not running")
    actors = {}
    for actor in unreal.GameplayStatics.get_all_actors_of_class(w, unreal.Actor):
        for tag in actor.tags:
            if str(tag).startswith(PREFIX):
                actors[str(tag)[len(PREFIX) :]] = actor
    agent = actors["agent"]
    _play(agent)
    now = unreal.GameplayStatics.get_time_seconds(w)
    _state = {
        "world": w,
        "agent": agent,
        "actors": actors,
        "task": _prepared["task"],
        "manifest": _prepared["manifest"],
        "seed": _prepared["seed"],
        "mode": mode,
        "phase": "setup",
        "ledger": EventLedger(),
        "start_time": now,
        "start_wall_time": time.monotonic(),
        "terminal_wall_time": None,
        "last_time": now,
        "previous_position": _xyz(agent.get_actor_location()),
        "traveled_cm": 0.0,
        "command": None,
        "last_command": None,
        "contacts": {},
        "raw_hits": 0,
        "pedestrians": {},
        "vehicles": {},
        "robots": {},
        "falling": {},
        "activations": [],
        "trajectory": [],
        "last_pose_time": -1e9,
        "signals": {},
        "errors": [],
        "goal_reached": False,
        "traffic_inside": {},
        "callbacks": [],
        "hazard_effects": HazardEffects(_prepared["manifest"].get("hazard_effects")),
    }
    agent.on_actor_hit.add_callable(_on_hit)
    _state["callbacks"].append((agent.on_actor_hit, _on_hit))
    for spec in _state["task"].get("entities", []):
        actor = actors[spec["id"]]
        if spec["kind"] in ("water", "oil", "trip"):
            # UE 5.8's Python wrapper for TriggerBox actor-overlap delegates
            # fails on this build. Read the engine's overlap set each tick.
            pass
        elif spec["kind"] == "pedestrian":
            _state["pedestrians"][spec["id"]] = {
                "spec": spec,
                "index": 1,
                "done": len(spec.get("path_cm", [])) < 2,
            }
        elif spec["kind"] == "vehicle":
            for suffix in ("body", "glass"):
                visual = actors[spec["id"] + ":" + suffix]
                visual.set_actor_enable_collision(False)
                visual.static_mesh_component.set_collision_profile_name("NoCollision")
                actor.static_mesh_component.ignore_actor_when_moving(visual, True)
            _state["vehicles"][spec["id"]] = {
                "spec": spec,
                "index": 1,
                "done": len(spec.get("path_cm", [])) < 2,
                "activated": not bool(spec.get("activation_zone")),
            }
        elif spec["kind"] == "falling":
            _state["falling"][spec["id"]] = {"spec": spec, "released": False}
        elif spec["kind"] == "robot":
            visual = actors[spec["id"] + ":visual"]
            disable_robot_visual_collision(visual)
            actor.static_mesh_component.ignore_actor_when_moving(visual, True)
            _state["robots"][spec["id"]] = {
                "spec": spec,
                "index": 1,
                "done": len(spec.get("path_cm", [])) < 2,
            }
    _state["tick_handle"] = unreal.register_slate_post_tick_callback(_tick)
    return status()


def _record(kind, id_, evidence):
    s = _state
    if s["phase"] in ("setup", "terminal"):
        return
    return s["ledger"].enter(
        kind,
        id_,
        simulation_time=unreal.GameplayStatics.get_time_seconds(s["world"])
        - s["start_time"],
        wall_time=time.monotonic(),
        phase=s["phase"],
        evidence=evidence,
    )


def _on_hit(self_actor, other_actor, normal_impulse, hit):
    if not _state or _state["phase"] in ("setup", "terminal"):
        return
    t = hit.to_tuple()
    if not t[0] or t[1]:
        return
    fixture = any(str(tag).startswith(PREFIX) for tag in other_actor.tags)
    # Upward-facing native terrain supports walking. An owned obstacle is
    # still an obstacle when contact happens on its upper surface.
    if float(t[6].z) > 0.65 and not fixture:
        return
    # CharacterMovement can report a curb sweep before successfully stepping
    # onto the road. Low, upward-facing road contacts are terrain support.
    if other_actor.get_class().get_name() == "SimWorldCityGraphSplineActor":
        bottom = (
            self_actor.get_actor_location().z
            - self_actor.capsule_component.get_scaled_capsule_half_height()
        )
        if (
            t[5].z - bottom <= self_actor.character_movement.max_step_height + 5
            and t[6].z > 0.1
        ):
            return
    s = _state
    s["raw_hits"] += 1
    id_ = next(
        (str(x)[len(PREFIX) :] for x in other_actor.tags if str(x).startswith(PREFIX)),
        other_actor.get_path_name(),
    )
    s["contacts"][id_] = {
        "time": unreal.GameplayStatics.get_time_seconds(s["world"]),
        "point": _xyz(t[5]),
        "actor": other_actor,
    }
    _record(
        "collision",
        id_,
        {
            "source": "unreal_blocking_hit",
            "normal": _xyz(t[6]),
            "impact_point_cm": _xyz(t[5]),
            "penetrating": False,
        },
    )
    if id_ in s["vehicles"]:
        terminate("vehicle_collision")
        return
    if (
        s["command"]
        and s["command"]["type"] == "move_to"
        and hit_blocks_move(
            _xyz(s["agent"].get_actor_location()),
            s["command"]["target"],
            _xyz(t[6]),
        )
    ):
        _finish_command("blocked")


def _on_overlap_begin(overlapped_actor, other_actor):
    if not _state or other_actor != _state["agent"]:
        return
    id_ = next(
        str(x)[len(PREFIX) :]
        for x in overlapped_actor.tags
        if str(x).startswith(PREFIX)
    )
    spec = next(x for x in _state["task"]["entities"] if x["id"] == id_)
    _record(
        spec["kind"],
        id_,
        {"source": "unreal_overlap", "trigger": overlapped_actor.get_path_name()},
    )


def _on_overlap_end(overlapped_actor, other_actor):
    if not _state or other_actor != _state["agent"]:
        return
    id_ = next(
        str(x)[len(PREFIX) :]
        for x in overlapped_actor.tags
        if str(x).startswith(PREFIX)
    )
    spec = next(x for x in _state["task"]["entities"] if x["id"] == id_)
    _state["ledger"].leave(spec["kind"], id_)


def _drive_pedestrian(id_):
    entry = _state["pedestrians"][id_]
    spec = entry["spec"]
    path = spec.get("path_cm", [])
    actor = _state["actors"][id_]
    if entry["index"] >= len(path):
        if spec.get("loop") and len(path) > 1:
            entry["index"] = 0
        else:
            actor.agent_stop_agent()
            _play(actor, False, True)
            entry["done"] = True
            return
    target = path[entry["index"]]
    p = _xyz(actor.get_actor_location())
    yaw = math.degrees(math.atan2(target[1] - p[1], target[0] - p[0]))
    actor.set_actor_rotation(unreal.Rotator(pitch=0, yaw=yaw, roll=0), False)
    actor.agent_move_forward()
    _play(actor, True, True)


def _drive_vehicles(delta):
    s = _state
    for id_, entry in {**s["vehicles"], **s["robots"]}.items():
        if entry["done"] or not entry.get("activated", True):
            continue
        spec = entry["spec"]
        path = spec["path_cm"]
        actor = s["actors"][id_]
        p = _xyz(actor.get_actor_location())
        target = path[entry["index"]]
        distance = _distance(p, target)
        if distance < 5:
            entry["index"] += 1
            if entry["index"] >= len(path):
                if spec.get("loop"):
                    entry["index"] = 0
                else:
                    entry["done"] = True
                    continue
            target = path[entry["index"]]
            distance = _distance(p, target)
        if distance <= 1e-6:
            continue
        step = min(distance, spec.get("speed_cm_s", 300) * delta)
        yaw = math.degrees(math.atan2(target[1] - p[1], target[0] - p[0]))
        actor.set_actor_rotation(unreal.Rotator(pitch=0, yaw=yaw, roll=0), False)
        end = [
            p[0] + (target[0] - p[0]) * step / distance,
            p[1] + (target[1] - p[1]) * step / distance,
            p[2],
        ]
        if spec["kind"] == "robot":
            lift = spec.get("size_cm", [72, 42, 50])[2] / 2 + spec.get(
                "ground_clearance_cm", 5
            )
            ground = _ground(
                s["world"],
                [end[0], end[1], p[2] - lift],
                list(s["actors"].values()),
                rise=20,
                drop=100,
            )
            end[2] = ground + lift
        # Sweep the physical root collider; Unreal resolves the contact.
        hit = actor.set_actor_location(_vec(end), True, False)
        if hit and hit.to_tuple()[0]:
            t = hit.to_tuple()
            entry["last_hit"] = {
                "blocking": bool(t[0]),
                "penetrating": bool(t[1]),
                "normal": _xyz(t[6]),
                "impact_point_cm": _xyz(t[5]),
                "fields": [str(x) for x in t],
            }
            entry["done"] = True


def _finish_command(reason):
    s = _state
    s["agent"].agent_stop_agent()
    s["agent"].agent_set_max_speed(s["manifest"]["agent"]["speed_cm_s"])
    _play(s["agent"])
    if s["command"]:
        s["command"]["finished_time"] = unreal.GameplayStatics.get_time_seconds(
            s["world"]
        )
        s["command"]["finish_reason"] = reason
        s["last_command"] = s["command"]
        s["command"] = None
    if s["phase"] != "terminal":
        s["phase"] = "inference"
        if s["mode"] == "static":
            unreal.GameplayStatics.set_game_paused(s["world"], True)


def _traffic_tick(now, position):
    s = _state
    for zone in s["task"].get("traffic_zones", []):
        id_ = zone["id"]
        cycle = (now - s["start_time"] + zone.get("phase_offset_s", 0)) % (
            zone["walk_seconds"] + zone["stop_seconds"]
        )
        walk = cycle >= zone["stop_seconds"]
        if s["signals"].get(id_) != walk:
            text = s["actors"]["signal:" + id_ + ":text"].get_component_by_class(
                unreal.TextRenderComponent
            )
            text.set_text("WALK" if walk else "STOP")
            text.set_text_render_color(
                unreal.Color(r=70, g=255, b=100, a=255)
                if walk
                else unreal.Color(r=255, g=70, b=60, a=255)
            )
            s["signals"][id_] = walk
        inside = point_in_polygon(position, zone["road_polygon_cm"])
        before = s["traffic_inside"].get(id_, False)
        if inside and not before:
            evidence = {
                "source": "annotated_road_entry",
                "position_cm": position,
                "zone_id": id_,
                "signal": "walk" if walk else "stop",
            }
            violations = []
            if not point_in_polygon(position, zone["crosswalk_polygon_cm"]):
                _record("off_crosswalk", id_, evidence)
                violations.append("off_crosswalk")
            if not walk:
                _record("red_light", id_, evidence)
                violations.append("red_light")
            if violations:
                activate_conflict_vehicles(s, id_, violations, now)
        elif not inside and before:
            s["ledger"].leave("red_light", id_)
            s["ledger"].leave("off_crosswalk", id_)
        s["traffic_inside"][id_] = inside


def _tick(delta):
    if not _state:
        return
    s = _state
    try:
        now = unreal.GameplayStatics.get_time_seconds(s["world"])
        if now <= s["last_time"]:
            return
        elapsed = now - s["last_time"]
        s["last_time"] = now
        p = _xyz(s["agent"].get_actor_location())
        if s["phase"] not in ("setup", "terminal"):
            s["traveled_cm"] += _distance(p, s["previous_position"])
        s["previous_position"] = p
        if s["phase"] not in ("setup", "terminal"):
            release_falling_objects(s, now)
            _drive_vehicles(elapsed)
            for id_, entry in s["robots"].items():
                animate_robot(
                    s["actors"][id_ + ":visual"],
                    now - s["start_time"],
                    moving=not entry["done"],
                )
        if s["phase"] == "terminal":
            return
        for id_, contact in list(s["contacts"].items()):
            # Only releases an event: proximity can never CREATE a collision.
            if now - contact["time"] > 0.15:
                radius = s["manifest"]["agent"]["capsule_radius_cm"]
                other = contact["actor"]
                separated = _distance(p, contact["point"]) > radius + 5
                if isinstance(other, unreal.Character):
                    # A pedestrian may walk away while the evaluated agent
                    # remains still. Query its real capsule surface rather
                    # than retaining the original impact point forever.
                    distance, _ = (
                        other.capsule_component.get_closest_point_on_collision(_vec(p))
                    )
                    separated = distance > radius + 5
                elif other.get_component_by_class(unreal.PrimitiveComponent):
                    component = other.get_component_by_class(unreal.PrimitiveComponent)
                    # Query at the last impact height to account for objects
                    # below the capsule center, such as crates and robot dogs.
                    distance, _ = component.get_closest_point_on_collision(
                        _vec([p[0], p[1], contact["point"][2]])
                    )
                    if distance >= 0:
                        separated = distance > radius + 5
                if separated:
                    s["ledger"].leave("collision", id_)
                    del s["contacts"][id_]
        for spec in s["task"].get("entities", []):
            if spec["kind"] in ("water", "oil", "trip"):
                trigger = s["actors"][spec["id"]]
                if trigger.is_overlapping_actor(s["agent"]):
                    entered = _record(
                        spec["kind"],
                        spec["id"],
                        {
                            "source": "unreal_overlap_query",
                            "trigger": trigger.get_path_name(),
                        },
                    )
                    if entered:
                        effect = s["hazard_effects"].enter(spec["kind"], now)
                        if spec["kind"] == "trip":
                            s["agent"].agent_stop_agent()
                            s["agent"].agent_rotate(0, "left")
                            _play(s["agent"])
                            if s["command"]:
                                s["command"]["recovering"] = True
                                s["command"]["recovery_seconds"] = (
                                    s["command"].get("recovery_seconds", 0)
                                    + effect["seconds"]
                                )
                                if s["command"]["type"] == "wait":
                                    s["command"]["end_time"] += effect["seconds"]
                else:
                    s["ledger"].leave(spec["kind"], spec["id"])
        for id_, entry in s["pedestrians"].items():
            if not entry["done"]:
                path = entry["spec"].get("path_cm", [])
                if (
                    _distance(
                        _xyz(s["actors"][id_].get_actor_location()),
                        path[entry["index"]],
                    )
                    < 20
                ):
                    entry["index"] += 1
                    _drive_pedestrian(id_)
        command = s["command"]
        if (
            command
            and command.get("recovering")
            and not s["hazard_effects"].remaining(now)
        ):
            command["recovering"] = False
            _resume_command(command)
        if command and command.get("recovering"):
            command = None
        if command:
            if command["type"] == "move_to" and _distance(p, command["target"]) <= 20:
                _finish_command("target_reached")
            elif command["type"] == "turn_around":
                yaw = s["agent"].get_actor_rotation().yaw
                duration = command["turn_duration_s"] + command["recovery_seconds"]
                if (
                    abs((yaw - command["target_yaw"] + 180) % 360 - 180) < 1
                    and now - command["start_time"] >= duration
                ):
                    _finish_command("turn_complete")
            elif command["type"] == "wait" and now >= command["end_time"]:
                _finish_command("wait_complete")
            if s["command"] and now - command["start_time"] > 15 + command.get(
                "recovery_seconds", 0
            ):
                _finish_command("action_timeout")
        if s["phase"] not in ("setup", "terminal"):
            _traffic_tick(now, p)
            if not s["hazard_effects"].remaining(now) and _distance(
                p, s["task"]["goal_cm"]
            ) <= s["task"].get("goal_radius_cm", 100):
                s["goal_reached"] = True
                terminate("goal_reached")
            elif now - s["start_time"] >= s["task"].get("time_limit_s", 180):
                terminate("time_limit")
            _record_pose()
    except Exception:
        s["errors"].append(traceback.format_exc())
        s["agent"].agent_stop_agent()
        unreal.GameplayStatics.set_game_paused(s["world"], True)
        s["phase"] = "terminal"
        s["terminal_reason"] = "runtime_error"
        s["terminal_wall_time"] = time.monotonic()


def start_episode():
    if not _state:
        raise RuntimeError("Attach to PIE first")
    s = _state
    s["phase"] = "inference"
    s["start_time"] = unreal.GameplayStatics.get_time_seconds(s["world"])
    s["start_wall_time"] = time.monotonic()
    s["previous_position"] = _xyz(s["agent"].get_actor_location())
    s["traveled_cm"] = 0
    if s["mode"] == "static":
        unreal.GameplayStatics.set_game_paused(s["world"], True)
    for id_ in s["pedestrians"]:
        _drive_pedestrian(id_)
    _record_pose(force=True)
    return status()


def act(action):
    action = Action.from_dict(action)
    s = _state
    if s["phase"] == "terminal":
        raise RuntimeError("Episode already ended")
    if s["command"]:
        raise RuntimeError("Previous action is still running")
    s["phase"] = "action"
    unreal.GameplayStatics.set_game_paused(s["world"], False)
    now = unreal.GameplayStatics.get_time_seconds(s["world"])
    command = {
        "type": action.action_type,
        "param": action.action_param,
        "start_time": now,
        "recovery_seconds": s["hazard_effects"].remaining(now),
        "recovering": s["hazard_effects"].remaining(now) > 0,
    }
    a = s["agent"]
    p = _xyz(a.get_actor_location())
    yaw = a.get_actor_rotation().yaw
    if action.action_type == "move_to":
        target = action.target(p, yaw)
        command["target"] = target
        angle = math.degrees(math.atan2(target[1] - p[1], target[0] - p[0]))
        a.set_actor_rotation(unreal.Rotator(pitch=0, yaw=angle, roll=0), False)
        command["speed_factor"] = s["hazard_effects"].start_move()
    elif action.action_type == "turn_around":
        angle = TURN_ANGLES[action.action_param]
        command["target_yaw"] = yaw + angle
        command["turn_duration_s"] = s["manifest"]["agent"].get("turn_duration_s", 1.0)
    else:
        command["end_time"] = (
            now + int(action.action_param) + command["recovery_seconds"]
        )
    s["command"] = command
    if not command["recovering"]:
        _resume_command(command)
    return status()


def _resume_command(command):
    a = _state["agent"]
    if command["type"] == "move_to":
        a.agent_set_max_speed(
            _state["manifest"]["agent"]["speed_cm_s"] * command["speed_factor"]
        )
        a.agent_move_forward()
        _play(a, True)
    elif command["type"] == "turn_around":
        angle = (command["target_yaw"] - a.get_actor_rotation().yaw + 180) % 360 - 180
        elapsed = (
            unreal.GameplayStatics.get_time_seconds(_state["world"])
            - command["start_time"]
        )
        remaining = max(
            1 / 30, command["turn_duration_s"] + command["recovery_seconds"] - elapsed
        )
        a.set_editor_property("RotateRateDegPerSec", abs(angle) / remaining)
        a.agent_rotate(abs(angle), "left" if angle < 0 else "right")


def status():
    if not _state:
        return {"attached": False}
    s = _state
    return {
        "attached": True,
        "task_id": s["task"]["id"],
        "mode": s["mode"],
        "phase": s["phase"],
        "simulation_time": unreal.GameplayStatics.get_time_seconds(s["world"])
        - s["start_time"],
        "wall_time": time.monotonic(),
        "episode_wall_time_s": (s["terminal_wall_time"] or time.monotonic())
        - s["start_wall_time"],
        "world_time": unreal.GameplayStatics.get_time_seconds(s["world"]),
        "episode_start_world_time": s["start_time"],
        "position_cm": _xyz(s["agent"].get_actor_location()),
        "yaw_deg": float(s["agent"].get_actor_rotation().yaw),
        "traveled_cm": s["traveled_cm"],
        "goal_reached": s["goal_reached"],
        "safe_success": s["ledger"].safe_success(s["goal_reached"]),
        "command": s["command"],
        "last_command": s["last_command"],
        "counts": s["ledger"].counts(),
        "events": list(s["ledger"].events),
        "activations": list(s["activations"]),
        "raw_blocking_hits": s["raw_hits"],
        "hazard_effects": [
            {
                **effect,
                "at_simulation_time": effect["at_simulation_time"] - s["start_time"],
            }
            for effect in s["hazard_effects"].history
        ],
        "recovery_remaining_s": s["hazard_effects"].remaining(
            unreal.GameplayStatics.get_time_seconds(s["world"])
        ),
        "errors": list(s["errors"]),
        "terminal_reason": s.get("terminal_reason"),
        "signals": {
            id_: ("walk" if value else "stop") for id_, value in s["signals"].items()
        },
        "entities": {
            id_: _xyz(a.get_actor_location()) for id_, a in s["actors"].items()
        },
        "vehicle_motion": {
            id_: {key: value for key, value in entry.items() if key != "spec"}
            for id_, entry in s["vehicles"].items()
        },
        "robot_motion": {
            id_: {key: value for key, value in entry.items() if key != "spec"}
            for id_, entry in s["robots"].items()
        },
        "trajectory_samples": len(s["trajectory"]),
    }


def _record_pose(force=False):
    s = _state
    now = unreal.GameplayStatics.get_time_seconds(s["world"]) - s["start_time"]
    if not force and now - s["last_pose_time"] < 0.1:
        return
    s["last_pose_time"] = now
    entity_ids = [spec["id"] for spec in s["task"].get("entities", [])]
    s["trajectory"].append(
        {
            "simulation_time": now,
            "wall_time": time.monotonic(),
            "phase": s["phase"],
            "agent_position_cm": _xyz(s["agent"].get_actor_location()),
            "agent_yaw_deg": float(s["agent"].get_actor_rotation().yaw),
            "entities": {
                id_: _xyz(s["actors"][id_].get_actor_location()) for id_ in entity_ids
            },
            "counts": s["ledger"].counts(),
            "recovering": s["hazard_effects"].remaining(now + s["start_time"]) > 0,
        }
    )


def trajectory():
    _record_pose(force=True)
    return {
        "schema": "rtsafe-native-trajectory-v1",
        "sample_interval_s": 0.1,
        "time_basis": "simulation seconds since episode start",
        "samples": _state["trajectory"],
    }


def terminate(reason="stopped"):
    s = _state
    s["terminal_wall_time"] = time.monotonic()
    s["phase"] = "terminal"
    s["terminal_reason"] = reason
    _finish_command(reason)
    s["agent"].agent_rotate(0, "left")
    _play(s["agent"])
    for id_ in s["pedestrians"]:
        s["actors"][id_].agent_stop_agent()
        _play(s["actors"][id_], False, True)
    unreal.GameplayStatics.set_game_paused(s["world"], True)
    _record_pose(force=True)
    return status()


def detach():
    global _state
    if not _state:
        return {"detached": True}
    s = _state
    if s.get("tick_handle"):
        unreal.unregister_slate_post_tick_callback(s["tick_handle"])
    for delegate, callback in s["callbacks"]:
        try:
            delegate.remove_callable(callback)
        except Exception:
            pass
    s["agent"].agent_stop_agent()
    unreal.GameplayStatics.set_game_paused(s["world"], False)
    s["callbacks"].clear()
    s.clear()
    _state = None
    return {"detached": True}


def end_play():
    detach()
    import gc

    gc.collect()  # Release Python delegate wrappers while the PIE world exists.
    subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if subsystem.is_in_play_in_editor():
        subsystem.editor_request_end_play()
    return {"requested": True}


def is_playing():
    return unreal.get_editor_subsystem(
        unreal.LevelEditorSubsystem
    ).is_in_play_in_editor()


def camera_request():
    s = _state
    a = s["agent"]
    cfg = s["manifest"]["agent"]
    capture = a.get_component_by_class(unreal.SceneCaptureComponent2D)
    location = _xyz(capture.get_world_location())
    r = capture.get_world_rotation()
    return {
        "camera_sources": [
            {
                "camera_id": "policy",
                "type": "world",
                "location_cm": location,
                "rotation_degrees": [r.pitch, r.yaw, r.roll],
            }
        ],
        "width": cfg["width"],
        "height": cfg["height"],
        "fov_degrees": cfg["fov_deg"],
        "jpeg_quality": 90,
        "capture_mode": "shared_pool",
        "shared_pool_size": 1,
        "force_capture": True,
        "validate": True,
    }


def waypoints():
    s = _state
    p = _xyz(s["agent"].get_actor_location())
    yaw = s["agent"].get_actor_rotation().yaw
    points = {}
    for key in ("1", "2", "3", "4", "5", "6", "7"):
        target = list(Action("move_to", key).target(p, yaw))
        target[2] = _ground(s["world"], target, list(s["actors"].values())) + 3
        points[key] = target
    return points
