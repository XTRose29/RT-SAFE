"""Native movable-object physics and a swept robot-dog fixture.

Imported only inside Unreal. The robot uses a blocking body collider and
procedural link animation; it is not a MuJoCo locomotion evaluation.
"""

import math
import unreal


def spawn_movable(es, spec, task, position, mesh_actor):
    size = spec.get("size_cm", [80, 80, 80])
    actor = mesh_actor(
        es,
        spec["id"],
        [position[0], position[1], position[2] + size[2] / 2 + 2],
        size,
        color=(0.62, 0.38, 0.16),
        yaw=task["yaw_deg"],
    )
    component = actor.static_mesh_component
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_collision_profile_name("PhysicsActor")
    component.set_notify_rigid_body_collision(True)
    component.set_simulate_physics(True)
    component.set_mass_override_in_kg("", spec.get("mass_kg", 20), True)
    component.set_linear_damping(0.5)
    component.set_angular_damping(1.0)
    return actor


def disable_robot_visual_collision(actor):
    actor.set_actor_enable_collision(False)
    actor.set_actor_tick_enabled(False)
    for component in actor.get_components_by_class(unreal.PrimitiveComponent):
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
        component.set_collision_profile_name("NoCollision")
        component.set_simulate_physics(False)


def spawn_robot(es, spec, task, position, mesh_actor, tag):
    size = spec.get("size_cm", [72, 42, 50])
    path = spec.get("path_cm", [])
    yaw = (
        task["yaw_deg"]
        if len(path) < 2
        else math.degrees(
            math.atan2(path[1][1] - position[1], path[1][0] - position[0])
        )
    )
    lift = size[2] / 2 + spec.get("ground_clearance_cm", 5)
    root = mesh_actor(
        es,
        spec["id"],
        [position[0], position[1], position[2] + lift],
        size,
        color=(0.15, 0.15, 0.15),
        yaw=yaw,
    )
    root.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    root.static_mesh_component.set_visibility(False, False)
    path = "/Game/Agent/Robot/Core/BP_Go1Robot.BP_Go1Robot_C"
    cls = unreal.load_class(None, path)
    if cls is None:
        raise RuntimeError("Missing native robot visual " + path)
    visual = es.spawn_actor_from_class(
        cls, unreal.Vector(*position), unreal.Rotator(pitch=0, yaw=yaw, roll=0)
    )
    tag(visual, spec["id"] + ":visual")
    disable_robot_visual_collision(visual)
    material = unreal.load_asset("/Game/Agent/Robot/go1/M_Go1_Body")
    if material:
        for component in visual.get_components_by_class(unreal.StaticMeshComponent):
            for slot in range(component.get_num_materials()):
                component.set_material(slot, material)
    center, extent = visual.get_actor_bounds(False)
    visual.set_actor_location(
        unreal.Vector(
            position[0], position[1], position[2] + position[2] - center.z + extent.z
        ),
        False,
        False,
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


def animate_robot(actor, elapsed, *, moving):
    """Presentation gait on native links; the separate swept body owns contact."""
    for component in actor.get_components_by_class(unreal.SceneComponent):
        name = component.get_name()
        if not (name.startswith("MjBody_") and name.endswith(("_thigh", "_calf"))):
            continue
        leg = name.split("_")[1]
        front, left = leg.startswith("F"), leg.endswith("L")
        hip = (18.8 if front else -18.8, -12.7 if left else 12.7, 27)
        phase = 0 if leg in ("FL", "RR") else math.pi
        cycle = math.sin(elapsed * 1.4 * 2 * math.pi + phase) if moving else 0
        upper, lower = 30 + 18 * cycle, -30 + 18 * cycle
        if name.endswith("_thigh"):
            location, pitch = hip, upper
        else:
            angle = math.radians(upper)
            location = (
                hip[0] + 21 * math.sin(angle),
                hip[1],
                hip[2] - 21 * math.cos(angle),
            )
            pitch = lower
        component.set_relative_location_and_rotation(
            unreal.Vector(*location),
            unreal.Rotator(pitch=pitch, yaw=0, roll=0),
            False,
            False,
        )
