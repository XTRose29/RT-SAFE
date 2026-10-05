"""Native falling-object release and traffic-triggered vehicle activation.

These are explicit pilot fixtures. They do not load or reproduce the old
cooked Blueprints' hidden trigger logic.
"""

import unreal


def spawn_falling(es, spec, task, position, spawn_movable, mesh_actor):
    actor = spawn_movable(es, spec, task, position, mesh_actor)
    component = actor.static_mesh_component
    component.set_simulate_physics(False)
    location = actor.get_actor_location()
    location.z += spec["drop_height_cm"]
    actor.set_actor_location(location, False, False)
    body = component.get_editor_property("body_instance")
    body.set_editor_property("use_ccd", True)
    return actor


def release_falling_objects(state, now):
    elapsed = now - state["start_time"]
    for id_, entry in state["falling"].items():
        if entry["released"] or elapsed < entry["spec"]["release_delay_s"]:
            continue
        component = state["actors"][id_].static_mesh_component
        component.set_simulate_physics(True)
        entry["released"] = True
        location = state["actors"][id_].get_actor_location()
        state["activations"].append(
            {
                "actor_id": id_,
                "kind": "falling",
                "simulation_time": elapsed,
                "reason": "authored_release_delay",
                "phase": state["phase"],
                "position_cm": [
                    float(location.x),
                    float(location.y),
                    float(location.z),
                ],
            }
        )


def activate_conflict_vehicles(state, zone_id, violation_types, now):
    for id_, entry in state["vehicles"].items():
        if entry["activated"] or entry["spec"].get("activation_zone") != zone_id:
            continue
        entry["activated"] = True
        state["activations"].append(
            {
                "actor_id": id_,
                "kind": "vehicle",
                "simulation_time": now - state["start_time"],
                "reason": "traffic_violation",
                "zone_id": zone_id,
                "violation_types": list(violation_types),
                "phase": state["phase"],
            }
        )
