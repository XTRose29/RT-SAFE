"""Editor-only collision preparation. Originals are never edited or saved."""

from __future__ import annotations
import hashlib
import unreal


def inventory(source_prefixes):
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise RuntimeError("Geometry preparation requires editor mode")
    sources = {}
    for actor in unreal.get_editor_subsystem(
        unreal.EditorActorSubsystem
    ).get_all_level_actors():
        for c in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = c.static_mesh
            if mesh and any(
                mesh.get_path_name().startswith(p) for p in source_prefixes
            ):
                sources.setdefault(mesh.get_path_name(), []).append(
                    actor.get_actor_label()
                )
    return [
        {"source": src, "actors": actors} for src, actors in sorted(sources.items())
    ]


def prepare_assets(sources, destination_root="/Game/RTSafeNYC/TransferCollision"):
    """Duplicate static geometry locally and use its actual triangle surface.

    Imported district-wide convex hulls can enclose otherwise empty streets.
    This operation replaces those hulls with static triangle collision; it does
    not disable collision. Dynamic actors retain their own simple colliders.
    Call in small batches so the transport timeout remains bounded.
    """
    if not destination_root.startswith("/Game/RTSafeNYC/"):
        raise ValueError("Collision copies must stay in the owned RT-SAFE folder")
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise RuntimeError("End PIE before preparing collision assets")
    settings = unreal.get_default_object(
        unreal.load_class(None, "/Script/UnrealEd.EditorLoadingSavingSettings")
    )
    settings.set_editor_property("bAutoSaveEnable", False)
    ss = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    mapping = {}
    report = []
    for source in sources:
        if source.startswith(destination_root + "/"):
            raise ValueError("Source and destination asset namespaces must differ")
        original = unreal.load_asset(source)
        if not isinstance(original, unreal.StaticMesh):
            raise ValueError("Expected static mesh " + source)
        original_flag = ss.get_collision_complexity(original)
        destination = (
            destination_root + "/SM_" + hashlib.sha256(source.encode()).hexdigest()[:12]
        )
        if unreal.EditorAssetLibrary.does_asset_exist(destination):
            mesh = unreal.load_asset(destination)
        else:
            mesh = unreal.EditorAssetLibrary.duplicate_asset(source, destination)
        if mesh is None:
            raise RuntimeError("Could not duplicate " + source)
        body = mesh.get_editor_property("body_setup")
        if body is None:
            raise RuntimeError("Missing BodySetup in " + destination)
        body.set_editor_property(
            "collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
        )
        ss.remove_collisions(mesh)
        if not unreal.EditorAssetLibrary.save_asset(destination, False):
            raise RuntimeError("Could not save " + destination)
        if ss.get_collision_complexity(original) != original_flag:
            raise RuntimeError("Source collision asset changed unexpectedly")
        mapping[source] = mesh
        report.append(
            {
                "source": source,
                "local": mesh.get_path_name(),
                "collision": "complex_as_simple",
            }
        )
    changed = 0
    for actor in unreal.get_editor_subsystem(
        unreal.EditorActorSubsystem
    ).get_all_level_actors():
        for c in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = c.static_mesh
            if mesh and mesh.get_path_name() in mapping:
                c.set_static_mesh(mapping[mesh.get_path_name()])
                changed += 1
    return {"assets": report, "replaced_components": changed}


def save_owned_map(map_path):
    if not map_path.startswith("/Game/RTSafeNYC/Maps/"):
        raise ValueError("Only an owned derived map can be saved")
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise RuntimeError("End PIE before saving a derived map")
    es = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in es.get_all_level_actors():
        if any(
            str(t).startswith(("RTSAFE_NYC_PILOT:", "RTSAFE_TRANSFER_"))
            for t in actor.tags
        ):
            es.destroy_actor(actor)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if not unreal.EditorLoadingAndSavingUtils.save_map(world, map_path):
        raise RuntimeError("Saving derived map failed")
    return {"saved": map_path}
