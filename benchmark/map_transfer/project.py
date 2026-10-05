"""Local-only preparation of an isolated editor project and launch command."""

from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import hashlib
import socket


def source_digests(content: Path, packages: list[str]) -> dict[str, str]:
    """Hash source packages before and after preparation without loading UE."""
    result = {}
    for package in packages:
        package = package.split(".", 1)[0]
        if not package.startswith("/Game/") or ".." in Path(package).parts:
            raise ValueError("Expected a source /Game/ package")
        stem = content / package.removeprefix("/Game/")
        matches = [
            p
            for p in (stem.with_suffix(".uasset"), stem.with_suffix(".umap"))
            if p.is_file()
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one local source package for {package}")
        with matches[0].open("rb") as handle:
            result[package] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def create_project(source_project: Path, destination: Path) -> dict:
    source_project = source_project.resolve()
    destination = destination.resolve()
    if not source_project.is_file() or source_project.suffix != ".uproject":
        raise ValueError("Source must be a local Unreal .uproject")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty")
    source = source_project.parent
    if destination == source or source in destination.parents:
        raise ValueError("Keep the owned derivative outside the source project")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source_project.name
    shutil.copy2(source_project, target)
    shutil.copytree(source / "Config", destination / "Config")
    for name in ("Binaries", "Plugins", "Source", "Build"):
        if (source / name).exists():
            (destination / name).symlink_to(source / name, target_is_directory=True)
    content = destination / "Content"
    content.mkdir()
    for entry in (source / "Content").iterdir():
        if entry.name == "RTSafeNYC":
            continue
        (content / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (content / "RTSafeNYC").mkdir()
    settings = destination / "Config" / "DefaultEditorPerProjectUserSettings.ini"
    with settings.open("a") as f:
        f.write(
            "\n[/Script/UnrealEd.EditorLoadingSavingSettings]\nbAutoSaveEnable=False\n"
        )
    record = {
        "schema": "rtsafe-owned-project-v1",
        "source_project": str(source_project),
        "project": str(target),
        "owned_asset_namespace": "/Game/RTSafeNYC/",
        "shared_content": "linked read-only by convention; never save source packages",
    }
    (destination / "rtsafe-owned-project.json").write_text(json.dumps(record, indent=2))
    return record


def launch_project(
    project: Path, config: Path, sdk_python: Path, output: Path, map_path: str, gpu: int
) -> None:
    project = project.resolve()
    config = config.resolve()
    sdk_python = sdk_python.resolve()
    output = output.resolve()
    marker = project.parent / "rtsafe-owned-project.json"
    if not marker.is_file() or marker.stat().st_uid != os.getuid():
        raise ValueError("Launch requires an owned project created with init-project")
    if not (sdk_python / "spear" / "__init__.py").is_file():
        raise ValueError("--sdk-python must contain the matching spear package")
    if not config.is_file():
        raise ValueError("Missing SPEAR configuration")
    if not map_path.startswith("/Game/") or gpu < 0:
        raise ValueError("Invalid map or GPU")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Launch output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    import spear
    import secrets

    resolved = spear.get_config(user_config_files=[str(config)])
    port = resolved.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"RPC port {port} is already occupied; choose a different port"
            )
    resolved.defrost()
    resolved.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = (
        secrets.randbelow(1_000_000_000) + 1
    )
    resolved.freeze()
    expanded = output / "spear-resolved.yaml"
    expanded.write_text(resolved.dump())
    executable = project.parent / "Binaries/Linux/SimWorldEditor"
    if not executable.is_file():
        raise ValueError("Missing project-matched SimWorldEditor binary")
    bootstrap = Path(__file__).with_name("bootstrap_editor.py").resolve()
    if any(char in str(bootstrap) for char in [",", '"', "\n", " "]):
        raise ValueError("Bootstrap path cannot contain whitespace, comma, or quote")
    cvars = [
        "t.MaxFPS 30",
        "r.DistanceFieldAO 0",
        "r.AmbientOcclusionLevels 2",
        "r.Lumen.DiffuseIndirect.Allow 0",
        "r.Lumen.Reflections.Allow 0",
        "r.Nanite 0",
        "r.Shadow.Virtual.Enable 0",
        "r.DynamicGlobalIlluminationMethod 0",
        "r.ReflectionMethod 0",
    ]
    command = [
        str(executable),
        str(project),
        map_path,
        "-sp-config-file=" + str(expanded),
        "-RenderOffScreen",
        "-norhithread",
        "-corelimit=8",
        "-graphicsadapter=" + str(gpu),
        "-nosplash",
        "-nop4",
        "-nosound",
        "-unattended",
        "-nozenautolaunch",
        "-NoLiveCoding",
        "-NoHotReloadFromIDE",
        "-stdout",
        "-FullStdOutLogOutput",
        "-abslog=" + str(output / "editor.log"),
        "-DDC=NoZenLocalFallback",
        "-NoDDCCleanup",
        "-AssetGatherAll=false",
        "-AssetGatherSync=true",
        "-NoDependsGathering",
        "-ExecCmds=" + ",".join(cvars + [f"py {bootstrap}"]),
    ]
    os.environ["RTSAFE_SPEAR_PYTHON"] = str(sdk_python)
    os.environ["RTSAFE_BOOTSTRAP_STATUS"] = str(output / "bootstrap.json")
    (output / "launch.json").write_text(
        json.dumps(
            {"command": command, "pid": os.getpid(), "sdk_python": str(sdk_python)},
            indent=2,
        )
    )
    (output / "editor.pid").write_text(str(os.getpid()))
    os.execv(str(executable), command)
