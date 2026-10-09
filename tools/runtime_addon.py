#!/usr/bin/env python3
"""Package or install the original RT-SAFE Linux content chunk, without the base.

This tool does not grant redistribution rights or download a SimWorld runtime.
The exact supported base is identified by hashes, not merely its UE version.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

ADDON = "SimWorld/Content/Paks/pakchunk3002-Linux.pak"
BASE_FILES = [
    "SimWorld/Binaries/Linux/SimWorld",
    *[f"SimWorld/Content/Paks/{name}" for name in (
        "pakchunk0-Linux.pak", "pakchunk1000-Linux.pak",
        "pakchunk1001-Linux.pak", "pakchunk1001optional-Linux.pak",
        "pakchunk3001-Linux.pak",
    )],
]


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("platform") != "Linux-x86_64":
        raise ValueError("Unsupported add-on manifest")
    if set(manifest.get("payload", {})) != {ADDON}:
        raise ValueError("Unexpected add-on payload paths")
    if set(manifest.get("base_files", {})) != set(BASE_FILES):
        raise ValueError("Manifest must identify every supported base file")
    for entry in [*manifest["payload"].values(), *manifest["base_files"].values()]:
        if (not isinstance(entry.get("size"), int) or entry["size"] <= 0
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])):
            raise ValueError("Invalid size or SHA-256 in manifest")


def build(runtime: Path, archive: Path, manifest_output: Path) -> dict:
    manifest = {
        "schema_version": 1,
        "name": "RT-SAFE legacy real-time content add-on",
        "platform": "Linux-x86_64",
        "engine_version": "5.3.2-29314046+++UE5+Release-5.3",
        "levels": ["RT10", "RT12", "RT15", "RT18", "RT20"],
        "payload": {ADDON: fingerprint(runtime / ADDON)},
        "base_files": {},
        "distribution_status": "candidate; asset redistribution review required",
        "limitations": [
            "Requires the exact matching SimWorld base, obtained separately.",
            "Cooked Linux content, not an editable Unreal project or Windows build.",
            "Contains sports-ball assets and textures whose notices must be supplied before publication.",
            "Does not include the Unreal executable or UnrealCV plugin source.",
        ],
    }
    for name in BASE_FILES:
        print(f"Hashing {name}", flush=True)
        manifest["base_files"][name] = fingerprint(runtime / name)
    validate_manifest(manifest)
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing a previously tested artifact.
    with archive.open("xb") as output, tarfile.open(fileobj=output, mode="w") as tar:
        for name, data in [("manifest.json", encoded), ("install.py", Path(__file__).read_bytes())]:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo(ADDON)
        info.size, info.mode, info.mtime = manifest["payload"][ADDON]["size"], 0o644, 0
        with (runtime / ADDON).open("rb") as source:
            tar.addfile(info, source)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_bytes(encoded)
    return manifest


def read_manifest(tar: tarfile.TarFile) -> dict:
    members = tar.getmembers()
    expected = {"manifest.json", "install.py", ADDON}
    if len(members) != len(expected) or {m.name for m in members} != expected:
        raise ValueError("Unexpected or duplicate archive members")
    if any(not m.isfile() for m in members):
        raise ValueError("Only regular archive files are permitted")
    if tar.getmember("manifest.json").size > 1024 * 1024:
        raise ValueError("Oversized manifest")
    manifest = json.load(tar.extractfile("manifest.json"))
    validate_manifest(manifest)
    if tar.getmember(ADDON).size != manifest["payload"][ADDON]["size"]:
        raise ValueError("Payload size does not match manifest")
    return manifest


def install(archive: Path, runtime: Path, *, check_only: bool = False) -> dict:
    runtime = runtime.resolve(strict=True)
    with tarfile.open(archive, "r:") as tar:
        manifest = read_manifest(tar)
        verify_base(runtime, manifest)
        with tar.extractfile(ADDON) as source:
            install_stream(source, runtime, manifest, check_only=check_only)
    return manifest


def verify_base(runtime: Path, manifest: dict) -> None:
    validate_manifest(manifest)
    for name, expected in manifest["base_files"].items():
        path = runtime / name
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"Missing or incompatible base file: {name}")
        print(f"Verifying {name}", flush=True)
        if fingerprint(path) != expected:
            raise ValueError(f"Base SHA-256 mismatch: {name}")


def install_stream(source, runtime: Path, manifest: dict, *, check_only=False) -> None:
    destination = runtime / ADDON
    # Don't follow a directory link into someone else's installation.
    if not destination.parent.is_dir() or destination.parent.resolve() != destination.parent:
        raise ValueError("Paks must be a real directory inside the selected runtime")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".rt-safe-", delete=False) as out:
        temporary = Path(out.name)
        try:
            shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
            out.flush()
            if fingerprint(temporary) != manifest["payload"][ADDON]:
                raise ValueError("Add-on SHA-256 mismatch")
            if destination.exists() or destination.is_symlink():
                if fingerprint(destination) != manifest["payload"][ADDON]:
                    raise ValueError("A different add-on is installed; refusing to overwrite it")
            elif not check_only:
                temporary.chmod(0o644)
                os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def install_pak(pak: Path, runtime: Path, manifest: dict) -> None:
    runtime = runtime.resolve(strict=True)
    verify_base(runtime, manifest)
    if pak.stat().st_size != manifest["payload"][ADDON]["size"]:
        raise ValueError("Unexpected add-on file size")
    with pak.open("rb") as source:
        install_stream(source, runtime, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("build")
    pack.add_argument("--runtime", required=True, type=Path)
    pack.add_argument("--archive", required=True, type=Path)
    pack.add_argument("--manifest", required=True, type=Path)
    raw = commands.add_parser("install-pak", help="Install a downloaded chunk against the pinned repository manifest")
    raw.add_argument("--runtime", required=True, type=Path)
    raw.add_argument("--pak", required=True, type=Path)
    for command in ("check", "install"):
        sub = commands.add_parser(command)
        sub.add_argument("--runtime", required=True, type=Path)
        sub.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.runtime, args.archive, args.manifest)
    elif args.command == "install-pak":
        manifest_path = Path(__file__).resolve().parents[1] / "validation/realtime-addon-manifest.json"
        install_pak(args.pak, args.runtime, json.loads(manifest_path.read_text()))
    else:
        install(args.archive, args.runtime, check_only=args.command == "check")
    print(f"{args.command}: passed")


if __name__ == "__main__":
    main()
