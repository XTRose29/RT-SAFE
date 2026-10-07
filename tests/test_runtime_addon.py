"""Exercise installation against a miniature base, including failure paths."""
import io
import json
from pathlib import Path
import tarfile

import pytest

from tools.runtime_addon import ADDON, BASE_FILES, build, install, install_pak


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for name in [*BASE_FILES, ADDON]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        if name != ADDON:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
    archive = tmp_path / "addon.tar"
    build(source, archive, tmp_path / "manifest.json")
    return source, target, archive


def test_check_install_and_reinstall(bundle):
    source, target, archive = bundle
    install(archive, target, check_only=True)
    assert not (target / ADDON).exists()
    install(archive, target)
    assert (target / ADDON).read_bytes() == (source / ADDON).read_bytes()
    install(archive, target)
    assert not list((target / ADDON).parent.glob(".rt-safe-*"))


def test_rejects_wrong_base_even_when_size_matches(bundle):
    _, target, archive = bundle
    executable = target / BASE_FILES[0]
    executable.write_bytes(b"x" * executable.stat().st_size)
    with pytest.raises(ValueError, match="Base SHA-256 mismatch"):
        install(archive, target)
    assert not (target / ADDON).exists()


def test_rejects_conflicting_install_without_overwrite(bundle):
    _, target, archive = bundle
    (target / ADDON).write_bytes(b"different build")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        install(archive, target)
    assert (target / ADDON).read_bytes() == b"different build"


def test_rejects_corruption_before_install(bundle):
    _, target, archive = bundle
    with tarfile.open(archive) as tar:
        offset = tar.getmember(ADDON).offset_data
    with archive.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"X")
    with pytest.raises(ValueError, match="Add-on SHA-256 mismatch"):
        install(archive, target)
    assert not (target / ADDON).exists()
    assert not list((target / ADDON).parent.glob(".rt-safe-*"))


@pytest.mark.parametrize("name", ["../../escape", "manifest.json"])
def test_rejects_extra_or_duplicate_members(bundle, name):
    _, target, archive = bundle
    with tarfile.open(archive, "a") as tar:
        info = tarfile.TarInfo(name)
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        install(archive, target)
    assert not (target / ADDON).exists()


def test_rejects_link_into_shared_installation(bundle, tmp_path):
    _, target, archive = bundle
    paks = (target / ADDON).parent
    shared = tmp_path / "shared-paks"
    paks.rename(shared)
    paks.symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        install(archive, target)
    assert not (shared / "pakchunk3002-Linux.pak").exists()


def test_configured_realtime_blueprints_exist_in_cooked_inventory():
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "validation/realtime-addon-inventory.json").read_text())
    cooked = {entry["path"] for entry in inventory["entries"]}
    assets = json.loads((root / "data/ue_assets.json").read_text())
    for name, definition in assets.items():
        path = definition.get("asset_path", "")
        if path.startswith("/Game/RealTimeBench/"):
            package = path.removeprefix("/Game/").split(".")[0]
            assert package + ".uasset" in cooked, name
            assert package + ".uexp" in cooked, name


def test_install_downloaded_pak_uses_same_base_and_payload_checks(bundle):
    source, target, archive = bundle
    manifest = json.loads((archive.parent / "manifest.json").read_text())
    install_pak(source / ADDON, target, manifest)
    assert (target / ADDON).read_bytes() == (source / ADDON).read_bytes()
