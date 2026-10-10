"""Distinguish the Vulkan device actually selected from enumerated candidates."""
import sys
from types import SimpleNamespace

import pytest

from tools import check_runtime_addon as checker
from tools.check_runtime_addon import check_renderer


@pytest.mark.parametrize("device", ["llvmpipe (LLVM 20.1.2, 256 bits)", "lavapipe"])
def test_selected_software_renderer_has_actionable_error(device):
    with pytest.raises(RuntimeError, match="CPU software rendering"):
        check_renderer(f"[0]LogVulkanRHI: Display: - DeviceName: {device}\n")


def test_software_candidate_does_not_reject_selected_hardware():
    check_renderer(
        "LogVulkanRHI: Checking device support (deviceName=llvmpipe, driverName=llvmpipe)\n"
        "LogVulkanRHI: Creating Vulkan Device using VkPhysicalDevice 0x123.\n"
        "LogVulkanRHI: Display: - DeviceName: NVIDIA RTX A5000\n"
    )


def test_candidate_line_alone_does_not_prove_software_selection():
    check_renderer("LogVulkanRHI: Checking device support (deviceName=llvmpipe)\n")


def test_last_created_device_is_used():
    check_renderer(
        "LogVulkanRHI: Display: - DeviceName: llvmpipe\n"
        "LogVulkanRHI: Display: - DeviceName: NVIDIA RTX A5000\n"
    )


def test_adapter_fallback_is_still_rejected():
    with pytest.raises(RuntimeError, match="verify the requested graphics adapter"):
        check_renderer("LogVulkanRHI: Falling back to first device\n")


def test_software_renderer_fails_before_client_or_probe_and_cleans_up(monkeypatch, tmp_path):
    class UnusedClient:
        def __new__(cls):
            pytest.fail("GPU preflight must run before creating the UnrealCV client")

    monkeypatch.setitem(sys.modules, "unrealcv", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "simworld.communicator.unrealcv",
                        SimpleNamespace(UnrealCV=UnusedClient))

    class FreePort:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(checker.socket, "socket", FreePort)
    process = SimpleNamespace(poll=lambda: None)

    def launch(command, **kwargs):
        kwargs["stdout"].write(
            "LoadMap(/Game/RealTimeBench/Maps/RT10)\n"
            "LogVulkanRHI: Display: - DeviceName: llvmpipe\n"
            "Start listening on 19091\n"
            "Engine is initialized. Leaving FEngineLoop::Init()\n"
        )
        kwargs["stdout"].flush()
        return process

    monkeypatch.setattr(checker.subprocess, "Popen", launch)
    stopped = []
    monkeypatch.setattr(checker, "stop_process_group", stopped.append)
    args = SimpleNamespace(port=19091, output=tmp_path, launcher=tmp_path / "SimWorld.sh",
                           gpu=0, startup_timeout=5)
    with pytest.raises(RuntimeError, match="CPU software rendering"):
        checker.run_level(args, "RT10")
    assert stopped == [process]
