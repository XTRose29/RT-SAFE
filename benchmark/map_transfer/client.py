"""Attach-only SPEAR transport for an explicitly owned native editor.

The SDK is supplied with the Unreal project and is not redistributed here.
Set PYTHONPATH to its Python package and compatible spear_ext build. The
same SDK's pure-Python package must be importable by the editor Python.
"""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import tempfile
import textwrap
import hashlib
import sys
from typing import Any


class NativeClient:
    def __init__(self, config_path: str | Path, *, expected_pid: int):
        import spear

        if isinstance(expected_pid, bool) or expected_pid <= 0:
            raise ValueError("Provide the PID of the editor you own")
        if Path(f"/proc/{expected_pid}").stat().st_uid != os.getuid():
            raise ValueError("Refusing to control another user's editor process")
        config = spear.get_config(user_config_files=[str(Path(config_path).resolve())])
        clock = config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE
        if (
            config.SP_SERVICES.ENGINE_SERVICE.STEPPING_MODE != "async"
            or clock.OVERRIDE_FIXED_DELTA_TIME
            or not clock.OVERRIDE_BENCHMARKING
            or clock.BENCHMARKING
        ):
            raise ValueError(
                "Native real-time pilot requires async stepping, benchmarking=false, and no fixed delta override"
            )
        config.defrost()
        config.SPEAR.LAUNCH_MODE = "none"
        config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = 60.0
        config.freeze()
        # One host controller per editor. OS locks are released even after a
        # host crash; a stale text PID file alone would not provide that safety.
        self._lock = (
            Path(tempfile.gettempdir())
            / f"rtsafe-native-{os.getuid()}-{expected_pid}.lock"
        ).open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError(
                "Another RT-SAFE controller already owns this editor connection"
            ) from None
        self.instance = None
        try:
            self._connect(spear, config, expected_pid)
            executable = Path(f"/proc/{expected_pid}/exe").resolve()
            names = [
                executable.name,
                "libSimWorldEditor-SimWorld.so",
                "libSimWorldEditor-SpServices.so",
                "SimWorldEditor.modules",
            ]
            hashes = {}
            for name in names:
                path = executable.parent / name
                if path.is_file():
                    with path.open("rb") as handle:
                        hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
            self.fingerprint = {
                "host_python": sys.version.split()[0],
                "spear_version": spear.__version__,
                "native_binary_sha256": hashes,
                "configuration_sha256": hashlib.sha256(
                    Path(config_path).read_bytes()
                ).hexdigest(),
            }
            modules = executable.parent / "SimWorldEditor.modules"
            if modules.is_file():
                self.fingerprint["unreal_build_id"] = json.loads(
                    modules.read_text()
                ).get("BuildId")
        except BaseException:
            self.close()
            raise

    def _connect(self, spear, config, expected_pid):
        self.instance = spear.Instance(config=config)
        self.game = None
        self.pool = None
        actual = int(self.instance.engine_globals_service.get_current_process_id())
        if actual != expected_pid:
            raise RuntimeError(
                f"Expected owned editor PID {expected_pid}, connected to {actual}"
            )
        self.pid = actual
        self.instance._engine_service.initialize()
        self.editor = self.instance._editor
        self.editor.set_world(world=self.instance._get_editor_world())
        with self.instance.begin_frame():
            self.instance._unreal_service.initialize()
            self.editor.initialize(unreal_service=self.instance._unreal_service)
        with self.instance.end_frame():
            pass
        # SDK v1.0.0 get_editor() waits for a global asset-registry scan that
        # can remain busy indefinitely in this content-heavy editor. This
        # adapter explicitly initializes the already-loaded world services.
        # It does NOT claim engine-wide idle: scene/asset readiness is checked
        # by the preparation and physical smoke tests before any rollout.

    def execute(
        self, source: str, *, outputs: tuple[str, ...] = ("result",)
    ) -> dict[str, Any]:
        if any(not name.isidentifier() for name in outputs):
            raise ValueError("Output names must be Python identifiers")
        # SPEAR requires Public scope for outputs. Keep request-local Unreal
        # wrappers inside a function, and export JSON values only. Retaining
        # PIE delegate wrappers in that public namespace can crash UE GC after
        # the associated world has been destroyed.
        body = (
            "import sys, json, unreal\n"
            'rt_transfer = sys.modules.get("benchmark.map_transfer.unreal_runtime")\n'
            + source
            + "\n"
            "return json.loads(json.dumps({"
            + ",".join(repr(name) + ":" + name for name in outputs)
            + "}))"
        )
        wrapped = (
            "def _rtsafe_rpc_request():\n"
            + textwrap.indent(body, "    ")
            + "\ntry:\n    _rtsafe_rpc_response = {'ok':True,'outputs':_rtsafe_rpc_request()}\n"
            + "except Exception:\n    import traceback\n    _rtsafe_rpc_response = {'ok':False,'error':traceback.format_exc()}\n"
            + "finally:\n    del _rtsafe_rpc_request\n"
        )
        with self.instance.begin_frame():
            value = self.editor.python_service.execute_string(
                string=wrapped,
                execution_scope="Public",
                outputs=["_rtsafe_rpc_response"],
            )
        with self.instance.end_frame():
            pass
        response = value["_rtsafe_rpc_response"]
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["outputs"]

    def call(self, function: str, **kwargs: Any) -> Any:
        if not function.isidentifier():
            raise ValueError("Expected a module function name")
        source = f"result = rt_transfer.{function}(**{kwargs!r})"
        return self.execute(source)["result"]

    def load_runtime(self, repo_root: str | Path) -> None:
        # Host and editor share a filesystem in this pilot. No arbitrary
        # remote input is evaluated; these are local repository sources.
        root = str(Path(repo_root).resolve())
        self.execute(
            "import sys\n"
            f"if {root!r} not in sys.path: sys.path.insert(0, {root!r})\n"
            "import benchmark.map_transfer.unreal_runtime as rt_transfer\n"
            'result = {"loaded": rt_transfer.__file__}'
        )

    def initialize_game(self) -> None:
        self.game = self.instance._game
        self.game.set_world(world=self.instance._get_game_world())
        with self.instance.begin_frame():
            self.game.initialize(unreal_service=self.instance._unreal_service)
            self.pool = self.game.unreal_service.get_subsystem(
                "UWorld",
                "USpCameraCapturePool",
                as_unreal_object=True,
                with_sp_funcs=False,
            )
        with self.instance.end_frame():
            pass
        if self.pool is None:
            raise RuntimeError("Native camera capture pool is unavailable")

    def capture(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.pool is None:
            self.initialize_game()
        with self.instance.begin_frame():
            value = self.pool.Agent_CaptureCamerasJson(RequestJson=json.dumps(request))
        with self.instance.end_frame():
            pass
        data = json.loads(value) if isinstance(value, str) else value
        if (
            data.get("missing_count")
            or data.get("error_count")
            or not data.get("images")
        ):
            raise RuntimeError(f'Native observation failed: {data.get("errors", data)}')
        return data

    def close(self) -> None:
        # LAUNCH_MODE=none releases this client without terminating Unreal.
        try:
            if self.instance is not None:
                self.instance.close()
                self.instance = None
        finally:
            if not self._lock.closed:
                self._lock.close()

    def __enter__(self) -> "NativeClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
