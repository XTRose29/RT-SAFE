"""Ephemeral Codex CLI transport for separately labeled SimWorld pilots.

This backend intentionally does not emulate the OpenAI Responses API.  Every
decision is a fresh ``codex exec`` turn authenticated by the local Codex CLI.
It exists only for experiments whose manifests explicitly declare the added
Codex agent/runtime layer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SUPPORTED_REASONING_EFFORTS = {
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
}


@dataclass(frozen=True)
class CodexCLIResult:
    text: str
    usage: dict[str, int | None]


def _serializable_usage(events: Iterable[dict[str, Any]]) -> dict[str, int | None]:
    """Return the final Codex turn usage in the benchmark's token vocabulary."""
    usage: dict[str, Any] = {}
    for event in events:
        if event.get("type") == "turn.completed" and isinstance(
            event.get("usage"), dict
        ):
            usage = event["usage"]
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reasoning_tokens": int(usage.get("reasoning_output_tokens") or 0),
        "cached_tokens": int(usage.get("cached_input_tokens") or 0),
        "cache_write_tokens": int(usage.get("cache_write_input_tokens") or 0),
        "cost_usd": None,
    }


def _write_image(image: Any, path: Path) -> None:
    """Persist an observation using the same lossless/lossy split as RTLLM."""
    if isinstance(image, Image.Image):
        image.save(path.with_suffix(".png"), format="PNG")
        return

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    # Match ``utils.img2base64.np_to_base64``, which is used by the Responses
    # API path and relies on Pillow's default JPEG encoding parameters.
    Image.fromarray(array).save(path.with_suffix(".jpg"), format="JPEG")


class CodexCLIBackend:
    """Invoke one fresh, non-interactive Codex session per policy decision."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        timeout: float | None,
        executable: str = "codex",
    ) -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise ValueError(f"Codex CLI executable was not found: {executable}")
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS | {None, "default"}:
            raise ValueError(
                "Codex CLI reasoning effort must be one of "
                f"{sorted(SUPPORTED_REASONING_EFFORTS)} or default"
            )
        self.executable = resolved
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    @staticmethod
    def _prompt(system_prompt: str, user_prompt: str, image_count: int) -> str:
        chronology = (
            "The attached images are ordered oldest to newest; the final image is "
            "the current annotated observation."
            if image_count > 1
            else "The attached image is the current annotated observation."
        )
        return f"""You are the policy model inside a SimWorld navigation benchmark.
Do not call tools, inspect files, browse, or modify the workspace. {chronology}
Follow the benchmark instructions and answer the benchmark user input directly.

<benchmark_system>
{system_prompt}
</benchmark_system>

<benchmark_user>
{user_prompt}
</benchmark_user>
"""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: Iterable[Any],
    ) -> CodexCLIResult:
        retained_images = [image for image in images if image is not None]
        with tempfile.TemporaryDirectory(prefix="simworld_codex_cli_") as temporary:
            workdir = Path(temporary)
            image_paths: list[Path] = []
            for index, image in enumerate(retained_images):
                stem = workdir / f"observation_{index:03d}"
                _write_image(image, stem)
                path = stem.with_suffix(".png" if isinstance(image, Image.Image) else ".jpg")
                image_paths.append(path)

            final_path = workdir / "final_response.txt"
            command = [
                self.executable,
                "exec",
                "--json",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "-C",
                str(workdir),
                "-m",
                self.model,
            ]
            if self.reasoning_effort not in {None, "default"}:
                command.extend(
                    ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
                )
            if image_paths:
                command.extend(["-i", *map(str, image_paths)])
            command.extend(["-o", str(final_path), "-"])

            environment = os.environ.copy()
            # The backend must consume the signed-in ChatGPT/Codex entitlement,
            # never an inherited Platform or OpenRouter API key.
            environment.pop("OPENAI_API_KEY", None)
            environment.pop("OPENROUTER_API_KEY", None)
            completed = subprocess.run(
                command,
                input=self._prompt(system_prompt, user_prompt, len(image_paths)),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                check=False,
                env=environment,
            )

            events: list[dict[str, Any]] = []
            for line in completed.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
            if completed.returncode != 0:
                errors = [
                    str(event.get("message") or event.get("error") or "")
                    for event in events
                    if event.get("type") == "error"
                ]
                detail = next((item for item in reversed(errors) if item), "")
                if not detail:
                    detail = completed.stderr.strip().splitlines()[-1:] or [""]
                    detail = detail[0]
                raise RuntimeError(
                    f"Codex CLI exited with status {completed.returncode}: {detail}"
                )
            if not final_path.is_file():
                raise RuntimeError("Codex CLI completed without a final response file")
            text = final_path.read_text(encoding="utf-8").strip()
            if not text:
                raise RuntimeError("Codex CLI returned an empty final response")
            usage = _serializable_usage(events)
            if usage["total_tokens"] == 0:
                raise RuntimeError("Codex CLI completed without machine-readable usage")
            return CodexCLIResult(text=text, usage=usage)
