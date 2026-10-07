from unittest import mock

import pytest

pytest.importorskip("torch", reason="Optional local model serving stack")
pytest.importorskip("transformers", reason="Optional local model serving stack")
pytest.importorskip("fastapi", reason="Optional local model serving stack")
pytest.importorskip("uvicorn", reason="Optional local model serving stack")

from evaluation import qwen3vl_thinking_openai_server as server


def test_chat_request_preserves_sampling_seed():
    request = server.ChatRequest(model="qwen", messages=[], seed=42)

    assert server.generation_seed(request) == 42


def test_generation_seed_rejects_negative_values():
    request = server.ChatRequest(model="qwen", messages=[], seed=-1)

    with pytest.raises(ValueError, match="seed must be non-negative"):
        server.generation_seed(request)


def test_apply_generation_seed_resets_cpu_and_cuda_rngs():
    with (
        mock.patch.object(server.torch, "manual_seed") as manual_seed,
        mock.patch.object(server.torch.cuda, "is_available", return_value=True),
        mock.patch.object(server.torch.cuda, "manual_seed_all") as manual_seed_all,
    ):
        server.apply_generation_seed(7)

    manual_seed.assert_called_once_with(7)
    manual_seed_all.assert_called_once_with(7)


def test_apply_generation_seed_leaves_rng_untouched_when_absent():
    with (
        mock.patch.object(server.torch, "manual_seed") as manual_seed,
        mock.patch.object(server.torch.cuda, "manual_seed_all") as manual_seed_all,
    ):
        server.apply_generation_seed(None)

    manual_seed.assert_not_called()
    manual_seed_all.assert_not_called()
