#!/usr/bin/env python3
import argparse
import base64
import io
import re
import threading
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


def decode_data_image(url: str) -> Image.Image:
    if "," in url:
        url = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")


def split_visible_thinking(text: str) -> tuple[str | None, str]:
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip(), (text[: match.start()] + text[match.end() :]).strip()
    end = re.search(r"\s*</think>", text, re.IGNORECASE)
    if end:
        return text[: end.start()].strip(), text[end.end() :].strip()
    start = re.search(r"<think>\s*", text, re.IGNORECASE)
    if start:
        return text[start.end() :].strip(), text[: start.start()].strip()
    return None, text


def normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    normalized = []
    for item in content or []:
        item_type = item.get("type")
        if item_type == "text":
            normalized.append({"type": "text", "text": item.get("text", "")})
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            normalized.append({"type": "image", "image": decode_data_image(url)})
        elif item_type == "image":
            image = item.get("image")
            if isinstance(image, Image.Image):
                normalized.append({"type": "image", "image": image})
            elif isinstance(image, str):
                normalized.append({"type": "image", "image": decode_data_image(image)})
    return normalized


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    extra_body: dict[str, Any] | None = None


def request_extra(request: ChatRequest) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    nested = request.extra_body or {}
    if isinstance(nested, dict):
        extra.update(nested)
    model_extra = getattr(request, "model_extra", None) or {}
    if isinstance(model_extra, dict):
        extra.update(model_extra)
    return extra


def int_extra(extra: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = extra.get(key, default)
    if value is None:
        return None
    return int(value)


def bool_extra(extra: dict[str, Any], key: str, default: bool = False) -> bool:
    value = extra.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def generation_sampling_kwargs(request: ChatRequest) -> dict[str, Any]:
    temperature = 0.7 if request.temperature is None else float(request.temperature)
    top_p = 1.0 if request.top_p is None else float(request.top_p)
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return {"do_sample": False}
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }


def generation_seed(request: ChatRequest) -> int | None:
    """Return the request seed using the same non-negative contract as vLLM."""
    if request.seed is None:
        return None
    seed = int(request.seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def apply_generation_seed(seed: int | None) -> None:
    """Seed the serialized generation request before its first token.

    The server holds a process-wide generation lock, so resetting the CPU and
    CUDA RNGs here makes the complete phase-1/forced-close/phase-2 sequence
    reproducible without allowing another request to consume RNG state in the
    middle of it.
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_template(processor: AutoProcessor, messages: list[dict[str, Any]], extra: dict[str, Any], device: str):
    chat_template_kwargs = extra.get("chat_template_kwargs") or {}
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **chat_template_kwargs,
        ).to(device)
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)


def with_input_ids(inputs: dict[str, torch.Tensor], input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    original_len = int(inputs["input_ids"].shape[-1])
    continued_len = int(input_ids.shape[-1])
    continued = dict(inputs)
    continued["input_ids"] = input_ids
    continued["attention_mask"] = torch.ones_like(input_ids, device=input_ids.device)
    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask"} or not torch.is_tensor(value):
            continue
        if value.ndim >= 2 and int(value.shape[-1]) == original_len:
            pad_shape = list(value.shape)
            pad_shape[-1] = continued_len - original_len
            pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            continued[key] = torch.cat([value, pad], dim=-1)
    return continued


def has_parseable_final_answer(text: str) -> bool:
    return bool(re.search(r"^\s*Action:\s*", text, re.IGNORECASE | re.MULTILINE)) and bool(
        re.search(r"^\s*Param:\s*", text, re.IGNORECASE | re.MULTILINE)
    )


def create_app(
    model_path: str,
    served_model_name: str,
    default_max_new_tokens: int,
    *,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    ensure_parseable_action: bool = False,
) -> FastAPI:
    app = FastAPI()
    lock = threading.Lock()

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    if not getattr(processor, "chat_template", None):
        processor.chat_template = processor.tokenizer.chat_template
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose at most one of load_in_4bit and load_in_8bit")
    if load_in_4bit:
        model_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    elif load_in_8bit:
        model_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            }
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if not (load_in_4bit or load_in_8bit):
        model = model.to(device)
    model.eval()

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.get("/v1/simworld/capabilities")
    def simworld_capabilities():
        return {
            "hard_thinking_budget": True,
            "reasoning_budget_mode": "force_close",
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        extra = request_extra(request)
        messages = [
            {"role": message.get("role", "user"), "content": normalize_content(message.get("content", ""))}
            for message in request.messages
        ]
        inputs = apply_template(processor, messages, extra, device)

        sampling_kwargs = generation_sampling_kwargs(request)
        sampling_seed = generation_seed(request)
        max_new_tokens = int(request.max_tokens or default_max_new_tokens)
        reasoning_budget_tokens = int_extra(extra, "reasoning_budget_tokens")
        answer_max_tokens = int_extra(extra, "reasoning_budget_answer_tokens", 512)
        force_close = bool_extra(extra, "reasoning_budget_force_close", True)
        forced_close_text = str(extra.get("reasoning_budget_forced_close_text", "</think>\n\n"))
        budget_mode = extra.get("reasoning_budget_mode", "none" if reasoning_budget_tokens is None else "force_close")

        with lock, torch.inference_mode():
            apply_generation_seed(sampling_seed)
            if reasoning_budget_tokens is None or budget_mode in {None, "none", "off"}:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **sampling_kwargs,
                )
                forced_close_tokens = 0
                phase1_new_tokens = int(generated_ids.shape[-1] - inputs.input_ids.shape[-1])
                phase2_new_tokens = 0
                force_closed = False
                if ensure_parseable_action:
                    phase1_text = processor.batch_decode(
                        generated_ids[:, inputs.input_ids.shape[-1] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    if not has_parseable_final_answer(phase1_text):
                        continued_inputs = with_input_ids(inputs, generated_ids)
                        continued_ids = model.generate(
                            **continued_inputs,
                            max_new_tokens=max_new_tokens,
                            **sampling_kwargs,
                        )
                        phase2_new_tokens = int(
                            continued_ids.shape[-1] - generated_ids.shape[-1]
                        )
                        generated_ids = continued_ids
            else:
                phase1_ids = model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget_tokens,
                    **sampling_kwargs,
                )
                phase1_new = phase1_ids[:, inputs.input_ids.shape[-1] :]
                phase1_text = processor.batch_decode(
                    phase1_new,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                _, phase1_model_content = split_visible_thinking(phase1_text)
                natural_close = bool(re.search(r"</think>", phase1_text, re.IGNORECASE))
                needs_final = not has_parseable_final_answer(phase1_model_content)
                continued_ids = phase1_ids
                forced_close_tokens = 0
                force_closed = False

                if needs_final:
                    if force_close and not natural_close:
                        close_ids = processor.tokenizer(
                            forced_close_text,
                            add_special_tokens=False,
                            return_tensors="pt",
                        ).input_ids.to(device)
                        continued_ids = torch.cat([continued_ids, close_ids], dim=-1)
                        forced_close_tokens = int(close_ids.shape[-1])
                        force_closed = True

                    phase2_inputs = with_input_ids(inputs, continued_ids)
                    phase2_ids = model.generate(
                        **phase2_inputs,
                        max_new_tokens=answer_max_tokens,
                        **sampling_kwargs,
                    )
                    generated_ids = phase2_ids
                    phase2_new_tokens = int(phase2_ids.shape[-1] - continued_ids.shape[-1])
                else:
                    generated_ids = phase1_ids
                    phase2_new_tokens = 0

                phase1_new_tokens = int(phase1_new.shape[-1])

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        reasoning, model_content = split_visible_thinking(output_text)
        completion_tokens = len(trimmed[0])
        prompt_tokens = len(inputs.input_ids[0])

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": len(processor.tokenizer.encode(reasoning)) if reasoning else 0,
                },
            },
            "local_debug": {
                "model_content": model_content,
                "reasoning_chars": len(reasoning or ""),
                "reasoning_budget_mode": budget_mode,
                "reasoning_budget_tokens": reasoning_budget_tokens,
                "reasoning_budget_answer_tokens": answer_max_tokens,
                "seed": sampling_seed,
                "force_closed_thinking": force_closed,
                "forced_close_tokens": forced_close_tokens,
                "phase1_new_tokens": phase1_new_tokens,
                "phase2_new_tokens": phase2_new_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/qwen3-vl-8b-thinking-smoke")
    parser.add_argument("--served-model-name", default="qwen3-vl-8b-thinking")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--default-max-new-tokens", type=int, default=32768)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load weights with bitsandbytes NF4 quantization for lower GPU memory use.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load weights with bitsandbytes int8 quantization.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit instead of silently loading the model on CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--ensure-parseable-action",
        action="store_true",
        help=(
            "If the first completion budget omits Action/Param, continue the same "
            "response for one additional budget instead of returning an invalid action."
        ),
    )
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")

    app = create_app(
        args.model_path,
        args.served_model_name,
        args.default_max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        ensure_parseable_action=args.ensure_parseable_action,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
