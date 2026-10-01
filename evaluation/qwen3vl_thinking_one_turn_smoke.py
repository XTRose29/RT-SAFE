#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def extract_section(text, title):
    pattern = (
        rf"=+\n{re.escape(title)}:\n=+\n"
        rf"(.*?)(?=\n=+\n[A-Z0-9 /()_-]+:\n=+\n|\Z)"
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find section: {title}")
    return match.group(1).strip()


def split_visible_thinking(text, assume_thinking_prefix=False):
    if not text:
        return None, text
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip(), (text[: match.start()] + text[match.end() :]).strip()
    end = re.search(r"\s*</think>", text, re.IGNORECASE)
    if end:
        return text[: end.start()].strip(), text[end.end() :].strip()
    start = re.search(r"<think>\s*", text, re.IGNORECASE)
    if start:
        return text[start.end() :].strip(), text[: start.start()].strip()
    if assume_thinking_prefix:
        return text.strip(), ""
    return None, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", required=True)
    parser.add_argument("--model-path", default="/tmp/qwen3-vl-8b-thinking-smoke")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--no-token-limit", action="store_true")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    step_dir = Path(args.step_dir)
    data_path = step_dir / "step_0000_decision_0001_data.txt"
    image_path = step_dir / "step_0000_decision_0001_input_frame_00.png"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_text = data_path.read_text()
    system_prompt = extract_section(data_text, "SYSTEM PROMPT")
    user_prompt = extract_section(data_text, "USER PROMPT (INPUT)")
    image = Image.open(image_path).convert("RGB")

    started_at = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    if not getattr(processor, "chat_template", None):
        processor.chat_template = processor.tokenizer.chat_template
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generate_started_at = time.time()
    generation_kwargs = {"do_sample": True} if args.sample else {"do_sample": False}
    if not args.no_token_limit:
        generation_kwargs["max_new_tokens"] = args.max_new_tokens
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    reasoning, model_content = split_visible_thinking(output_text, assume_thinking_prefix=True)

    result = {
        "model_path": args.model_path,
        "step_dir": str(step_dir),
        "image_path": str(image_path),
        "data_path": str(data_path),
        "load_and_generate_seconds": time.time() - started_at,
        "generate_seconds": time.time() - generate_started_at,
        "max_new_tokens": None if args.no_token_limit else args.max_new_tokens,
        "no_token_limit": args.no_token_limit,
        "sample": args.sample,
        "raw_output": output_text,
        "reasoning_content": reasoning,
        "model_content": model_content,
    }
    (out_dir / "qwen3vl_thinking_one_turn_response.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    (out_dir / "qwen3vl_thinking_one_turn_response.txt").write_text(
        "REASONING_CONTENT:\n"
        f"{reasoning or 'N/A'}\n\n"
        "MODEL_CONTENT:\n"
        f"{model_content or ''}\n\n"
        "RAW_OUTPUT:\n"
        f"{output_text}\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
