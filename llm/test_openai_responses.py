from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


_simworld = types.ModuleType("simworld")
_simworld.__path__ = [str(Path(__file__).resolve().parents[1] / "SimWorld" / "simworld")]
sys.modules.setdefault("simworld", _simworld)

from llm.rt_llm import RTLLM, _model_supports_reasoning  # noqa: E402
from llm.reasoning_capabilities import (  # noqa: E402
    OPENROUTER_REASONING_PROFILES,
    openrouter_reasoning_profile,
)


class FakeLogger:
    def info(self, _message):
        pass

    def error(self, _message):
        pass


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text="Action: move_to\nParam: 5\nReasoning: Safe forward progress.",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                output_tokens_details=SimpleNamespace(reasoning_tokens=4),
            ),
        )


class FakeChatCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="Action: wait\nParam: 1\nReasoning: Hold position."
            ))],
            usage=None,
        )


class FakeOpenRouterChatCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        effort = kwargs.get("extra_body", {}).get("reasoning", {}).get("effort")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="Action: move_to\nParam: 5",
                reasoning="Brief summarized thinking." if effort == "low" else None,
            ))],
            provider="provider-under-test",
            model="model-snapshot-under-test",
            usage=SimpleNamespace(
                prompt_tokens=101,
                completion_tokens=25,
                total_tokens=126,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=7, cache_write_tokens=3
                ),
                cost=0.0012,
                cost_details=SimpleNamespace(upstream_inference_cost=0.0011),
            ),
        )

class OpenAIResponsesAdapterTest(unittest.TestCase):
    def test_target_openrouter_models_are_registered_as_reasoning_models(self):
        self.assertEqual(len(OPENROUTER_REASONING_PROFILES), 8)
        for model, profile in OPENROUTER_REASONING_PROFILES.items():
            with self.subTest(model=model):
                self.assertTrue(_model_supports_reasoning(model))
                self.assertEqual(openrouter_reasoning_profile(model), profile)

    def test_action_only_response_has_no_reasoning(self):
        llm = object.__new__(RTLLM)

        action = llm._parse_structured_response("Action: move_to\nParam: 2")

        self.assertEqual((action.action_type, action.action_param), ("move_to", "2"))
        self.assertIsNone(action.reasoning)
        self.assertEqual(str(action), "Action: move_to, Param: 2")

    def test_template_echo_before_valid_action_is_ignored(self):
        llm = object.__new__(RTLLM)
        response = (
            "Action: [move_to] Param: 2 Reasoning: template echo\n"
            "Action: move_to\n"
            "Param: 2\n"
            "Reasoning: Safe forward progress."
        )

        action = llm._parse_structured_response(response)

        self.assertEqual((action.action_type, action.action_param), ("move_to", "2"))

    def test_multimodal_response_is_parsed_and_usage_is_returned(self):
        llm = object.__new__(RTLLM)
        llm.model_name = "gpt-5.6-terra"
        llm.provider = "openai"
        llm.reasoning = True
        llm.reasoning_effort = "low"
        llm.max_tokens = 512
        llm.extra_body = {}
        llm.api_mode = "responses"
        llm.image_detail = "low"
        llm.text_verbosity = "low"
        llm.service_tier = None
        llm.store = False
        llm.request_timeout = 30.0
        llm.logger = FakeLogger()
        fake_responses = FakeResponses()
        llm.client = SimpleNamespace(responses=fake_responses)

        action, _, _, raw, total, output, reasoning = llm.generate_response_openai(
            system_prompt="Navigate safely.",
            user_prompt="Choose an action.",
            images=[Image.new("RGB", (8, 8))],
        )

        self.assertEqual((action.action_type, action.action_param), ("move_to", "5"))
        self.assertIn("Action: move_to", raw)
        self.assertEqual((total, output, reasoning), (120, 20, 4))
        request = fake_responses.kwargs
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["max_output_tokens"], 512)
        self.assertEqual(request["input"][0]["content"][1]["type"], "input_image")
        self.assertEqual(request["input"][0]["content"][1]["detail"], "low")
        self.assertNotIn("temperature", request)

    def test_responses_request_omits_output_cap_but_keeps_usage(self):
        llm = object.__new__(RTLLM)
        llm.model_name = "gpt-5.6-luna"
        llm.provider = "openai"
        llm.reasoning = True
        llm.reasoning_effort = "low"
        llm.max_tokens = None
        llm.extra_body = {}
        llm.api_mode = "responses"
        llm.image_detail = "low"
        llm.text_verbosity = "low"
        llm.service_tier = None
        llm.store = False
        llm.request_timeout = 30.0
        llm.logger = FakeLogger()
        fake_responses = FakeResponses()
        llm.client = SimpleNamespace(responses=fake_responses)

        _, _, _, _, total, output, reasoning = llm.generate_response_openai(
            "Navigate safely.", "Choose an action.", images=[]
        )

        self.assertNotIn("max_output_tokens", fake_responses.kwargs)
        self.assertEqual((total, output, reasoning), (120, 20, 4))

    def test_default_effort_omits_reasoning_field_and_keeps_usage(self):
        llm = object.__new__(RTLLM)
        llm.model_name = "gpt-5.6-luna"
        llm.provider = "openai"
        llm.reasoning = True
        llm.reasoning_effort = "default"
        llm.max_tokens = None
        llm.extra_body = {}
        llm.api_mode = "responses"
        llm.image_detail = "low"
        llm.text_verbosity = "low"
        llm.service_tier = None
        llm.store = False
        llm.request_timeout = 30.0
        llm.logger = FakeLogger()
        fake_responses = FakeResponses()
        llm.client = SimpleNamespace(responses=fake_responses)

        _, _, _, _, total, output, reasoning = llm.generate_response_openai(
            "Navigate safely.", "Choose an action.", images=[]
        )

        self.assertNotIn("reasoning", fake_responses.kwargs)
        self.assertEqual((total, output, reasoning), (120, 20, 4))

    def test_openrouter_claude_none_and_low_multimodal_contracts(self):
        for effort, reasoning_enabled, max_tokens in (
            ("none", False, 128),
            ("low", True, 1280),
        ):
            with self.subTest(effort=effort):
                llm = object.__new__(RTLLM)
                llm.model_name = "anthropic/claude-haiku-4.5"
                llm.provider = "openrouter"
                llm.reasoning = reasoning_enabled
                llm.reasoning_effort = effort
                llm.max_tokens = max_tokens
                llm.extra_body = {}
                llm.api_mode = "chat_completions"
                llm.image_detail = "low"
                llm.text_verbosity = "low"
                llm.service_tier = None
                llm.store = False
                llm.request_timeout = 30.0
                llm.temperature = 0.7
                llm.top_p = 1.0
                llm.seed = 0
                llm.logger = FakeLogger()
                fake_chat = FakeOpenRouterChatCompletions()
                llm.client = SimpleNamespace(
                    chat=SimpleNamespace(completions=fake_chat)
                )

                action, _, _, raw, total, output, reasoning = (
                    llm.generate_response_openai(
                        "Navigate safely.",
                        "Choose an action.",
                        images=[Image.new("RGB", (8, 8))],
                    )
                )

                self.assertEqual(
                    (action.action_type, action.action_param), ("move_to", "5")
                )
                self.assertEqual((total, output, reasoning), (126, 25, 12))
                self.assertEqual(
                    fake_chat.kwargs["extra_body"]["reasoning"]["effort"], effort
                )
                self.assertEqual(fake_chat.kwargs["max_tokens"], max_tokens)
                self.assertEqual(
                    fake_chat.kwargs["messages"][1]["content"][1]["type"],
                    "image_url",
                )
                self.assertEqual(
                    fake_chat.kwargs["messages"][1]["content"][1]["image_url"]["detail"],
                    "low",
                )
                self.assertNotIn("temperature", fake_chat.kwargs)
                self.assertNotIn("top_p", fake_chat.kwargs)
                self.assertEqual(llm.last_usage_metadata["cost_usd"], 0.0012)
                self.assertEqual(llm.last_usage_metadata["cached_tokens"], 7)
                self.assertEqual(
                    llm.last_usage_metadata["served_provider"],
                    "provider-under-test",
                )
                self.assertEqual(
                    llm.last_usage_metadata["served_model"],
                    "model-snapshot-under-test",
                )
                if effort == "low":
                    self.assertIn("REASONING_CONTENT", raw)

    def test_openrouter_default_effort_is_omitted_for_registered_models(self):
        llm = object.__new__(RTLLM)
        llm.model_name = "google/gemini-3.8-flash"
        llm.provider = "openrouter"
        llm.reasoning = True
        llm.reasoning_effort = "default"
        llm.max_tokens = 128
        llm.extra_body = {}
        llm.api_mode = "chat_completions"
        llm.image_detail = "low"
        llm.text_verbosity = "low"
        llm.service_tier = None
        llm.store = False
        llm.request_timeout = 30.0
        llm.temperature = 0.7
        llm.top_p = 1.0
        llm.seed = 0
        llm.logger = FakeLogger()
        fake_chat = FakeOpenRouterChatCompletions()
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake_chat))

        llm.generate_response_openai("system", "user", images=[])

        self.assertNotIn("reasoning", fake_chat.kwargs.get("extra_body", {}))
        self.assertNotIn("temperature", fake_chat.kwargs)
        self.assertNotIn("top_p", fake_chat.kwargs)

    def test_realtime_usage_schema_records_reasoning_and_cache_tokens(self):
        usage = SimpleNamespace(
            input_tokens=95,
            output_tokens=57,
            total_tokens=152,
            input_token_details=SimpleNamespace(cached_tokens=4),
            output_token_details=SimpleNamespace(reasoning_tokens=30),
        )

        metadata = RTLLM._extract_usage_metadata(usage)

        self.assertEqual(metadata["prompt_tokens"], 95)
        self.assertEqual(metadata["completion_tokens"], 57)
        self.assertEqual(metadata["reasoning_tokens"], 30)
        self.assertEqual(metadata["cached_tokens"], 4)

    def test_self_hosted_chat_completions_behavior_is_preserved(self):
        llm = object.__new__(RTLLM)
        llm.model_name = "Qwen/Qwen3-VL-8B-Instruct"
        llm.provider = "self-hosted"
        llm.reasoning = False
        llm.reasoning_effort = None
        llm.max_tokens = 256
        llm.extra_body = {}
        llm.api_mode = "chat_completions"
        llm.image_detail = None
        llm.text_verbosity = None
        llm.service_tier = None
        llm.store = False
        llm.request_timeout = None
        llm.temperature = 0.7
        llm.top_p = 1.0
        llm.seed = 42
        llm.logger = FakeLogger()
        fake_chat = FakeChatCompletions()
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake_chat))

        action, *_ = llm.generate_response_openai("system", "user", images=[])

        self.assertEqual((action.action_type, action.action_param), ("wait", "1"))
        self.assertEqual(fake_chat.kwargs["max_tokens"], 256)
        self.assertEqual(fake_chat.kwargs["temperature"], 0.7)
        self.assertEqual(fake_chat.kwargs["top_p"], 1.0)
        self.assertEqual(fake_chat.kwargs["seed"], 42)
        self.assertNotIn("reasoning_effort", fake_chat.kwargs)


if __name__ == "__main__":
    unittest.main()
