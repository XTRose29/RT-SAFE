import time
import re
import os
import copy
import openai
from typing import Optional
from PIL import Image
from simworld.llm.base_llm import BaseLLM
from simworld.utils.logger import Logger
from utils.img2base64 import pil_to_base64, np_to_base64
from base.rt_action_space import RTActionSpace, MOVE_TO, TURN_AROUND, WAIT
from llm.codex_cli_backend import CodexCLIBackend
from llm.reasoning_capabilities import openrouter_reasoning_profile
from llm.claude_code_backend import ClaudeCodeBackend

# Normalize LLM output to canonical action names
ACTION_ALIASES = {
    'moveto': MOVE_TO, 'move_to': MOVE_TO,
    'turnaround': TURN_AROUND, 'turn_around': TURN_AROUND, 'turn': TURN_AROUND,
    'wait': WAIT,
}

# Models that support native reasoning mode (extra_body={"reasoning": {"enabled": True}})
# Match by substring, e.g. "openai/gpt-5.2" contains "gpt-5.2"
REASONING_MODELS = {
    'gpt-6',
    'gpt-5',
    'gpt-5.6',
    'gpt-5.5',
    'gpt-5.4',
    'gpt-5.2',
    'gpt-realtime-2.1',
    'o1', 'o3', 'o4',
    'gemini-3.1-pro', 'gemini-3.1-flash',
    'claude-haiku-4.5', 'claude-sonnet-4.6', 'claude-opus-4.6',
    'grok-4.1',
    'qwen3-max-thinking',
    'qwen3-vl-8b-thinking',
    'qwen3-vl-thinking',
    'glm-4.5v',
}


def _split_visible_thinking(response_text: Optional[str]):
    """Split inline Qwen-style <think>...</think> content from the visible answer."""
    if not response_text:
        return None, response_text

    match = re.search(r'<think>\s*(.*?)\s*</think>', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        model_content = (response_text[:match.start()] + response_text[match.end():]).strip()
        return thinking, model_content

    end_match = re.search(r'\s*</think>', response_text, re.IGNORECASE)
    if end_match:
        thinking = response_text[:end_match.start()].strip()
        model_content = response_text[end_match.end():].strip()
        return thinking, model_content

    start_match = re.search(r'<think>\s*', response_text, re.IGNORECASE)
    if start_match:
        thinking = response_text[start_match.end():].strip()
        model_content = response_text[:start_match.start()].strip()
        return thinking, model_content

    return None, response_text


def _model_supports_reasoning(model_name: str) -> bool:
    """Check if model supports native reasoning mode."""
    model_lower = model_name.lower()
    return (
        openrouter_reasoning_profile(model_name) is not None
        or any(pattern in model_lower for pattern in REASONING_MODELS)
    )


class RTLLM(BaseLLM):
    def __init__(self, model_name: str, url: Optional[str] = None, provider: Optional[str] = 'self-hosted',
                 reasoning: bool = False, reasoning_effort: Optional[str] = None,
                 max_tokens: Optional[int] = None, extra_body: Optional[dict] = None,
                 prompt_suffix: Optional[str] = None,
                 api_mode: str = 'chat_completions', image_detail: Optional[str] = None,
                 text_verbosity: Optional[str] = None, service_tier: Optional[str] = None,
                 store: bool = False, request_timeout: Optional[float] = None,
                 raise_on_api_error: bool = False,
                 temperature: float = 0.7, top_p: float = 1.0,
                 seed: Optional[int] = None):
        if provider == 'openrouter' and not url:
            url = 'https://openrouter.ai/api/v1'
        if provider in ['openai', 'openrouter']:
            super().__init__(model_name, url, provider)
        elif provider in ('codex-cli', 'claude-code'):
            self.client = None
        elif provider == 'dashscope':
            self.client = openai.OpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
        else:
            self.client = openai.OpenAI(api_key='empty', base_url=url)

        self.model_name = model_name
        self.provider = provider
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.prompt_suffix = prompt_suffix
        self.api_mode = api_mode
        self.image_detail = image_detail
        self.text_verbosity = text_verbosity
        self.service_tier = service_tier
        self.store = store
        self.request_timeout = request_timeout
        self.raise_on_api_error = raise_on_api_error
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = None if seed is None else int(seed)
        self.last_usage_metadata = {}
        if self.api_mode not in {'chat_completions', 'responses', 'realtime', 'codex_cli', 'claude_code'}:
            raise ValueError(f'Unsupported API mode: {self.api_mode}')
        if self.api_mode in {'responses', 'realtime'} and self.provider != 'openai':
            raise ValueError(
                f'The {self.api_mode} API mode currently requires provider="openai"'
            )
        if (self.provider == 'codex-cli') != (self.api_mode == 'codex_cli'):
            raise ValueError(
                'provider="codex-cli" and api_mode="codex_cli" must be selected together'
            )
        if (self.provider == 'claude-code') != (self.api_mode == 'claude_code'):
            raise ValueError(
                'provider="claude-code" and api_mode="claude_code" must be selected together'
            )
        self.codex_cli_backend = None
        if self.provider == 'codex-cli':
            self.codex_cli_backend = CodexCLIBackend(
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,
                timeout=self.request_timeout,
            )
        self.claude_code_backend = None
        if self.provider == 'claude-code':
            # Subscription-authenticated Claude Code CLI; never the Anthropic
            # API or OpenRouter. No output or thinking cap is imposed.
            self.claude_code_backend = ClaudeCodeBackend(
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,
                timeout=self.request_timeout,
            )
        self.logger = Logger.get_logger('RTLLM')
        self.logger.info(
            f'Initialized LLM client for model -- {model_name}, url -- {url or "default"}, '
            f'api_mode -- {api_mode}, reasoning -- {reasoning}, '
            f'reasoning_effort -- {reasoning_effort}, max_tokens -- {max_tokens}'
        )

    def _effective_reasoning_effort(self):
        if not _model_supports_reasoning(self.model_name):
            return None
        # ``default`` is a local sentinel meaning that the API field must be
        # omitted. GPT-5.6 currently resolves this omission to medium; it is
        # not an adaptive/auto effort mode.
        if self.reasoning and self.reasoning_effort == 'default':
            return None
        if self.reasoning:
            return self.reasoning_effort or 'medium'
        return 'none'

    def _generate_response_responses_api(self, system_prompt, user_prompt, images, max_tokens):
        user_content = [{'type': 'input_text', 'text': user_prompt}]
        for image in images:
            image_item = {
                'type': 'input_image',
                'image_url': self._image_data_url(image),
            }
            if self.image_detail:
                image_item['detail'] = self.image_detail
            user_content.append(image_item)

        create_kwargs = {
            'model': self.model_name,
            'instructions': system_prompt,
            'input': [{'role': 'user', 'content': user_content}],
            'store': self.store,
        }
        if max_tokens is not None:
            create_kwargs['max_output_tokens'] = max_tokens
        effort = self._effective_reasoning_effort()
        if effort is not None:
            create_kwargs['reasoning'] = {'effort': effort}
        if self.text_verbosity:
            create_kwargs['text'] = {'verbosity': self.text_verbosity}
        if self.service_tier:
            create_kwargs['service_tier'] = self.service_tier
        if self.extra_body:
            create_kwargs['extra_body'] = copy.deepcopy(self.extra_body)
        if self.request_timeout is not None:
            create_kwargs['timeout'] = self.request_timeout

        response = self.client.responses.create(**create_kwargs)
        response_text = response.output_text
        usage = getattr(response, 'usage', None)
        self.last_usage_metadata = self._extract_usage_metadata(usage)
        total_tokens = getattr(usage, 'total_tokens', None) if usage else None
        # ``completion_tokens`` is the repository's legacy cross-provider name.
        # For the Responses API it is the full billed output-token count,
        # including hidden reasoning tokens as well as visible response tokens.
        completion_tokens = getattr(usage, 'output_tokens', None) if usage else None
        output_details = getattr(usage, 'output_tokens_details', None) if usage else None
        reasoning_tokens = getattr(output_details, 'reasoning_tokens', None) if output_details else None
        if usage is not None and reasoning_tokens is None:
            reasoning_tokens = 0
        self.last_usage_metadata['reasoning_tokens'] = reasoning_tokens
        if usage:
            self.logger.info(f'Usage: {getattr(usage, "input_tokens", None)} input tokens')
            self.logger.info(f'Usage: {completion_tokens} output tokens')
            self.logger.info(f'Usage: {total_tokens} total tokens')
            if reasoning_tokens is not None:
                self.logger.info(f'Usage: {reasoning_tokens} reasoning tokens')
        return response_text, total_tokens, completion_tokens, reasoning_tokens

    def _generate_response_realtime_api(self, system_prompt, user_prompt, images, max_tokens):
        """Run one stateless text+image decision over the Realtime WebSocket API."""
        user_content = [{'type': 'input_text', 'text': user_prompt}]
        for image in images:
            image_item = {
                'type': 'input_image',
                'image_url': self._image_data_url(image),
            }
            if self.image_detail:
                image_item['detail'] = self.image_detail
            user_content.append(image_item)

        session = {
            'type': 'realtime',
            'instructions': system_prompt,
            'output_modalities': ['text'],
            'max_output_tokens': max_tokens if max_tokens is not None else 'inf',
        }
        effort = self._effective_reasoning_effort()
        if effort is not None:
            session['reasoning'] = {'effort': effort}

        receive_timeout = self.request_timeout or 300.0

        def receive(connection):
            raw = connection._connection.recv(timeout=receive_timeout)
            return connection.parse_event(raw)

        response = None
        with self.client.realtime.connect(
            model=self.model_name,
            max_retries=0,
            websocket_connection_options={
                'open_timeout': min(30.0, receive_timeout),
                'close_timeout': 5.0,
            },
        ) as connection:
            while True:
                event = receive(connection)
                if event.type == 'error':
                    raise RuntimeError(f'Realtime session creation failed: {event}')
                if event.type == 'session.created':
                    break

            connection.session.update(session=session)
            while True:
                event = receive(connection)
                if event.type == 'error':
                    raise RuntimeError(f'Realtime session update failed: {event}')
                if event.type == 'session.updated':
                    break

            connection.conversation.item.create(item={
                'type': 'message',
                'role': 'user',
                'content': user_content,
            })
            connection.response.create(response={'output_modalities': ['text']})
            while True:
                event = receive(connection)
                if event.type == 'error':
                    raise RuntimeError(f'Realtime response failed: {event}')
                if event.type == 'response.done':
                    response = event.response
                    break

        if response is None or response.status != 'completed':
            raise RuntimeError(
                f'Realtime response did not complete: '
                f'{getattr(response, "status", None)!r} '
                f'{getattr(response, "status_details", None)!r}'
            )

        response_parts = []
        for item in response.output or []:
            if getattr(item, 'type', None) != 'message':
                continue
            for content in getattr(item, 'content', None) or []:
                if getattr(content, 'type', None) == 'output_text':
                    text = getattr(content, 'text', None)
                    if text:
                        response_parts.append(text)
        response_text = '\n'.join(response_parts)
        usage = response.usage
        self.last_usage_metadata = self._extract_usage_metadata(usage)
        total_tokens = getattr(usage, 'total_tokens', None) if usage else None
        completion_tokens = getattr(usage, 'output_tokens', None) if usage else None
        output_details = getattr(usage, 'output_token_details', None) if usage else None
        reasoning_tokens = getattr(output_details, 'reasoning_tokens', None) if output_details else None
        if usage is not None and reasoning_tokens is None:
            reasoning_tokens = 0
        self.last_usage_metadata['reasoning_tokens'] = reasoning_tokens
        if usage:
            self.logger.info(f'Usage: {getattr(usage, "input_tokens", None)} input tokens')
            self.logger.info(f'Usage: {completion_tokens} output tokens')
            self.logger.info(f'Usage: {total_tokens} total tokens')
            if reasoning_tokens is not None:
                self.logger.info(f'Usage: {reasoning_tokens} reasoning tokens')
        return response_text, total_tokens, completion_tokens, reasoning_tokens

    def generate_response_openai(self, system_prompt, user_prompt, images=[], max_tokens=None, temperature=None, top_p=None):
        start_time = time.time()
        self.last_usage_metadata = {}
        request_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if self.api_mode in {'codex_cli', 'claude_code'}:
            backend = (
                self.codex_cli_backend
                if self.api_mode == 'codex_cli'
                else self.claude_code_backend
            )
            try:
                result = backend.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    images=images,
                )
                self.last_usage_metadata = dict(result.usage)
                action_space = self._parse_structured_response(result.text)
                return (
                    action_space,
                    time.time() - start_time,
                    len(result.text),
                    result.text,
                    result.usage['total_tokens'],
                    result.usage['completion_tokens'],
                    result.usage['reasoning_tokens'],
                )
            except Exception as e:
                self.logger.error(f'Error in {self.api_mode} generation: {e}')
                if self.raise_on_api_error:
                    raise
                return None, time.time() - start_time, 0, None, None, None, None
        if self.api_mode in {'responses', 'realtime'}:
            try:
                response_text, total_tokens, completion_tokens, reasoning_tokens = (
                    (self._generate_response_realtime_api if self.api_mode == 'realtime' else self._generate_response_responses_api)(
                        system_prompt, user_prompt, images, request_max_tokens
                    )
                )
                char_count = len(response_text) if response_text else 0
                if response_text:
                    action_space = self._parse_structured_response(response_text)
                    return (
                        action_space,
                        time.time() - start_time,
                        char_count,
                        response_text,
                        total_tokens,
                        completion_tokens,
                        reasoning_tokens,
                    )
                return None, time.time() - start_time, 0, response_text, total_tokens, completion_tokens, reasoning_tokens
            except Exception as e:
                self.logger.error(f'Error in {self.api_mode} API generation: {e}')
                if self.raise_on_api_error:
                    raise
                return None, time.time() - start_time, 0, None, None, None, None

        user_content = []
        user_content.append({'type': 'text', 'text': user_prompt})

        for image in images:
            image_url = {'url': self._image_data_url(image)}
            if self.image_detail:
                image_url['detail'] = self.image_detail
            user_content.append({
                'type': 'image_url',
                'image_url': image_url,
            })

        try:
            create_kwargs = dict(
                model=self.model_name,
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}],
            )
            if self.request_timeout is not None:
                create_kwargs['timeout'] = self.request_timeout
            if request_max_tokens is not None:
                if self.provider == 'openai' and _model_supports_reasoning(self.model_name):
                    create_kwargs['max_completion_tokens'] = request_max_tokens
                else:
                    create_kwargs['max_tokens'] = request_max_tokens
            uses_gateway_reasoning = (
                self.provider in {'openai', 'openrouter'}
                and _model_supports_reasoning(self.model_name)
            )
            if not uses_gateway_reasoning:
                create_kwargs['temperature'] = (
                    getattr(self, 'temperature', 0.7)
                    if temperature is None else temperature
                )
                create_kwargs['top_p'] = (
                    getattr(self, 'top_p', 1.0)
                    if top_p is None else top_p
                )
                sampling_seed = getattr(self, 'seed', None)
                if sampling_seed is not None:
                    create_kwargs['seed'] = int(sampling_seed)
            extra_body = copy.deepcopy(self.extra_body)
            model_lower = self.model_name.lower()
            if self.reasoning and 'qwen3' in model_lower and ('thinking' in model_lower or 'qwen3-vl' in model_lower):
                chat_template_kwargs = copy.deepcopy(extra_body.get('chat_template_kwargs') or {})
                chat_template_kwargs['enable_thinking'] = True
                extra_body['chat_template_kwargs'] = chat_template_kwargs
            if _model_supports_reasoning(self.model_name) and self.provider == 'openai':
                create_kwargs['reasoning_effort'] = self._effective_reasoning_effort()
            elif _model_supports_reasoning(self.model_name):
                # effort: "xhigh"|"high"|"medium"|"low"|"minimal"|"none"; explicit to override provider default
                effort = self._effective_reasoning_effort()
                if effort is not None:
                    extra_body['reasoning'] = {'effort': effort}
            if extra_body:
                create_kwargs['extra_body'] = extra_body

            response = self.client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            action_json = message.content
            reasoning_content = (
                getattr(message, 'reasoning_content', None)
                or getattr(message, 'reasoning', None)
            )
            visible_thinking, model_content = _split_visible_thinking(action_json)
            if not reasoning_content and visible_thinking:
                reasoning_content = visible_thinking
            parse_text = model_content if visible_thinking else action_json
            recorded_response = action_json
            if reasoning_content:
                recorded_response = (
                    "REASONING_CONTENT:\n"
                    f"{reasoning_content}\n\n"
                    "MODEL_CONTENT:\n"
                    f"{model_content or ''}"
                )

            total_tokens = None
            completion_tokens = None
            reasoning_tokens = None
            if response.usage:
                self.last_usage_metadata = self._extract_usage_metadata(response.usage)
                total_tokens = response.usage.total_tokens
                completion_tokens = response.usage.completion_tokens
                ctd = getattr(response.usage, 'completion_tokens_details', None)
                if ctd is not None:
                    reasoning_tokens = getattr(ctd, 'reasoning_tokens', None)
                if reasoning_tokens is None:
                    reasoning_tokens = getattr(response.usage, 'reasoning_tokens', None)
                self.logger.info(f'Usage: {response.usage.prompt_tokens} prompt tokens')
                self.logger.info(f'Usage: {response.usage.completion_tokens} completion tokens')
                self.logger.info(f'Usage: {response.usage.total_tokens} total tokens')
                if reasoning_tokens is not None:
                    self.logger.info(f'Usage: {reasoning_tokens} reasoning tokens')
            served_provider = getattr(response, 'provider', None)
            served_model = getattr(response, 'model', None)
            if served_provider is not None:
                self.last_usage_metadata['served_provider'] = served_provider
            if served_model is not None:
                self.last_usage_metadata['served_model'] = served_model

            char_count = len(recorded_response) if recorded_response else 0

            if parse_text:
                try:
                    action_space = self._parse_structured_response(parse_text)
                    return action_space, time.time() - start_time, char_count, recorded_response, total_tokens, completion_tokens, reasoning_tokens
                except Exception as parse_error:
                    self.logger.error(f'Error parsing structured response: {parse_error}')
                    return None, time.time() - start_time, char_count, recorded_response, total_tokens, completion_tokens, reasoning_tokens

            if recorded_response:
                return None, time.time() - start_time, char_count, recorded_response, total_tokens, completion_tokens, reasoning_tokens

        except Exception as e:
            self.logger.error(f'Error in generate_response_openai: {e}')
            if self.raise_on_api_error:
                raise
            return None, time.time() - start_time, 0, None, None, None, None

        return None, time.time() - start_time, 0, None, None, None, None

    @staticmethod
    def _extract_usage_metadata(usage):
        """Return serializable cross-provider token and charge metadata."""
        if usage is None:
            return {}

        completion_details = (
            getattr(usage, 'completion_tokens_details', None)
            or getattr(usage, 'output_tokens_details', None)
            or getattr(usage, 'output_token_details', None)
        )
        prompt_details = (
            getattr(usage, 'prompt_tokens_details', None)
            or getattr(usage, 'input_tokens_details', None)
            or getattr(usage, 'input_token_details', None)
        )
        cost_details = getattr(usage, 'cost_details', None)
        return {
            'prompt_tokens': getattr(
                usage, 'prompt_tokens', getattr(usage, 'input_tokens', None)
            ),
            'completion_tokens': getattr(
                usage, 'completion_tokens', getattr(usage, 'output_tokens', None)
            ),
            'total_tokens': getattr(usage, 'total_tokens', None),
            'reasoning_tokens': getattr(completion_details, 'reasoning_tokens', None),
            'cached_tokens': getattr(prompt_details, 'cached_tokens', None),
            'cache_write_tokens': getattr(prompt_details, 'cache_write_tokens', None),
            'cost_usd': getattr(usage, 'cost', None),
            'upstream_inference_cost_usd': getattr(
                cost_details, 'upstream_inference_cost', None
            ),
        }

    def _process_image(self, image):
        if isinstance(image, Image.Image):
            return pil_to_base64(image)
        else:
            return np_to_base64(image)

    def _image_data_url(self, image):
        """Encode an image with a MIME type matching its actual bytes."""
        mime_type = 'image/png' if isinstance(image, Image.Image) else 'image/jpeg'
        return f'data:{mime_type};base64,{self._process_image(image)}'

    def _parse_structured_response(self, response_text):
        """Parse structured text response into RTActionSpace object."""
        action_type = None
        action_param = None
        reasoning = None
        
        try:
            # Extract action type (Action: move_to / turn_around / wait)
            # Select the first *recognized* Action line. Some APIs echo the
            # bracketed output template before emitting the actual response;
            # treating that template echo as the action discards an otherwise
            # valid decision later in the same response.
            for action_match in re.finditer(
                r'^\s*Action:\s*([^\n\r]+)',
                response_text,
                re.IGNORECASE | re.MULTILINE,
            ):
                raw = (
                    action_match.group(1)
                    .strip()
                    .lower()
                    .replace(' ', '')
                    .replace('_', '')
                )
                action_type = ACTION_ALIASES.get(raw)
                if action_type:
                    break
            
            # Extract param (Param: 1-7 / L30 etc / 1-3)
            param_match = re.search(r'^\s*Param:\s*([^\n\r]+)', response_text, re.IGNORECASE | re.MULTILINE)
            if param_match:
                action_param = param_match.group(1).strip()
                if action_type == WAIT:
                    wait_param_match = re.search(r'\b([123])\b', action_param)
                    if wait_param_match:
                        action_param = wait_param_match.group(1)
                elif action_type == MOVE_TO:
                    move_param_match = re.search(r'\b([1-7])\b', action_param)
                    if move_param_match:
                        action_param = move_param_match.group(1)
                elif action_type == TURN_AROUND:
                    turn_param_match = re.search(r'\b([LR]\s*(?:30|60|90))\b', action_param, re.IGNORECASE)
                    if turn_param_match:
                        action_param = turn_param_match.group(1).upper().replace(' ', '')
            
            # Fallback: try Waypoint for move_to (param 1-7)
            if not action_param and action_type == MOVE_TO:
                wp_match = re.search(r'Waypoint:\s*([1-7])', response_text, re.IGNORECASE)
                if wp_match:
                    action_param = wp_match.group(1)

            # The move-only SFT model emits just the waypoint parameter. In
            # that dedicated output contract, a bare digit 1-7 means move_to.
            if not action_type and not action_param:
                bare_waypoint_match = re.fullmatch(r'\s*([1-7])\s*', response_text)
                if bare_waypoint_match:
                    action_type = MOVE_TO
                    action_param = bare_waypoint_match.group(1)

            # Never infer an executable action from free-form reasoning.  A
            # truncated response can mention a waypoint only to call it unsafe
            # (for example, "waypoint 6 is blocked; I must wait").  Treat
            # missing structured fields as invalid instead of executing a
            # contradicted or speculative movement.
            
            # Fallback: try Waypoint for turn_around (L30/R30 etc)
            if not action_param and action_type == TURN_AROUND:
                turn_match = re.search(r'Waypoint:\s*([LR][3690]+)', response_text, re.IGNORECASE)
                if turn_match:
                    action_param = turn_match.group(1).upper()
                    if '90' in action_param:
                        action_param = action_param[0] + '90'
                    elif '60' in action_param:
                        action_param = action_param[0] + '60'
                    elif '30' in action_param:
                        action_param = action_param[0] + '30'
            
            # Fallback: try Wait duration for wait
            if not action_param and action_type == WAIT:
                dur_match = re.search(r'Wait duration:\s*([123])', response_text, re.IGNORECASE)
                if dur_match:
                    action_param = dur_match.group(1)
            
            # Extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*([^\n\r]+)', response_text, re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            return RTActionSpace(
                action_type=action_type,
                action_param=action_param,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f'Error parsing structured response: {e}')
            return RTActionSpace()
