from unittest import mock

import openai
import pytest

from simworld.llm.retry import retry_api_call


class FakeClient:
    def __init__(self, *, strict):
        self.raise_on_api_error = strict
        self.calls = 0

    @retry_api_call(max_retries=2, initial_delay=1, rate_limit_per_min=20)
    def request(self):
        self.calls += 1
        raise openai.APIError("server failed", request=None, body=None)


def test_strict_benchmark_call_fails_immediately_without_throttle():
    client = FakeClient(strict=True)

    with mock.patch("simworld.llm.retry.time.sleep") as sleep:
        with pytest.raises(openai.APIError, match="server failed"):
            client.request()

    assert client.calls == 1
    sleep.assert_not_called()


def test_non_strict_client_retains_rate_limit_and_retries():
    client = FakeClient(strict=False)

    with mock.patch("simworld.llm.retry.time.sleep") as sleep:
        with pytest.raises(openai.APIError, match="server failed"):
            client.request()

    assert client.calls == 3
    assert sleep.call_args_list == [
        mock.call(3.0),
        mock.call(1),
        mock.call(3.0),
        mock.call(2),
        mock.call(3.0),
    ]
