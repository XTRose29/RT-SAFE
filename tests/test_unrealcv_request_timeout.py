import importlib
import logging
import queue

import pytest

unrealcv_module = importlib.import_module('simworld.communicator.unrealcv')
UnrealCV = unrealcv_module.UnrealCV


class _FakeClient:
    def __init__(self, endpoint=('127.0.0.1', 9011)):
        self.endpoint = endpoint
        self.send_message_id = 7
        self.recv_message_id = 7
        self.recv_num_q = queue.Queue()
        self.recv_data_q = queue.Queue()
        self.sent = []
        self.disconnected = False

    def request(self, message, timeout=5):
        raise AssertionError('the unbounded upstream request must be replaced')

    def request_batch(self, batch):
        raise AssertionError('the unbounded upstream batch request must be replaced')

    def send(self, message):
        self.sent.append(message)
        return True

    def disconnect(self):
        self.disconnected = True


def _wrapper(timeout_seconds=0.01):
    wrapper = UnrealCV.__new__(UnrealCV)
    wrapper.client = _FakeClient()
    wrapper.logger = logging.getLogger('test-unrealcv-timeout')
    wrapper.request_timeout_seconds = timeout_seconds
    wrapper.request_reconnect_retries = 0
    wrapper._install_bounded_request()
    return wrapper


def test_bounded_request_returns_matching_response():
    wrapper = _wrapper()
    wrapper.client.recv_data_q.put('ok')

    assert wrapper.client.request('vget /objects') == 'ok'
    assert wrapper.client.sent == [b'7:vget /objects']
    assert wrapper.client.recv_num_q.get_nowait() == -1
    assert wrapper.client.disconnected is False


def test_bounded_request_disconnects_and_raises_on_missing_response():
    wrapper = _wrapper()

    with pytest.raises(TimeoutError, match='vset /object/road/rotation'):
        wrapper.client.request('vset /object/road/rotation 0 180 0')

    assert wrapper.client.disconnected is True


def test_only_idempotent_commands_are_reconnect_retryable():
    assert UnrealCV._is_idempotent_retry_command('vget /objects')
    assert UnrealCV._is_idempotent_retry_command('vset /action/game/resume')
    assert UnrealCV._is_idempotent_retry_command(
        'vset /object/RT_TRAFFIC_SIGNAL_16/scale 1 1 1'
    )
    assert not UnrealCV._is_idempotent_retry_command(
        'vset /objects/spawn pedestrian pedestrian_1'
    )
    assert not UnrealCV._is_idempotent_retry_command('vbp actor MutatingCall')


def test_bounded_request_reconnects_once_for_safe_command(monkeypatch):
    wrapper = _wrapper()
    wrapper.request_reconnect_retries = 1
    replacement = _FakeClient()
    replacement.send_message_id = 0
    replacement.recv_data_q.put('ok')
    replacement.connect = lambda: True
    monkeypatch.setattr(
        unrealcv_module.unrealcv,
        'Client',
        lambda endpoint: replacement,
    )

    assert wrapper.client.request('vset /action/game/resume') == 'ok'
    assert wrapper.client is replacement
    assert wrapper.client.sent == [b'8:vset /action/game/resume']
    assert wrapper.client.recv_message_id == 8


def test_bounded_request_does_not_replay_spawn(monkeypatch):
    wrapper = _wrapper()
    wrapper.request_reconnect_retries = 1
    constructed = []
    monkeypatch.setattr(
        unrealcv_module.unrealcv,
        'Client',
        lambda endpoint: constructed.append(endpoint),
    )

    with pytest.raises(TimeoutError):
        wrapper.client.request('vset /objects/spawn pedestrian pedestrian_1')

    assert constructed == []


def test_bounded_batch_returns_all_matching_responses():
    wrapper = _wrapper()
    wrapper.client.recv_data_q.put('first')
    wrapper.client.recv_data_q.put('second')

    assert wrapper.client.request_batch(
        ['vget /object/a/location', 'vget /object/b/location']
    ) == ['first', 'second']
    assert wrapper.client.sent == [
        b'7:vget /object/a/location',
        b'8:vget /object/b/location',
    ]
    assert wrapper.client.recv_num_q.get_nowait() == -2


def test_list_request_uses_bounded_batch_timeout():
    wrapper = _wrapper()

    with pytest.raises(TimeoutError, match='batch request timed out'):
        wrapper.client.request(
            ['vget /object/a/location', 'vget /object/b/location']
        )

    assert wrapper.client.disconnected is True


def test_bounded_batch_reconnects_once_when_all_commands_are_safe(monkeypatch):
    wrapper = _wrapper()
    wrapper.request_reconnect_retries = 1
    replacement = _FakeClient()
    replacement.send_message_id = 0
    replacement.recv_data_q.put('first')
    replacement.recv_data_q.put('second')
    replacement.connect = lambda: True
    monkeypatch.setattr(
        unrealcv_module.unrealcv,
        'Client',
        lambda endpoint: replacement,
    )

    assert wrapper.client.request_batch(
        ['vget /object/a/location', 'vget /object/b/location']
    ) == ['first', 'second']
    assert wrapper.client is replacement
    assert wrapper.client.sent == [
        b'9:vget /object/a/location',
        b'10:vget /object/b/location',
    ]
    assert wrapper.client.recv_message_id == 9


def test_bounded_batch_does_not_replay_when_any_command_is_unsafe(monkeypatch):
    wrapper = _wrapper()
    wrapper.request_reconnect_retries = 1
    constructed = []
    monkeypatch.setattr(
        unrealcv_module.unrealcv,
        'Client',
        lambda endpoint: constructed.append(endpoint),
    )

    with pytest.raises(TimeoutError):
        wrapper.client.request_batch(
            ['vget /objects', 'vbp actor MutatingCall']
        )

    assert constructed == []
