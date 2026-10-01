import os
import logging
import queue
import sys
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SimWorld"))

from simworld.communicator.unrealcv import UnrealCV


def make_unrealcv(responses=None):
    unrealcv = UnrealCV.__new__(UnrealCV)
    unrealcv.client = Mock()
    unrealcv.client.request.side_effect = responses
    unrealcv.lock = Lock()
    unrealcv._tick_interval = 0.05
    return unrealcv


class FakeQueueClient:
    def __init__(self, endpoint=("127.0.0.1", 9011)):
        self.endpoint = endpoint
        self.send_message_id = 7
        self.recv_message_id = 7
        self.recv_num_q = queue.Queue()
        self.recv_data_q = queue.Queue()
        self.sent = []
        self.disconnected = False

    def request(self, message, timeout=5):
        raise AssertionError("unbounded upstream request must be replaced")

    def request_batch(self, batch):
        raise AssertionError("unbounded upstream batch request must be replaced")

    def send(self, message):
        self.sent.append(message)
        return True

    def disconnect(self):
        self.disconnected = True


def make_bounded_unrealcv(timeout_seconds=0.01):
    wrapper = UnrealCV.__new__(UnrealCV)
    wrapper.client = FakeQueueClient()
    wrapper.logger = logging.getLogger("test-unrealcv-timeout")
    wrapper.request_timeout_seconds = timeout_seconds
    wrapper.request_reconnect_retries = 0
    wrapper._install_bounded_requests()
    return wrapper


class RequestMetricsRecorder:
    def __init__(self):
        self.calls = []

    def record(self, elapsed, failed):
        self.calls.append((elapsed, failed))


class UnrealCVTimeControlTests(unittest.TestCase):
    def test_explicit_pause_and_resume_use_supported_commands(self):
        unrealcv = make_unrealcv(["paused", "resumed"])

        self.assertEqual("paused", unrealcv.pause_simulation())
        self.assertEqual("resumed", unrealcv.resume_simulation())
        self.assertEqual(
            [
                unittest.mock.call("vset /action/game/pause"),
                unittest.mock.call("vset /action/game/resume"),
            ],
            unrealcv.client.request.call_args_list,
        )

    def test_bounded_request_returns_response(self):
        wrapper = make_bounded_unrealcv()
        wrapper.client.recv_data_q.put("ok")

        self.assertEqual("ok", wrapper.client.request("vget /objects"))
        self.assertEqual([b"7:vget /objects"], wrapper.client.sent)
        self.assertEqual(-1, wrapper.client.recv_num_q.get_nowait())
        self.assertFalse(wrapper.client.disconnected)

    def test_bounded_request_disconnects_and_raises_on_timeout(self):
        wrapper = make_bounded_unrealcv()

        with self.assertRaisesRegex(TimeoutError, "rotation"):
            wrapper.client.request("vset /object/road/rotation 0 180 0")

        self.assertTrue(wrapper.client.disconnected)

    def test_camera_transport_failure_is_not_replaced_by_black_frame(self):
        unrealcv = make_unrealcv([TimeoutError("lost camera response")])

        with self.assertRaisesRegex(RuntimeError, "Camera capture failed"):
            unrealcv.get_image(2, "lit")

    def test_bounded_batch_has_one_total_deadline(self):
        wrapper = make_bounded_unrealcv()
        wrapper.client.recv_data_q.put("1 2 3")
        wrapper.client.recv_data_q.put("4 5 6")

        response = wrapper.client.request_batch(
            [
                "vget /object/1/location",
                "vget /object/2/location",
            ]
        )

        self.assertEqual(["1 2 3", "4 5 6"], response)
        self.assertEqual(
            [
                b"7:vget /object/1/location",
                b"8:vget /object/2/location",
            ],
            wrapper.client.sent,
        )
        self.assertEqual(-2, wrapper.client.recv_num_q.get_nowait())

    def test_bounded_wrapper_preserves_runner_request_metrics(self):
        recorder = RequestMetricsRecorder()
        original_request = FakeQueueClient.request
        original_request._simworld_request_metrics = recorder
        try:
            wrapper = make_bounded_unrealcv()
            wrapper.client.recv_data_q.put("ok")
            wrapper.client.recv_data_q.put("first")
            wrapper.client.recv_data_q.put("second")

            self.assertEqual("ok", wrapper.client.request("vget /objects"))
            self.assertEqual(
                ["first", "second"],
                wrapper.client.request_batch(
                    ["vget /object/1/location", "vget /object/2/location"]
                ),
            )
        finally:
            del original_request._simworld_request_metrics

        self.assertEqual([failed for _, failed in recorder.calls], [False, False])

    def test_bounded_wrapper_records_failed_request(self):
        recorder = RequestMetricsRecorder()
        original_request = FakeQueueClient.request
        original_request._simworld_request_metrics = recorder
        try:
            wrapper = make_bounded_unrealcv()
            with self.assertRaises(TimeoutError):
                wrapper.client.request("vget /objects")
        finally:
            del original_request._simworld_request_metrics

        self.assertEqual(len(recorder.calls), 1)
        self.assertTrue(recorder.calls[0][1])

    def test_only_read_commands_are_retryable(self):
        self.assertTrue(UnrealCV._is_read_only_retry_command("vget /objects"))
        self.assertTrue(
            UnrealCV._is_read_only_retry_command("vbp light GetState")
        )
        self.assertFalse(
            UnrealCV._is_read_only_retry_command(
                "vset /object/light/scale 1 1 1"
            )
        )
        self.assertFalse(
            UnrealCV._is_read_only_retry_command(
                "vset /action/game/resume"
            )
        )
        self.assertFalse(
            UnrealCV._is_read_only_retry_command(
                "vset /objects/spawn pedestrian pedestrian_1"
            )
        )
        self.assertFalse(
            UnrealCV._is_read_only_retry_command("vbp actor MutatingCall")
        )

    @patch("simworld.communicator.unrealcv.unrealcv.Client")
    def test_bounded_request_reconnects_once_for_read(self, client_class):
        wrapper = make_bounded_unrealcv()
        wrapper.request_reconnect_retries = 1
        replacement = FakeQueueClient()
        replacement.send_message_id = 0
        replacement.recv_data_q.put("ok")
        replacement.connect = Mock(return_value=True)
        client_class.return_value = replacement

        self.assertEqual("ok", wrapper.client.request("vget /objects"))

        self.assertIs(wrapper.client, replacement)
        self.assertEqual([b"8:vget /objects"], replacement.sent)
        self.assertEqual(8, replacement.recv_message_id)
        client_class.assert_called_once_with(("127.0.0.1", 9011))

    @patch("simworld.communicator.unrealcv.unrealcv.Client")
    def test_bounded_request_never_replays_spawn(self, client_class):
        wrapper = make_bounded_unrealcv()
        wrapper.request_reconnect_retries = 1

        with self.assertRaises(TimeoutError):
            wrapper.client.request(
                "vset /objects/spawn pedestrian pedestrian_1"
            )

        client_class.assert_not_called()

    def test_sync_mode_uses_supported_pause_command_only(self):
        unrealcv = make_unrealcv(["ok"])

        unrealcv.set_mode("sync", tick_interval=2.5)

        self.assertEqual(2.5, unrealcv._tick_interval)
        unrealcv.client.request.assert_called_once_with("vset /action/game/pause")

    @patch.dict(
        os.environ,
        {"SIMWORLD_TIME_ADVANCE_MODE": "legacy_tick"},
    )
    def test_legacy_sync_initializes_tick_scheduler_before_pause(self):
        unrealcv = make_unrealcv(["ok", "ok"])

        unrealcv.set_mode("sync", tick_interval=0.05)

        self.assertEqual(
            [
                unittest.mock.call("vset /action/tick_intervel 0.05"),
                unittest.mock.call("vset /action/game/pause"),
            ],
            unrealcv.client.request.call_args_list,
        )

    @patch("simworld.communicator.unrealcv.time.sleep")
    def test_advance_resumes_waits_and_pauses(self, sleep_mock):
        unrealcv = make_unrealcv(["ok", "ok"])

        unrealcv.advance_simulation_time(4.0, time_scale=2.0)

        self.assertEqual(
            [
                unittest.mock.call("vset /action/game/resume"),
                unittest.mock.call("vset /action/game/pause"),
            ],
            unrealcv.client.request.call_args_list,
        )
        sleep_mock.assert_called_once_with(2.0)

    @patch("simworld.communicator.unrealcv.time.sleep")
    def test_advance_repauses_when_wait_is_interrupted(self, sleep_mock):
        unrealcv = make_unrealcv(["ok", "ok"])
        sleep_mock.side_effect = KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            unrealcv.advance_simulation_time(1.0)

        self.assertEqual(
            "vset /action/game/pause",
            unrealcv.client.request.call_args_list[-1].args[0],
        )

    @patch.dict(
        os.environ,
        {"SIMWORLD_TIME_ADVANCE_MODE": "legacy_tick"},
    )
    @patch("simworld.communicator.unrealcv.time.sleep")
    def test_legacy_city_advance_uses_native_tick_commands(
        self,
        sleep_mock,
    ):
        unrealcv = make_unrealcv(["ok", "ok"])

        unrealcv.advance_simulation_time(4.0, time_scale=2.0)

        self.assertEqual(
            [
                unittest.mock.call("vset /action/tick_intervel 4.0"),
                unittest.mock.call("vset /action/tick"),
            ],
            unrealcv.client.request.call_args_list,
        )
        self.assertEqual(
            [unittest.mock.call(1.0), unittest.mock.call(3.0)],
            sleep_mock.call_args_list,
        )

    def test_server_error_is_not_silently_treated_as_success(self):
        unrealcv = make_unrealcv(["error Unrecognized command"])

        with self.assertRaisesRegex(RuntimeError, "Unrecognized command"):
            unrealcv.set_mode("async")


if __name__ == "__main__":
    unittest.main()
