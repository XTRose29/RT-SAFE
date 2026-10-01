"""UnrealCV communication module.

This module provides a client interface for communicating with Unreal Engine,
allowing for various operations such as object spawning, movement, and image
capture.
"""
import json
import os
import queue
import time
from io import BytesIO
from threading import Lock

import cv2
import numpy as np
import PIL.Image
import unrealcv
from IPython.display import display

from simworld.utils.logger import Logger


class UnrealCV(object):
    """Interface class for communication with Unreal Engine.

    This class provides various functionalities for communicating with Unreal Engine,
    including basic operations and traffic system operations.
    """

    def __init__(self, port=9000, ip='127.0.0.1', resolution=(1280, 720)):
        """Initialize the UnrealCV client.

        Args:
            port: Connection port, defaults to 9000.
            ip: Connection IP address, defaults to 127.0.0.1.
            resolution: Resolution, defaults to (320, 240).
        """
        self.ip = ip
        self.logger = Logger.get_logger('UnrealCV')
        self.request_timeout_seconds = max(
            1.0,
            float(os.environ.get('SIMWORLD_UNREALCV_REQUEST_TIMEOUT_SECONDS', '30')),
        )
        self.request_reconnect_retries = max(
            0,
            int(os.environ.get('SIMWORLD_UNREALCV_RECONNECT_RETRIES', '1')),
        )
        # Build a client to connect to the environment
        self.client = unrealcv.Client((ip, port))
        self._install_bounded_request()
        self.client.connect()

        self.resolution = resolution

        self.lock = Lock()
        # Newer UnrealCV builds no longer expose the legacy
        # ``/action/tick_intervel`` and ``/action/tick`` commands.  Keep the
        # requested interval locally and advance a paused world by briefly
        # resuming it instead (see ``advance_simulation_time`` below).
        self._tick_interval = 0.05
        self.ini_unrealcv(resolution)

    def _install_bounded_request(self):
        """Make synchronous UnrealCV requests honor a real wall-clock timeout.

        The upstream client accepts a ``timeout`` argument but performs an
        unbounded ``Queue.get``.  If the UE multi-connect server loses one
        response while keeping the TCP socket open, the benchmark otherwise
        waits forever and the outer attempt timeout cannot produce useful
        fail-closed diagnostics.
        """
        client = self.client
        original_request = client.request
        original_request_batch = client.request_batch
        request_metrics = getattr(
            original_request,
            '_simworld_request_metrics',
            None,
        )

        def reconnect(active_client, commands):
            # The packaged multi-connect server keeps its request sequence
            # across TCP reconnects.  Starting the replacement client at zero
            # makes its receive thread reject the first valid reply (for
            # example, ``got 109218, expected 0``) and silently kills that
            # thread.  The timed-out send already advanced ``send_message_id``;
            # use that next ID for both sides of the replacement connection.
            next_message_id = int(active_client.send_message_id)
            replacement = unrealcv.Client(active_client.endpoint)
            replacement.send_message_id = next_message_id
            replacement.recv_message_id = next_message_id
            if not replacement.connect():
                rendered = '; '.join(commands)
                failure = ConnectionError(
                    'Failed to reconnect UnrealCV for idempotent command(s): '
                    f'{rendered}'
                )
                self.logger.error('%s', failure)
                raise failure
            self.client = replacement
            self._install_bounded_requests()
            return replacement

        def bounded_request_raw(message, timeout=5):
            if isinstance(message, list):
                return bounded_request_batch_raw(message)
            if timeout < 0:
                return original_request(message, timeout)

            payload = message if isinstance(message, bytes) else message.encode('utf-8')
            wait_seconds = float(self.request_timeout_seconds)
            command = payload.decode('utf-8', errors='replace')
            retry_limit = (
                self.request_reconnect_retries
                if self._is_idempotent_retry_command(command)
                else 0
            )
            active_client = client

            for attempt in range(retry_limit + 1):
                raw_message = b'%d:%s' % (active_client.send_message_id, payload)
                if not active_client.send(raw_message):
                    failure = ConnectionError('Failed to send: socket is closed')
                else:
                    active_client.send_message_id += 1
                    active_client.recv_num_q.put(-1)
                    try:
                        response = active_client.recv_data_q.get(timeout=wait_seconds)
                    except queue.Empty:
                        failure = TimeoutError(
                            f'UnrealCV request timed out after {wait_seconds:.1f}s: {command}'
                        )
                    else:
                        if response is not None:
                            return response
                        failure = ConnectionError(
                            'UnrealCV connection lost while awaiting response'
                        )

                active_client.disconnect()
                if attempt >= retry_limit:
                    self.logger.error('%s', failure)
                    raise failure

                self.logger.warning(
                    'Reconnecting after recoverable UnrealCV request failure '
                    '(attempt %d/%d): %s',
                    attempt + 1,
                    retry_limit,
                    command,
                )
                active_client = reconnect(active_client, [command])

            raise AssertionError('unreachable UnrealCV request retry state')

        def bounded_request_batch_raw(batch):
            payloads = [
                message if isinstance(message, bytes) else message.encode('utf-8')
                for message in batch
            ]
            if not payloads:
                return []
            commands = [
                payload.decode('utf-8', errors='replace') for payload in payloads
            ]
            retry_limit = (
                self.request_reconnect_retries
                if all(
                    self._is_idempotent_retry_command(command)
                    for command in commands
                )
                else 0
            )
            active_client = client

            for attempt in range(retry_limit + 1):
                failure = None
                for payload in payloads:
                    raw_message = b'%d:%s' % (
                        active_client.send_message_id,
                        payload,
                    )
                    if not active_client.send(raw_message):
                        failure = ConnectionError(
                            'Failed to send batch: socket is closed'
                        )
                        break
                    active_client.send_message_id += 1

                responses = []
                if failure is None:
                    active_client.recv_num_q.put(-len(payloads))
                    deadline = time.monotonic() + float(
                        self.request_timeout_seconds
                    )
                    for _ in payloads:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            failure = TimeoutError(
                                'UnrealCV batch request timed out after '
                                f'{self.request_timeout_seconds:g}s: '
                                + '; '.join(commands)
                            )
                            break
                        try:
                            response = active_client.recv_data_q.get(
                                timeout=remaining
                            )
                        except queue.Empty:
                            failure = TimeoutError(
                                'UnrealCV batch request timed out after '
                                f'{self.request_timeout_seconds:.1f}s: '
                                + '; '.join(commands)
                            )
                            break
                        if response is None:
                            failure = ConnectionError(
                                'UnrealCV connection lost while awaiting batch response'
                            )
                            break
                        responses.append(response)
                    if failure is None:
                        return responses

                active_client.disconnect()
                if attempt >= retry_limit:
                    self.logger.error('%s', failure)
                    raise failure

                self.logger.warning(
                    'Reconnecting after recoverable UnrealCV batch failure '
                    '(attempt %d/%d): %s',
                    attempt + 1,
                    retry_limit,
                    '; '.join(commands),
                )
                active_client = reconnect(active_client, commands)

            raise AssertionError('unreachable UnrealCV batch retry state')

        def measured(operation):
            if request_metrics is None:
                return operation()
            started = time.monotonic()
            failed = False
            try:
                return operation()
            except Exception:
                failed = True
                raise
            finally:
                request_metrics.record(time.monotonic() - started, failed)

        def bounded_request(message, timeout=5):
            return measured(lambda: bounded_request_raw(message, timeout))

        def bounded_request_batch(batch):
            return measured(lambda: bounded_request_batch_raw(batch))

        client.request = bounded_request
        client.request_batch = bounded_request_batch
        client._simworld_original_request = original_request
        client._simworld_original_request_batch = original_request_batch

    def _install_bounded_requests(self):
        """Compatibility name retained by benchmark-clean integrations."""
        return self._install_bounded_request()

    @staticmethod
    def _is_read_only_retry_command(command):
        """Return whether a lost response is a read that is safe to replay."""
        normalized = str(command).strip().lower()
        if normalized.startswith('vget '):
            return True
        tokens = normalized.split()
        return bool(
            len(tokens) >= 3
            and tokens[0] == 'vbp'
            and tokens[2].startswith('get')
        )

    @staticmethod
    def _is_idempotent_retry_command(command):
        """Return whether replaying ``command`` cannot duplicate world state.

        Actor spawning/destruction and arbitrary Blueprint calls are deliberately
        excluded.  The allowlist only covers reads and setters whose repeated
        execution converges to the same state; these are the commands observed
        losing replies under the packaged multi-connect server.
        """
        normalized = str(command).strip().lower()
        if normalized.startswith('vget '):
            return True
        if normalized in {
            'vset /action/game/resume',
            'vset /action/game/pause',
        }:
            return True
        if not normalized.startswith('vset /object/'):
            return False
        return any(
            marker in normalized
            for marker in (
                '/location ',
                '/rotation ',
                '/scale ',
                '/color ',
                '/physics ',
                '/collision ',
                '/object_mobility ',
            )
        )

    ###################################################
    # Basic Operations
    ###################################################
    def disconnect(self):
        """Disconnect from Unreal Engine."""
        self.client.disconnect()

    def ini_unrealcv(self, resolution=(320, 240)):
        """Initialize UnrealCV settings.

        Args:
            resolution: Resolution, defaults to (320, 240).
        """
        self.check_connection()
        [w, h] = resolution
        # Keep initialization requests synchronous.  Sending these as
        # fire-and-forget and immediately issuing the sync-mode pause can make
        # the packaged UnrealCV server deliver the earlier response to the
        # later request, leaving the client blocked despite a server response.
        # Matrix recovery can opt out when UE was already launched at exactly
        # this resolution.  Re-applying the same viewport size can deadlock the
        # packaged RT15 renderer while it waits on its render thread; keeping
        # this opt-in preserves the historical behavior for every other run.
        skip_initial_setres = os.environ.get(
            'SIMWORLD_SKIP_INITIAL_SETRES', ''
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        if skip_initial_setres:
            self.logger.info(
                'Skipping redundant initial setres request for %dx%d viewport',
                w,
                h,
            )
        else:
            self.client.request(
                f'vrun setres {w}x{h}w'
            )  # Set resolution of display window
        skip_async_skinned_compilation = os.environ.get(
            'SIMWORLD_SKIP_ASYNC_SKINNED_ASSET_COMPILATION', ''
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        if skip_async_skinned_compilation:
            self.logger.info(
                'Skipping redundant AsyncSkinnedAssetCompilation startup request'
            )
        else:
            self.client.request('vrun Editor.AsyncSkinnedAssetCompilation 2')
        time.sleep(1)

    def check_connection(self):
        """Check connection status, attempt to reconnect if not connected."""
        while self.client.isconnected() is False:
            self.logger.error('UnrealCV server is not running. Please try again')
            time.sleep(1)
            self.client.connect()

    # Deprecated
    def spawn(self, prefab, name):
        """Spawn an object (deprecated).

        Args:
            prefab: Prefab.
            name: Object name.
        """
        cmd = f'vset /objects/spawn {prefab} {name}'
        with self.lock:
            self.client.request(cmd)

    def spawn_bp_asset(self, prefab_path, name):
        """Spawn a blueprint asset.

        Args:
            prefab_path: Prefab path.
            name: Object name.
        """
        cmd = f'vset /objects/spawn_bp_asset {prefab_path} {name}'
        with self.lock:
            self.client.request(cmd)

    def clean_garbage(self):
        """Clean garbage objects."""
        with self.lock:
            self.client.request('vset /action/clean_garbage')

    def set_location(self, loc, name):
        """Set object location.

        Args:
            loc: Location coordinates in the form [x, y, z].
            name: Object name.
        """
        [x, y, z] = loc
        cmd = f'vset /object/{name}/location {x} {y} {z}'
        with self.lock:
            self.client.request(cmd)

    def set_orientation(self, orientation, name):
        """Set object orientation.

        Args:
            orientation: Orientation in the form [pitch, yaw, roll].
            name: Object name.
        """
        [pitch, yaw, roll] = orientation
        cmd = f'vset /object/{name}/rotation {pitch} {yaw} {roll}'
        with self.lock:
            self.client.request(cmd)

    def set_scale(self, scale, name):
        """Set object scale.

        Args:
            scale: Scale in the form [x, y, z].
            name: Object name.
        """
        [x, y, z] = scale
        cmd = f'vset /object/{name}/scale {x} {y} {z}'
        with self.lock:
            self.client.request(cmd)

    def set_color(self, actor_name, color):
        """Set object color.

        Args:
            actor_name: Object name.
            color: Color in the form [R, G, B].
        """
        [R, G, B] = color
        cmd = f'vset /object/{actor_name}/color {R} {G} {B}'
        with self.lock:
            self.client.request(cmd)

    def enable_controller(self, name, enable_controller):
        """Enable or disable controller.

        Args:
            name: Object name.
            enable_controller: Whether to enable controller.
        """
        cmd = f'vbp {name} EnableController {enable_controller}'
        with self.lock:
            self.client.request(cmd)

    def set_physics(self, actor_name, hasPhysics):
        """Set physics properties.

        Args:
            actor_name: Actor name.
            hasPhysics: Whether to enable physics.
        """
        cmd = f'vset /object/{actor_name}/physics {hasPhysics}'
        with self.lock:
            self.client.request(cmd)

    def set_collision(self, actor_name, hasCollision):
        """Set collision properties.

        Args:
            actor_name: Actor name.
            hasCollision: Whether to enable collision.
        """
        cmd = f'vset /object/{actor_name}/collision {hasCollision}'
        with self.lock:
            self.client.request(cmd)

    def set_movable(self, actor_name, isMovable):
        """Set movable properties.

        Args:
            actor_name: Actor name.
            isMovable: Whether the object is movable.
        """
        cmd = f'vset /object/{actor_name}/object_mobility {isMovable}'
        with self.lock:
            self.client.request(cmd)

    def set_fps(self, fps):
        """Set FPS.

        Args:
            fps: FPS.
        """
        cmd = f'vset /action/set_fixed_frame_rate {fps}'
        with self.lock:
            self.client.request(cmd)

    def pause_simulation(self):
        """Pause world simulation using the supported UnrealCV command."""
        return self._request_checked('vset /action/game/pause')

    def resume_simulation(self):
        """Resume world simulation using the supported UnrealCV command."""
        return self._request_checked('vset /action/game/resume')

    def set_mode(self, mode='async', tick_interval=0.05):
        """Set asynchronous or synchronous mode.

        Args:
            mode: Mode.
            tick_interval: Tick interval if synchronous mode.
        """
        if mode == 'sync':
            self.set_tick_interval(tick_interval)
            if self._time_advance_mode() == 'legacy_tick':
                # The packaged city plugin initializes its custom synchronous
                # scheduler when the interval command is received.  This must
                # happen before pausing, matching the original benchmark.
                self._request_checked(
                    f'vset /action/tick_intervel {float(tick_interval)}'
                )
            self.pause_simulation()
        elif mode == 'async':
            self.resume_simulation()
        else:
            raise ValueError(f'Invalid mode: {mode}. Please choose from "sync" or "async".')

    def set_tick_interval(self, interval):
        """Set the simulated duration advanced by :meth:`tick`.

        Older SimWorld packages implemented this with the misspelled custom
        UnrealCV command ``tick_intervel``.  The current plugin removed that
        command, so storing the interval client-side is the portable contract.
        """
        interval = float(interval)
        if interval < 0:
            raise ValueError('Tick interval must be non-negative')
        self._tick_interval = interval

    def tick(self):
        """Advance a paused world by the configured tick interval."""
        self.advance_simulation_time(self._tick_interval)

    def _request_checked(self, cmd):
        """Execute an UnrealCV command and fail fast on server errors."""
        with self.lock:
            response = self.client.request(cmd)
        if isinstance(response, str) and response.strip().lower().startswith('error'):
            raise RuntimeError(f'UnrealCV command failed: {cmd}: {response}')
        return response

    @staticmethod
    def _time_advance_mode():
        return os.environ.get(
            'SIMWORLD_TIME_ADVANCE_MODE',
            'resume_pause',
        ).strip().lower()

    def advance_simulation_time(self, seconds, time_scale=1.0):
        """Advance a synchronously paused world by ``seconds``.

        Native-controller packages use resume/run/pause.  The original city
        package instead exposes SimWorld's custom ``tick_intervel`` and ``tick``
        commands; its legacy Blueprint timers only advance through that path.
        Select the latter explicitly with
        ``SIMWORLD_TIME_ADVANCE_MODE=legacy_tick``.
        """
        seconds = float(seconds)
        time_scale = float(time_scale)
        if seconds < 0:
            raise ValueError('Simulation duration must be non-negative')
        if time_scale <= 0:
            raise ValueError('Time scale must be positive')
        if seconds == 0:
            return

        advance_mode = self._time_advance_mode()
        if advance_mode == 'legacy_tick':
            self._request_checked(f'vset /action/tick_intervel {seconds}')
            # The packaged city UnrealCV implementation schedules the custom
            # tick asynchronously.  Match the original benchmark contract:
            # let the interval update settle, then wait until the simulated
            # interval has actually been consumed before reading UE state.
            time.sleep(1.0)
            self._request_checked('vset /action/tick')
            time.sleep(seconds / time_scale + 1.0)
            return
        if advance_mode != 'resume_pause':
            raise ValueError(
                'SIMWORLD_TIME_ADVANCE_MODE must be '
                "'legacy_tick' or 'resume_pause', got "
                f'{advance_mode!r}'
            )

        self.resume_simulation()
        try:
            time.sleep(seconds / time_scale)
        finally:
            self.pause_simulation()

    def destroy(self, actor_name):
        """Destroy an object.

        Args:
            actor_name: Actor name.
        """
        cmd = f'vset /object/{actor_name}/destroy'
        with self.lock:
            self.client.request(cmd)

    def get_objects(self):
        """Get all objects.

        Returns:
            List of objects.
        """
        with self.lock:
            res = self.client.request('vget /objects')
        # UnrealCV returns bytes in some packaged runtimes and text in others.
        # Keep that transport detail out of higher-level scene cleanup: mixing
        # a bytes ndarray with string prefixes makes ``np.char.startswith``
        # raise and leaves every actor alive for the next rollout.
        objects = np.asarray([
            item.decode('utf-8', errors='replace')
            if isinstance(item, (bytes, bytearray, np.bytes_))
            else str(item)
            for item in res.split()
        ], dtype=str)
        return objects

    def set_object_name(self, name, new_name):
        """Set object name.

        Args:
            name: Object name.
            new_name: New object name.
        """
        cmd = f'vset /object/{name}/name {new_name}'
        with self.lock:
            self.client.request(cmd)

    def get_collision_num(self, actor_name):
        """Get collision number.

        Args:
            actor_name: Actor name.

        Returns:
            json: {
                "HumanCollision": 0,
                "ObjectCollision": 0,
                "BuildingCollision": 0,
                "VehicleCollision": 0
            }
        """
        with self.lock:
            res = self.client.request(f'vbp {actor_name} GetCollisionNum')
        return res

    def get_location(self, actor_name):
        """Get object location.

        Args:
            actor_name: Actor name.

        Returns:
            Location coordinates array.
        """
        cmd = f'vget /object/{actor_name}/location'
        with self.lock:
            res = self.client.request(cmd)
        location = [float(i) for i in res.split()]
        return np.array(location)

    def get_location_batch(self, actor_names):
        """Batch get object locations.

        Args:
            actor_names: List of actor names.

        Returns:
            List of location coordinate arrays.
        """
        cmd = [f'vget /object/{actor_name}/location' for actor_name in actor_names]
        with self.lock:
            res = self.client.request_batch(cmd)
        # Parse each response and convert to numpy array
        locations = [np.array([float(i) for i in r.split()]) for r in res]
        return locations

    def get_orientation(self, actor_name):
        """Get object orientation.

        Args:
            actor_name: Actor name.

        Returns:
            Orientation array.
        """
        cmd = f'vget /object/{actor_name}/rotation'
        with self.lock:
            res = self.client.request(cmd)
            orientation = [float(i) for i in res.split()]
        return np.array(orientation)

    def get_orientation_batch(self, actor_names):
        """Batch get object orientations.

        Args:
            actor_names: List of actor names.

        Returns:
            List of orientation arrays.
        """
        cmd = [f'vget /object/{actor_name}/rotation' for actor_name in actor_names]
        with self.lock:
            res = self.client.request_batch(cmd)
        # Parse each response and convert to numpy array
        orientations = [np.array([float(i) for i in r.split()]) for r in res]
        return orientations

    ##############################################################
    # Traffic System
    ##############################################################
    def v_set_state(self, object_name, throttle, brake, steering):
        """Set vehicle state.

        Args:
            object_name: Object name.
            throttle: Throttle value.
            brake: Brake value.
            steering: Steering value.
        """
        cmd = f'vbp {object_name} SetState {throttle} {brake} {steering}'
        with self.lock:
            self.client.request(cmd)

    def v_make_u_turn(self, object_name):
        """Make vehicle U-turn.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} MakeUTurn'
        with self.lock:
            self.client.request(cmd)

    def v_set_states(self, manager_object_name, states: str):
        """Batch set vehicle states.

        Args:
            manager_object_name: Manager object name.
            states: States string.
        """
        cmd = f'vbp {manager_object_name} VSetState {states}'
        with self.lock:
            self.client.request(cmd)

    def p_set_states(self, manager_object_name, states: str):
        """Batch set pedestrian states.

        Args:
            manager_object_name: Manager object name.
            states: States string.
        """
        cmd = f'vbp {manager_object_name} PSetState {states}'
        with self.lock:
            self.client.request(cmd)

    def p_stop(self, object_name):
        """Stop pedestrian movement.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} StopPedestrian'
        with self.lock:
            self.client.request(cmd)

    def p_move_forward(self, object_name):
        """Move pedestrian forward.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} MoveForward'
        with self.lock:
            self.client.request(cmd)

    def p_rotate(self, object_name, angle, direction='left'):
        """Rotate pedestrian.

        Args:
            object_name: Object name.
            angle: Angle.
            direction: Direction, defaults to 'left'.
        """
        if direction == 'right':
            clockwise = 1
        elif direction == 'left':
            angle = -angle
            clockwise = -1
        cmd = f'vbp {object_name} Rotate_Angle {1} {angle} {clockwise}'
        with self.lock:
            self.client.request(cmd)

    def p_set_speed(self, object_name, speed):
        """Set pedestrian speed.

        Args:
            object_name: Object name.
            speed: Speed.
        """
        cmd = f'vbp {object_name} SetMaxSpeed {speed}'
        with self.lock:
            self.client.request(cmd)

    def p_movement_simulation(self, object_name):
        """Start pedestrian movement simulation.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} MovementSimulation'
        with self.lock:
            self.client.request(cmd)

    def p_set_waypoints(self, object_name, waypoints):
        """Set pedestrian waypoints.

        Args:
            object_name: Object name.
            waypoints: String of waypoints. Use semicolon to separate waypoints and use comma to separate coordinates. Example: "100,100;200,200;300,300"
        """
        cmd = f'vbp {object_name} SetWaypoints {waypoints}'
        with self.lock:
            self.client.request(cmd)

    def tl_set_vehicle_green(self, object_name: str):
        """Set vehicle traffic light to green.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} SwitchVehicleFrontGreen'
        with self.lock:
            self.client.request(cmd)

    def tl_set_pedestrian_walk(self, object_name: str):
        """Set pedestrian traffic light to walk.

        Args:
            object_name: Object name.
        """
        cmd = f'vbp {object_name} SetPedestrianWalk'
        with self.lock:
            self.client.request(cmd)

    def tl_set_vehicle_red(self, object_name: str):
        """Hold both vehicle faces at red on a legacy traffic head."""
        commands = (
            f'vbp {object_name} SwitchVehicleFrontRed',
            f'vbp {object_name} SwitchVehicleBackRed',
        )
        with self.lock:
            return [self.client.request(command) for command in commands]

    def tl_set_pedestrian_stop(self, object_name: str):
        """Render the pedestrian DON'T WALK indication on a legacy head."""
        cmd = f'vbp {object_name} SetPedestrianStop'
        with self.lock:
            return self.client.request(cmd)

    def tl_get_state(self, object_name: str):
        """Read the state currently rendered by a traffic-light Blueprint.

        The RT traffic-light assets expose ``GetState`` and return a JSON object
        containing ``green light``, ``ped green``, and ``ped time``. Reading the
        Blueprint directly keeps model input and evaluation aligned with the
        visible light instead of a separate Python-side timer.

        Args:
            object_name: Spawned traffic-light Blueprint object name.

        Returns:
            Raw JSON response returned by the Blueprint.
        """
        cmd = f'vbp {object_name} GetState'
        with self.lock:
            return self.client.request(cmd)

    def tl_set_duration(self, object_name: str, green_duration: float, yellow_duration: float, pedestrian_green_duration: float):
        """Set traffic light duration.

        Args:
            object_name: Object name.
            green_duration: Green duration.
            yellow_duration: Yellow duration.
            pedestrian_green_duration: Pedestrian green duration.
        """
        cmd = f'vbp {object_name} SetDuration {green_duration} {yellow_duration} {pedestrian_green_duration}'
        with self.lock:
            self.client.request(cmd)

    def get_informations(self, manager_object_name):
        """Get information.

        Args:
            manager_object_name: Name of the manager object to get information from.

        Returns:
            str: Information string containing the current state of the environment.
        """
        cmd = f'vbp {manager_object_name} GetInformation'
        with self.lock:
            return self.client.request(cmd)

    def update_ue_manager(self, manager_object_name):
        """Update UE manager.

        Args:
            manager_object_name: Name of the manager object to update.
        """
        cmd = f'vbp {manager_object_name} UpdateObjects'
        with self.lock:
            self.client.request(cmd)

    def add_vehicle_signal(self, intersection_name, vehicle_signal_name):
        """Add vehicle signal.

        Args:
            intersection_name: Name of the intersection to add traffic signal to.
            vehicle_signal_name: Name of the vehicle signal to add.
        """
        cmd = f'vbp {intersection_name} AddVehicleSignal {vehicle_signal_name}'
        with self.lock:
            self.client.request(cmd)

    def add_pedestrian_signal(self, intersection_name, pedestrian_signal_name):
        """Add pedestrian signal.

        Args:
            intersection_name: Name of the intersection to add pedestrian signal to.
            pedestrian_signal_name: Name of the pedestrian signal to add.
        """
        cmd = f'vbp {intersection_name} AddPedSignal {pedestrian_signal_name}'
        with self.lock:
            self.client.request(cmd)

    def traffic_signal_start_simulation(self, intersection_name):
        """Start traffic signal simulation."""
        cmd = f'vbp {intersection_name} StartSimulation'
        with self.lock:
            self.client.request(cmd)

    def traffic_signal_configure_phase_plan(
        self,
        intersection_name,
        vehicle_green_s,
        vehicle_yellow_s,
        all_red_s,
        pedestrian_walk_s,
        pedestrian_clearance_s,
    ):
        """Configure the intersection-level real-world phase plan.

        This is the Python side of the upgraded ``RT_BP_Intersection`` API.
        Packaged builds without ``ConfigurePhasePlan`` must keep the feature
        disabled and use the per-signal ``SetDuration`` compatibility path.
        """
        cmd = (
            f'vbp {intersection_name} ConfigurePhasePlan '
            f'{vehicle_green_s} {vehicle_yellow_s} {all_red_s} '
            f'{pedestrian_walk_s} {pedestrian_clearance_s}'
        )
        with self.lock:
            return self.client.request(cmd)

    def traffic_signal_get_intersection_state(self, intersection_name):
        """Return the canonical state of every head in one intersection."""
        cmd = f'vbp {intersection_name} GetIntersectionState'
        with self.lock:
            return self.client.request(cmd)

    def traffic_signal_set_pedestrian_occupancy(
        self,
        intersection_name,
        pedestrian_occupied,
    ):
        """Feed passive crosswalk occupancy to the native controller."""
        occupied = 'true' if pedestrian_occupied else 'false'
        cmd = f'vbp {intersection_name} SetPedestrianOccupancy {occupied}'
        with self.lock:
            return self.client.request(cmd)

    def traffic_signal_set_intersection_phase(
        self,
        intersection_name,
        phase,
        active_vehicle_group=0,
    ):
        """Force a deterministic phase for UE render/integration testing."""
        cmd = (
            f'vbp {intersection_name} SetPhase '
            f'{phase} {active_vehicle_group}'
        )
        with self.lock:
            return self.client.request(cmd)

    ##############################################################
    # Robot System
    ##############################################################

    def dog_move(self, robot_name, action):
        """Apply transition action.

        Args:
            robot_name: Robot name.
            action: Action in the form [speed, duration, direction].
        """
        [speed, duration, direction] = action
        if speed < 0:
            # Switch direction
            if direction == 0:
                direction = 1
            elif direction == 1:
                direction = 0
            elif direction == 2:
                direction = 3
            elif direction == 3:
                direction = 2
        cmd = f'vbp {robot_name} Move_Speed {speed} {duration} {direction}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(duration)

    def dog_rotate(self, robot_name, action):
        """Apply rotation action.

        Args:
            robot_name: Robot name.
            action: Action in the form [duration, angle, direction].
        """
        [duration, angle, direction] = action
        cmd = f'vbp {robot_name} Rotate_Angle {duration} {angle} {direction}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(duration)

    def dog_look_up(self, robot_name):
        """Apply look up action.

        Args:
            robot_name: Robot name.
        """
        cmd = f'vbp {robot_name} lookup'
        with self.lock:
            self.client.request(cmd)

    def dog_look_down(self, robot_name):
        """Apply look down action.

        Args:
            robot_name: Robot name.
        """
        cmd = f'vbp {robot_name} lookdown'
        with self.lock:
            self.client.request(cmd)

    ##############################################################
    # Humanoid System
    # For '/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C'
    ##############################################################

    def humanoid_move_forward(self, object_name):
        """Move humanoid forward.

        Args:
            object_name: Name of the humanoid object to move forward.
        """
        cmd = f'vbp {object_name} MoveForward'
        with self.lock:
            self.client.request(cmd)

    def humanoid_rotate(self, object_name, angle, direction='left'):
        """Rotate humanoid.

        Args:
            object_name: Name of the humanoid object to rotate.
            angle: Rotation angle in degrees.
            direction: Direction of rotation, either 'left' or 'right'. Defaults to 'left'.
        """
        if direction == 'right':
            clockwise = 1
        elif direction == 'left':
            angle = -angle
            clockwise = -1
        cmd = f'vbp {object_name} TurnAround {1} {angle} {clockwise}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(1)

    def humanoid_stop(self, object_name):
        """Stop humanoid.

        Args:
            object_name: Name of the humanoid object to stop.
        """
        cmd = f'vbp {object_name} StopAgent'
        with self.lock:
            self.client.request(cmd)

    def humanoid_step_forward(self, object_name, duration, direction=0):
        """Step forward.

        Args:
            object_name: Name of the humanoid object to step forward.
            duration: Duration of the step forward movement in seconds.
            direction: Direction of the step forward movement.
        """
        cmd = f'vbp {object_name} StepForward {duration} {direction}'
        with self.lock:
            self.client.request(cmd)
        time.sleep(duration)

    def humanoid_set_speed(self, object_name, speed):
        """Set humanoid speed.

        Args:
            object_name: Name of the humanoid object to set speed.
            speed: Speed to set.
        """
        cmd = f'vbp {object_name} SetMaxSpeed {speed}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_sit_down(self, object_name):
        """Sit down.

        Args:
            object_name: Name of the humanoid object to sit down.
        """
        cmd = f'vbp {object_name} SitDown'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_stand_up(self, object_name):
        """Stand up.

        Args:
            object_name: Name of the humanoid object to sit down.
        """
        cmd = f'vbp {object_name} StandUp'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_get_on_scooter(self, object_name):
        """Get on scooter.

        Args:
            object_name: Name of the humanoid object to get on scooter.
        """
        cmd = f'vbp {object_name} GetOnScooter'
        with self.lock:
            self.client.request(cmd)
        self.clean_garbage()

    def humanoid_get_off_scooter(self, object_name):
        """Get off scooter.

        Args:
            object_name: Name of the humanoid object to get off scooter.
        """
        cmd = f'vbp {object_name} GetOffScooter'
        with self.lock:
            self.client.request(cmd)

    def humanoid_pick_up_object(self, humanoid_name, object_name):
        """Pick up object.

        Args:
            humanoid_name: Name of the humanoid to pick up object.
            object_name: Name of the object to pick up.
        """
        cmd = f'vbp {humanoid_name} PickUp {object_name}'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_drop_object(self, humanoid_name):
        """Drop object.

        Args:
            humanoid_name: Name of the humanoid to drop object.
        """
        cmd = f'vbp {humanoid_name} DropOff'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_enter_vehicle(self, humanoid_name, vehicle_name):
        """Enter vehicle.

        Args:
            humanoid_name: Name of the humanoid to enter vehicle.
            vehicle_name: Name of the vehicle to enter.
        """
        cmd = f'vbp {humanoid_name} EnterVehicle {vehicle_name}'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_exit_vehicle(self, humanoid_name, vehicle_name):
        """Exit vehicle.

        Args:
            humanoid_name: Name of the humanoid to enter vehicle.
            vehicle_name: Name of the vehicle to exit.
        """
        cmd = f'vbp {humanoid_name} ExitVehicle {vehicle_name}'
        with self.lock:
            res = self.client.request(cmd)
            success = str(json.loads(res)['Success'])
            if success == 'false':
                return False
            elif success == 'true':
                return True

    def humanoid_discuss(self, humanoid_name, discuss_type):
        """Discuss.

        Args:
            humanoid_name: Name of the humanoid to discuss.
            discuss_type: Type of discussion. Can be [0, 1]
        """
        cmd = f'vbp {humanoid_name} Discussion {discuss_type}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_argue(self, humanoid_name, argue_type):
        """Argue.

        Args:
            humanoid_name: Name of the humanoid to argue.
            argue_type: Type of arguing. Can be [0, 1]
        """
        cmd = f'vbp {humanoid_name} Arguing {argue_type}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_listen(self, humanoid_name):
        """Listen.

        Args:
            humanoid_name: Name of the humanoid to discuss.
        """
        cmd = f'vbp {humanoid_name} Listening'
        with self.lock:
            self.client.request(cmd)

    def humanoid_wave_to_dog(self, humanoid_name):
        """Wave to dog.

        Args:
            humanoid_name: Name of the humanoid to wave to dog.
        """
        cmd = f'vbp {humanoid_name} Wave2Dog'
        with self.lock:
            self.client.request(cmd)

    def humanoid_directing_path(self, humanoid_name):
        """Directing path.

        Args:
            humanoid_name: Name of the humanoid to directing path.
        """
        cmd = f'vbp {humanoid_name} Directing'
        with self.lock:
            self.client.request(cmd)

    def humanoid_stop_current_action(self, humanoid_name):
        """Stop current action.

        Args:
            humanoid_name: Name of the humanoid to stop current action.
        """
        cmd = f'vbp {humanoid_name} StopAction'
        with self.lock:
            self.client.request(cmd)

    def s_set_state(self, object_name, throttle, brake, steering):
        """Set scooter state.

        Args:
            object_name: Name of the scooter object.
            throttle: Throttle value.
            brake: Brake value.
            steering: Steering value.
        """
        cmd = f'vbp {object_name} SetState {throttle} {brake} {steering}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_set_path(self, object_name, path):
        """Set humanoid path.

        Args:
            object_name: Name of the humanoid object.
            path: String of path. Use semicolon to separate waypoints and use comma to separate coordinates. Example: "100,100;200,200;300,300"
        """
        cmd = f'vbp {object_name} SetPath {path}'
        with self.lock:
            self.client.request(cmd)

    def humanoid_follow_path(self, object_name):
        """Follow path.

        Args:
            object_name: Name of the humanoid object.
        """
        cmd = f'vbp {object_name} FollowPath'
        with self.lock:
            self.client.request(cmd)

    ##############################################################
    # Camera
    ##############################################################
    def get_cameras(self):
        """Get all cameras.

        Returns:
            List of camera names.
        """
        cmd = 'vget /cameras'
        with self.lock:
            return self.client.request(cmd)

    def spawn_camera(self):
        """Spawn a free UnrealCV camera and return the server response."""
        cmd = 'vset /cameras/spawn'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_location(self, camera_id: int):
        """Get camera location.

        Args:
            camera_id: ID of the camera to get location.

        Returns:
            Location (x, y, z) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/location'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_rotation(self, camera_id: int):
        """Get camera rotation.

        Args:
            camera_id: ID of the camera to get rotation.

        Returns:
            Rotation (yaw, pitch, roll) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/rotation'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_fov(self, camera_id: int):
        """Get camera field of view.

        Args:
            camera_id: ID of the camera to get field of view.

        Returns:
            Horizontal field of view of the camera.
        """
        cmd = f'vget /camera/{camera_id}/fov'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_resolution(self, camera_id: int):
        """Get camera resolution.

        Args:
            camera_id: ID of the camera to get resolution.

        Returns:
            Resolution (width, height) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/size'
        with self.lock:
            return self.client.request(cmd)

    def set_camera_location(self, camera_id: int, location: tuple):
        """Set camera location.

        Args:
            camera_id: ID of the camera to set location.
            location: Location (x, y, z) of the camera.
        """
        cmd = f'vset /camera/{camera_id}/location {location[0]} {location[1]} {location[2]}'
        with self.lock:
            self.client.request(cmd)

    def set_camera_rotation(self, camera_id: int, rotation: tuple):
        """Set camera rotation.

        Args:
            camera_id: ID of the camera to set rotation.
            rotation: Rotation (pitch, yaw, roll) of the camera.
        """
        cmd = f'vset /camera/{camera_id}/rotation {rotation[0]} {rotation[1]} {rotation[2]}'
        with self.lock:
            self.client.request(cmd)

    def set_camera_fov(self, camera_id: int, fov: float):
        """Set camera field of view.

        Args:
            camera_id: ID of the camera to set field of view.
            fov: Horizontal field of view of the camera.
        """
        cmd = f'vset /camera/{camera_id}/fov {fov}'
        with self.lock:
            self.client.request(cmd)

    def set_camera_resolution(self, camera_id: int, resolution: tuple):
        """Set camera resolution.

        Args:
            camera_id: ID of the camera to set resolution.
            resolution: Resolution (width, height) of the camera.
        """
        cmd = f'vset /camera/{camera_id}/size {resolution[0]} {resolution[1]}'
        with self.lock:
            self.client.request(cmd)

    def show_img(self, img, title='raw_img'):
        """Display an image.

        Args:
            img: Image.
            title: Title, defaults to "raw_img".
        """
        try:
            # Check if the image is a depth image (single channel)
            if len(img.shape) == 2:
                # Normalize depth image for display
                img_normalized = img / img.max()
                # Convert to 8-bit grayscale
                img_display = (img_normalized * 255).astype(np.uint8)
                # Convert to RGB for display
                img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
            else:
                # Convert OpenCV BGR image to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Ensure the image is in uint8 format before converting to PIL Image
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)

            # Convert to PIL Image
            pil_img = PIL.Image.fromarray(img_rgb)
            # Display in notebook
            display(pil_img)
        except ImportError:
            # Fallback to OpenCV display if not in notebook
            if len(img.shape) == 2:
                # Normalize depth image for display
                img_normalized = img / img.max()
                # Convert to 8-bit grayscale
                img_display = (img_normalized * 255).astype(np.uint8)
                cv2.imshow(title, img_display)
            else:
                cv2.imshow(title, img)
            cv2.waitKey(3)

    def get_image(self, cam_id, viewmode, mode='direct', img_path=None):
        """Get image.

        Args:
            cam_id: Camera ID.
            viewmode: View mode.
            mode: Mode.
            img_path: Image path.
        """
        image = None
        try:
            if mode == 'direct':  # get image from unrealcv in png format
                if viewmode == 'depth':
                    cmd = f'vget /camera/{cam_id}/{viewmode} npy'
                    with self.lock:
                        res = self.client.request(cmd)
                    image = self._decode_npy(res)
                else:
                    cmd = f'vget /camera/{cam_id}/{viewmode} png'
                    with self.lock:
                        res = self.client.request(cmd)
                    image = self._decode_png(res)
            elif mode == 'file':  # save image to file and read it
                img_path = os.path.join(os.getcwd(), f'{cam_id}-{viewmode}.png')
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                with self.lock:
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)

            elif mode == 'fast':  # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                with self.lock:
                    res = self.client.request(cmd)
                image = self._decode_bmp(res)

            elif mode == 'file_path':  # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                with self.lock:
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)

            if image is None:
                raise ValueError(f'Failed to read image with mode={mode}, viewmode={viewmode}')
            return image

        except Exception as e:
            # A synthetic black frame is not a valid benchmark observation:
            # it hides transport loss, changes the configured resolution, and
            # lets a corrupted policy input be accepted as an ordinary step.
            # Fail closed so the rollout supervisor archives this partial
            # attempt and restarts only the interrupted seeded task in new UE.
            raise RuntimeError(
                f'Camera capture failed for camera {cam_id} ({viewmode}, '
                f'mode={mode}): {e}'
            ) from e

    def _decode_npy(self, res):
        """Decode NPY image.

        Args:
            res: NPY image.

        Returns:
            Decoded image.
        """
        image = np.load(BytesIO(res))
        eps = 1e-6
        depth_log = np.log(image + eps)

        depth_min = np.min(depth_log)
        depth_max = np.max(depth_log)
        normalized_depth = (depth_log - depth_min) / (depth_max - depth_min)

        gamma = 0.5
        normalized_depth = np.power(normalized_depth, gamma)

        image = (normalized_depth * 255).astype(np.uint8)

        # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        # image = cv2.applyColorMap(image, cv2.COLOR_GRAY2BGR)
        return image

    def _decode_png(self, res):
        """Decode PNG image.

        Args:
            res: PNG image.

        Returns:
            Decoded image.
        """
        # img = np.asarray(PIL.Image.open(BytesIO(res)))
        # img = img[:, :, :-1]  # delete alpha channel
        # img = img[:, :, ::-1]  # transpose channel order
        # return img
        pil_img = PIL.Image.open(BytesIO(res))

        rgb_img = pil_img.convert('RGB')

        img = np.asarray(rgb_img)

        img = img[:, :, ::-1]

        return img

    def _decode_bmp(self, res, channel=4):
        """Decode BMP image.

        Args:
            res: BMP image.
            channel: Channel.

        Returns:
            Decoded image.
        """
        img = np.fromstring(res, dtype=np.uint8)
        img = img[-self.resolution[1]*self.resolution[0]*channel:]
        img = img.reshape(self.resolution[1], self.resolution[0], channel)
        return img[:, :, :-1]

    def update_objects(self, object_name):
        """Update objects.

        Args:
            object_name: UE_Manager object name.
        """
        cmd = f'vbp {object_name} UpdateObjects'
        with self.lock:
            self.client.request(cmd)

    ##############################################################
    # Weather
    ##############################################################
    def set_sun_direction(self, weather_manager_name, pitch, yaw):
        """Set sun direction.

        Args:
            weather_manager_name: Name of the weather manager.
            pitch: Pitch of the sun. -89 - 89
            yaw: Yaw of the sun. 0 - 360
        """
        cmd = f'vbp {weather_manager_name} SetSunDirection {pitch} {yaw}'
        with self.lock:
            self.client.request(cmd)

    def get_sun_direction(self, weather_manager_name):
        """Get sun direction.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Sun direction.
        """
        cmd = f'vbp {weather_manager_name} GetSunDirection'
        with self.lock:
            return self.client.request(cmd)

    def set_sun_intensity(self, weather_manager_name, intensity):
        """Set sun intensity.

        Args:
            weather_manager_name: Name of the weather manager.
            intensity: Intensity of the sun. 0 - 100
        """
        cmd = f'vbp {weather_manager_name} SetSunIntensity {intensity}'
        with self.lock:
            self.client.request(cmd)

    def get_sun_intensity(self, weather_manager_name):
        """Get sun intensity.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Sun intensity.
        """
        cmd = f'vbp {weather_manager_name} GetSunIntensity'
        with self.lock:
            return self.client.request(cmd)

    def set_fog(self, weather_manager_name, density, distance, falloff):
        """Set fog.

        Args:
            weather_manager_name: Name of the weather manager.
            density: Density of the fog. 0 - 100
            distance: Distance of the fog. 0 - 5000, cm
            falloff: Falloff of the fog. 0 - 2
        """
        cmd = f'vbp {weather_manager_name} SetFog {density} {distance} {falloff}'
        with self.lock:
            self.client.request(cmd)

    def get_fog(self, weather_manager_name):
        """Get fog.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Fog.
        """
        cmd = f'vbp {weather_manager_name} GetFog'
        with self.lock:
            return self.client.request(cmd)

    def set_atmosphere(self, weather_manager_name, rayleigh, mie):
        """Set atmosphere.

        Args:
            weather_manager_name: Name of the weather manager.
            rayleigh: RayleighScatteringScale of the atmosphere. 0 - 2.
            mie: MieScatteringScale of the atmosphere. 0 - 5.
        """
        cmd = f'vbp {weather_manager_name} SetAtmosphere {rayleigh} {mie}'
        with self.lock:
            self.client.request(cmd)

    def get_atmosphere(self, weather_manager_name):
        """Get atmosphere.

        Args:
            weather_manager_name: Name of the weather manager.

        Returns:
            Atmosphere.
        """
        cmd = f'vbp {weather_manager_name} GetAtmosphere'
        with self.lock:
            return self.client.request(cmd)
