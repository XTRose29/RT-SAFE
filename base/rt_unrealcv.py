from simworld.communicator.unrealcv import UnrealCV
import json
import math

class RTUnrealCV(UnrealCV):
    def get_actor_sim_time(self, name):
        """Read UE game time relative to actor creation, including dilation.

        Unlike wall sleep, this clock stops while the world is paused. Relative
        differences on the same actor are valid throughout an episode.
        A paused spawn can return a tiny negative creation-time offset (observed
        -1 microsecond). Preserve it for differences; reject values below -100us.
        """
        with self.lock:
            raw = self.client.request(f'vbp {name} GetGameTimeSinceCreation')
        try:
            value = float(json.loads(raw)['ReturnValue'])
            if not math.isfinite(value) or value < -0.0001:
                raise ValueError('invalid game clock')
            return value
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f'Runtime did not return a valid native game clock: {str(raw)[:200]!r}'
            ) from exc

    def tl_get_states(self, names):
        """Read rendered heads in one bounded transport batch, in input order."""
        if not names:
            return []
        with self.lock:
            responses = self.client.request_batch([
                f'vbp {name} GetState' for name in names
            ])
        if not isinstance(responses, (list, tuple)) or len(responses) != len(names):
            raise RuntimeError('Incomplete native traffic-light state batch')
        return responses

    def move_actor_with_sweep(self, name, location):
        """Move the root collision shape on a straight segment, never teleport.

        The ordinary /object/location command disables sweep. K2_SetActorLocation
        is available on the packaged actor and sweeps its collision capsule,
        dispatching UE hit/overlap callbacks even across thin obstacles. Its
        return value does not mean the target was reached: read the actual pose.
        """
        x, y, z = map(float, location)
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError('Swept movement requires a finite target')
        command = (
            f'vbp {name} K2_SetActorLocation '
            f'(X={x:.9f},Y={y:.9f},Z={z:.9f}) true false'
        )
        with self.lock:
            response = self.client.request(command)
        try:
            result = json.loads(response)
            if str(result['ReturnValue']).lower() not in {'true', 'false'}:
                raise ValueError('Invalid native return value')
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f'Runtime does not acknowledge swept movement: {response!r}'
            ) from exc
        return tuple(float(v) for v in self.get_location(name))

    def __init__(
        self,
        port: int = 9000,
        ip: str = '127.0.0.1',
        resolution=(1280, 720),
    ):
        super().__init__(port, ip, resolution=resolution)

    def rt_agent_move_to(self, name, x, y, time):
        cmd = f"vbp {name} MoveTo {x},{y} {time}"
        with self.lock:
            self.client.request(cmd)
    
    def rt_agent_move_to_waypoint(self, name, waypoint_x, waypoint_y):
        cmd = f"vbp {name} MoveToPoint {waypoint_x},{waypoint_y}"
        with self.lock:
            self.client.request(cmd)

    def rt_agent_get_overlap_type(self, name):
        cmd = f"vbp {name} GetOverlapType"
        with self.lock:
            return self.client.request(cmd)

    def rt_set_game_speed(self, scale):
        cmd = f"vrun slomo {scale}"
        with self.lock:
            return self.client.request(cmd)

    def rt_scooter_set_path(self, name, path):
        cmd = f"vbp {name} SetPath {path}"
        with self.lock:
            return self.client.request(cmd)

    def rt_scooter_follow_path(self, name):
        cmd = f"vbp {name} MovementFollowPath"
        with self.lock:
            return self.client.request(cmd)

    def rt_set_pedestrian_speed(self, name, speed):
        cmd = f"vbp {name} SetMaxSpeed {speed}"
        with self.lock:
            return self.client.request(cmd)
    
    def activate_object_movement(self, name):
        cmd = f"vbp {name} ActivateMovement"
        with self.lock:
            return self.client.request(cmd)

    def rt_get_states(self, name):
        cmd = f"vbp {name} GetStates"
        with self.lock:
            return self.client.request(cmd)
