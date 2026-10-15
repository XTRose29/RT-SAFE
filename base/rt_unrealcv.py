from simworld.communicator.unrealcv import UnrealCV
import json
import math

class RTUnrealCV(UnrealCV):
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
