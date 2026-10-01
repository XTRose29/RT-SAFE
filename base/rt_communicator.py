import json
import os
import random
import math
import time
import numpy as np
from simworld.communicator.communicator import Communicator
from base.rt_unrealcv import RTUnrealCV
from base.rt_traffic_system import (
    TrafficPhaseTiming,
    build_intersection_snapshot,
)

class RTCommunicator(Communicator):
    def __init__(self, rt_unrealcv: RTUnrealCV):
        super().__init__(rt_unrealcv)

    def set_pedestrians_waypoints(self, pedestrians):
        for pedestrian in pedestrians:
            self.p_set_waypoints(pedestrian.id, pedestrian.waypoints)

    def set_pedestrians_speed(self, pedestrians):
        for pedestrian in pedestrians:
            self.unrealcv.rt_set_pedestrian_speed(self.get_pedestrian_name(pedestrian.id), pedestrian.speed)

    def start_pedestrians_simulation(self, pedestrians):
        for pedestrian in pedestrians:
            self.unrealcv.p_movement_simulation(self.get_pedestrian_name(pedestrian.id))

    def start_scooters_simulation(self, scooters):
        for scooter in scooters:
            self.unrealcv.rt_scooter_follow_path(self.get_scooter_name(scooter.id))

    def set_scooters_path(self, scooters):
        for scooter in scooters:
            name = self.get_scooter_name(scooter.id)
            str_waypoints = ''
            for waypoint in scooter.path:
                str_waypoints += f'{waypoint.x},{waypoint.y};'
            str_waypoints = str_waypoints[:-1]
            self.unrealcv.rt_scooter_set_path(name, str_waypoints)

    def rt_agent_move_to(self, name, waypoint, time):
        self.unrealcv.rt_agent_move_to(name, waypoint.x, waypoint.y, time)
        # self.unrealcv.rt_agent_move_to_waypoint(name, waypoint.x, waypoint.y)

    def rt_agent_turn_around(self, name, angle, clockwise):
        # p_rotate 'right' increases UE yaw. UE is left-handed, so increasing
        # yaw turns the rendered view to the right: R (clockwise) maps to
        # 'right' and L to 'left', matching what the policy sees.
        self.unrealcv.p_rotate(name, angle, 'right' if clockwise else 'left')   # rotate takes 1 sim second

    def rt_agent_adjust_speed(self, name, speed):
        self.unrealcv.humanoid_set_speed(name, speed)

    def get_overlap_type(self, name):
        """Get overlap type.
        Args:
            name: Name of the agent.

        Returns:
            Overlap type (str). 0 for no overlap, 1 for fall, 2 for stall, 3 for slip.
        """
        response = self.unrealcv.rt_agent_get_overlap_type(name)
        overlap_data = json.loads(response)
        return int(overlap_data['overlap_type'])

    def set_game_speed(self, scale):
        self.unrealcv.rt_set_game_speed(scale) 

    def get_states(self, name):
        """Get states.

        Args:
            name: Name of the agent.

        Returns:
            Human collision number, object collision number, building collision number, collision impulse, overlap type, touched road.
        """
        states_json = self.unrealcv.rt_get_states(name)
        states_data = json.loads(states_json)
        human_collision_num = int(states_data['HumanCollision'])
        object_collision_num = int(states_data['ObjectCollision'])
        building_collision_num = int(states_data['BuildingCollision'])
        collision_impulse = float(states_data['Intensity'])
        overlap_type = int(states_data['OverlapType'])
        touched_road = int(states_data['TouchedRoad'])  # 0 for no, 1 for yes
        return human_collision_num, object_collision_num, building_collision_num, collision_impulse, overlap_type, touched_road

    def get_vehicle_collision_number(self, name):
        """Return the UE vehicle-collision counter for an RT agent.

        ``GetStates`` in the released RT agent Blueprint predates vehicle
        traffic and does not expose ``VehicleCollision``. The inherited
        ``GetCollisionNum`` Blueprint API does, so keep this as a separate
        query rather than changing the long-standing ``get_states`` tuple.
        """
        collision_json = self.unrealcv.get_collision_num(name)
        collision_data = json.loads(collision_json)
        return int(collision_data.get('VehicleCollision', 0))

    def spawn_pedestrians(self, pedestrians, type=2):
        """Spawn pedestrians.

        Args:
            pedestrians: List of pedestrian objects.
            type: Type of pedestrian. 0 for dog, 1 for humanoid, 2 for pedestrian.
        """
        mutation_delay = max(
            0.0,
            float(os.environ.get('SIMWORLD_ACTOR_MUTATION_SETTLE_SECONDS', '0')),
        )

        def settle():
            if mutation_delay:
                time.sleep(mutation_delay)

        for pedestrian in pedestrians:
            if type == 0:
                model_path = '/Game/RealTimeBench/Agent/RT_Base_Dog.RT_Base_Dog_C'
            elif type == 1:
                model_path = '/Game/RealTimeBench/Agent/RT_Base_Humanoid.RT_Base_Humanoid_C'
            elif type == 2:
                model_path = '/Game/RealTimeBench/Agent/RT_Base_Pedestrian.RT_Base_Pedestrian_C'

            name = self.get_pedestrian_name(pedestrian.id)
            self.unrealcv.spawn_bp_asset(model_path, name)
            settle()
            # Convert 2D position to 3D (x,y -> x,y,z)
            location_3d = (
                pedestrian.position.x,  # Unreal X = 2D Y
                pedestrian.position.y,  # Unreal Y = 2D X
                110  # Z coordinate (ground level)
            )
            # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
            orientation_3d = (
                0,  # Pitch
                math.degrees(math.atan2(pedestrian.direction.y, pedestrian.direction.x)),  # Yaw
                0  # Roll
            )
            self.unrealcv.set_location(location_3d, name)
            settle()
            self.unrealcv.set_orientation(orientation_3d, name)
            settle()
            self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
            settle()
            self.unrealcv.set_collision(name, True)
            settle()
            self.unrealcv.set_movable(name, True)
            settle()


    def clear_agents(self):
        """Clear all agents in the environment."""
        objects = [
            obj.decode('utf-8', errors='replace')
            if isinstance(obj, (bytes, bytearray, np.bytes_))
            else str(obj)
            for obj in self.unrealcv.get_objects()
        ]
        for object_name in objects:
            if object_name.casefold().startswith('rt_'):
                self.unrealcv.destroy(object_name)

        self.unrealcv.clean_garbage()

    def get_pedestrian_name(self, id):
        return 'RT_PEDESTRIAN_' + str(id)

    def get_traffic_signal_name(self, id):
        return 'RT_TRAFFIC_SIGNAL_' + str(id)

    def get_traffic_signal_state(self, signal_id):
        """Return the state rendered by an RT traffic-light Blueprint.

        This intentionally queries UE on demand. The RT intersection Blueprint
        owns the phase timer, so the Python ``TrafficSignal.state`` value is not
        authoritative unless it has just been refreshed from UE.
        """
        object_name = self.get_traffic_signal_name(signal_id)
        raw_response = self.unrealcv.tl_get_state(object_name)
        payload = json.loads(raw_response)

        def parse_bool(value):
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() == 'true'

        remaining_time = payload.get('ped time')
        try:
            remaining_time = float(remaining_time)
        except (TypeError, ValueError):
            remaining_time = None

        return {
            'signal_id': signal_id,
            'object_name': object_name,
            'vehicle_green': parse_bool(payload.get('green light', False)),
            'pedestrian_walk': parse_bool(payload.get('ped green', False)),
            'remaining_time_s': remaining_time,
            'source': 'ue_blueprint:GetState',
            'raw': payload,
        }


    def set_traffic_signal_pedestrian_stop(self, signal_id):
        """Synchronize a legacy pedestrian head to DON'T WALK."""
        object_name = self.get_traffic_signal_name(signal_id)
        return self.unrealcv.tl_set_pedestrian_stop(object_name)

    def set_traffic_signal_pedestrian_walk(self, signal_id):
        """Keep a legacy pedestrian head at WALK during occupancy hold."""
        object_name = self.get_traffic_signal_name(signal_id)
        return self.unrealcv.tl_set_pedestrian_walk(object_name)

    def set_traffic_signal_vehicle_stop(self, signal_id):
        """Hold both vehicle faces at red on a legacy traffic head."""
        object_name = self.get_traffic_signal_name(signal_id)
        return self.unrealcv.tl_set_vehicle_red(object_name)

    def get_intersection_traffic_state(
        self,
        intersection,
        timing: TrafficPhaseTiming = None,
        timestamp_s: float = None,
    ):
        """Read every rendered head and return one intersection snapshot."""
        signals = list(intersection.traffic_lights) + list(
            intersection.pedestrian_lights
        )
        states = [self.get_traffic_signal_state(signal.id) for signal in signals]
        return build_intersection_snapshot(
            intersection,
            states,
            timing=timing,
            timestamp_s=timestamp_s,
        )

    def configure_intersection_phase_plan(
        self,
        intersection_name,
        timing: TrafficPhaseTiming,
    ):
        """Send the full phase plan to an upgraded intersection Blueprint."""
        return self.unrealcv.traffic_signal_configure_phase_plan(
            intersection_name,
            timing.vehicle_green_s,
            timing.vehicle_yellow_s,
            timing.all_red_s,
            timing.pedestrian_walk_s,
            timing.pedestrian_clearance_s,
        )

    def get_canonical_intersection_state(self, intersection_name):
        """Parse the native UE controller's whole-intersection state payload.

        UnrealCV's ``vbp`` bridge serializes every reflected output as a JSON
        string. Nested arrays therefore arrive as JSON-encoded strings and
        booleans arrive as ``"true"``/``"false"``. Decode both forms here so
        benchmark consumers see one stable typed schema.
        """
        raw_response = self.unrealcv.traffic_signal_get_intersection_state(
            intersection_name
        )
        reflected_payload = json.loads(raw_response)
        if not isinstance(reflected_payload, dict):
            raise ValueError('GetIntersectionState did not return a JSON object')

        # UnrealCV preserves most C++ output parameter names verbatim, but UE's
        # reflection metadata exports ``phase`` as ``Phase`` in packaged builds.
        # Treat reflected field names case-insensitively so the native controller
        # and Python agent share one stable schema.
        payload = {
            str(key).strip().lower(): value
            for key, value in reflected_payload.items()
        }
        required = {'phase', 'remaining_time_s', 'signal_states'}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                'GetIntersectionState missing required fields: '
                + ', '.join(missing)
            )
        def decode_nested_json(value, fallback):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return fallback
            return value

        def decode_bool(value, default=True):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {'true', '1', 'yes'}:
                    return True
                if normalized in {'false', '0', 'no'}:
                    return False
            return default

        violations = decode_nested_json(payload.get('violations', []), [])
        signal_states = decode_nested_json(payload['signal_states'], [])
        if not isinstance(violations, list):
            violations = []
        if not isinstance(signal_states, list):
            raise ValueError('GetIntersectionState signal_states is not a list')

        return {
            'schema_version': int(payload.get('schema_version', 1)),
            'intersection_id': payload.get('intersection_id'),
            'observed_phase': str(payload['phase']).upper(),
            'phase_remaining_time_s': float(payload['remaining_time_s']),
            'safe': decode_bool(payload.get('safe', True)),
            'violations': violations,
            'signal_states': signal_states,
            'source': 'ue_controller:GetIntersectionState',
            'capabilities': {
                'vehicle_green_exposed': True,
                'vehicle_yellow_exposed': True,
                'all_red_exposed': True,
                'pedestrian_walk_exposed': True,
                'pedestrian_clearance_exposed': True,
                'remaining_time_exposed': True,
            },
            'raw': reflected_payload,
            'pedestrian_occupied': decode_bool(
                payload.get('pedestrian_occupied', False),
                default=False,
            ),
            'clearance_extended': decode_bool(
                payload.get('clearance_extended', False),
                default=False,
            ),
            'clearance_extension_elapsed_s': float(
                payload.get('clearance_extension_elapsed_s', 0.0)
            ),
        }

    def set_intersection_pedestrian_occupancy(
        self,
        intersection_name,
        pedestrian_occupied,
    ):
        """Update the native controller's passive crosswalk detector."""
        return self.unrealcv.traffic_signal_set_pedestrian_occupancy(
            intersection_name,
            bool(pedestrian_occupied),
        )

    def get_scooter_name(self, id):
        return 'RT_SCOOTER_' + str(id)
    
    def spawn_scooters(self, scooters):
        """Spawn scooters.

        Args:
            scooters: List of scooter objects.
        """
        for scooter in scooters:
            # Scooter model path
            model_path = '/Game/TrafficSystem/Vehicle/Base_Scooter.Base_Scooter_C'
            
            name = self.get_scooter_name(scooter.id)
            self.unrealcv.spawn_bp_asset(model_path, name)
            # Convert 2D position to 3D (x,y -> x,y,z)
            location_3d = (
                scooter.position.x,  # Unreal X = 2D Y
                scooter.position.y,  # Unreal Y = 2D X
                110  # Z coordinate (ground level)
            )
            # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
            orientation_3d = (
                0,  # Pitch
                math.degrees(math.atan2(scooter.direction.y, scooter.direction.x)),  # Yaw
                0  # Roll
            )
            self.unrealcv.set_location(location_3d, name)
            self.unrealcv.set_orientation(orientation_3d, name)
            self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
            self.unrealcv.set_collision(name, True)
            self.unrealcv.set_movable(name, True)

    def activate_object_movement(self, name):
        return self.unrealcv.activate_object_movement(name)

    def spawn_object(self, object_name, model_path, position, direction):
        """Spawn object.

        Args:
            object_name: Object name.
            model_path: Model path.
            position: Position. Tuple (x, y, z).
            direction: Direction. Tuple (pitch, yaw, roll).
        """
        self.unrealcv.spawn_bp_asset(model_path, object_name)
        # Convert 2D position to 3D (x,y -> x,y,z)
        location_3d = (
            position[0],  # Unreal X = 2D Y
            position[1],  # Unreal Y = 2D X
            position[2]  # Z coordinate (ground level)
        )
        # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
        orientation_3d = (
            direction[0],  # Pitch
            direction[1],  # Yaw
            direction[2]  # Roll
        )
        self.unrealcv.set_location(location_3d, object_name)
        self.unrealcv.set_orientation(orientation_3d, object_name)
        self.unrealcv.set_scale((1, 1, 1), object_name)
        self.unrealcv.set_collision(object_name, True)
        self.unrealcv.set_movable(object_name, True)

    def spawn_traffic_signals(
        self,
        traffic_signals,
        traffic_light_model_path=(
            '/Game/city_props/BP/props/street_light/'
            'BP_street_light.BP_street_light_C'
        ),
        pedestrian_light_model_path=(
            '/Game/city_props/BP/props/street_light/'
            'BP_street_light_ped.BP_street_light_ped_C'
        ),
        traffic_light_scale=1.0,
        pedestrian_light_scale=1.0,
    ):
        """Spawn traffic signals.

        Args:
            traffic_signals: List of traffic signal objects to spawn.
            traffic_light_model_path: Path to the traffic light model asset.
            pedestrian_light_model_path: Path to the pedestrian signal light model asset.
        """
        mutation_delay = max(
            0.0,
            float(os.environ.get('SIMWORLD_ACTOR_MUTATION_SETTLE_SECONDS', '0')),
        )

        def settle():
            if mutation_delay:
                time.sleep(mutation_delay)

        for traffic_signal in traffic_signals:
            name = self.get_traffic_signal_name(traffic_signal.id)
            if traffic_signal.type == 'pedestrian':
                model_name = pedestrian_light_model_path
                actor_scale = float(pedestrian_light_scale)
            elif traffic_signal.type == 'both':
                model_name = traffic_light_model_path
                actor_scale = float(traffic_light_scale)
            else:
                raise ValueError(
                    f'Unsupported traffic signal type: {traffic_signal.type}'
                )
            self.unrealcv.spawn_bp_asset(model_name, name)
            settle()
            # Convert 2D position to 3D (x,y -> x,y,z)
            location_3d = (
                traffic_signal.position.x,
                traffic_signal.position.y,
                0  # Z coordinate (ground level)
            )
            # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
            orientation_3d = (
                0,  # Pitch
                math.degrees(math.atan2(traffic_signal.direction.y, traffic_signal.direction.x)),  # Yaw
                0  # Roll
            )
            self.unrealcv.set_location(location_3d, name)
            settle()
            self.unrealcv.set_orientation(orientation_3d, name)
            settle()
            self.unrealcv.set_scale(
                (actor_scale, actor_scale, actor_scale),
                name,
            )
            settle()
            self.unrealcv.set_collision(name, True)
            settle()
            self.unrealcv.set_movable(name, False)
            settle()


class RTSignalTrafficCommunicator(RTCommunicator):
    """Namespaced bridge for signal-controlled background traffic."""

    def get_vehicle_name(self, vehicle_id):
        if vehicle_id not in self.vehicle_id_to_name:
            self.vehicle_id_to_name[vehicle_id] = (
                f'RT_SIGNAL_VEHICLE_{vehicle_id}'
            )
        return self.vehicle_id_to_name[vehicle_id]

    def get_pedestrian_name(self, pedestrian_id):
        if pedestrian_id not in self.pedestrian_id_to_name:
            self.pedestrian_id_to_name[pedestrian_id] = (
                f'RT_SIGNAL_PEDESTRIAN_{pedestrian_id}'
            )
        return self.pedestrian_id_to_name[pedestrian_id]

    def spawn_pedestrians(
        self,
        pedestrians,
        model_path='/Game/TrafficSystem/Pedestrian/'
        'Base_Pedestrian.Base_Pedestrian_C',
    ):
        return Communicator.spawn_pedestrians(
            self,
            pedestrians,
            model_path,
        )

    def spawn_ue_manager(self, ue_manager_path):
        self.ue_manager_name = 'RT_SIGNAL_UE_MANAGER'
        self.unrealcv.spawn_bp_asset(ue_manager_path, self.ue_manager_name)
