"""Vehicle agent module for simulating vehicles in traffic."""
from enum import Enum, auto

import numpy as np

from simworld.agent.base_agent import BaseAgent
from simworld.traffic.ad_algorithm.pid_controller import PIDController
from simworld.utils.traffic_utils import cal_waypoints
from simworld.utils.vector import Vector


class VehicleState(Enum):
    """Enumeration of possible vehicle states."""
    WAITING = auto()  # waiting for making u-turn
    MAKING_U_TURN = auto()  # making u-turn
    MOVING = auto()  # moving
    STOPPED = auto()  # stopped for red light or avoiding collision


class Vehicle(BaseAgent):
    """Vehicle agent for traffic simulation."""

    _id_counter = 0

    def __init__(self, position: Vector, direction: Vector, current_lane, vehicle_reference: str, config, length: float = 500, width: float = 200):
        """Initialize a vehicle agent.

        Args:
            position: Initial position vector.
            direction: Initial direction vector.
            current_lane: Lane where the vehicle starts.
            vehicle_reference: Reference identifier for the vehicle.
            config: Configuration dictionary.
            length: Vehicle length in units.
            width: Vehicle width in units.
        """
        super().__init__(position, direction)
        self.id = Vehicle._id_counter
        Vehicle._id_counter += 1

        self.config = config

        # movement attributes
        self.current_lane = current_lane
        self.waypoints = []

        self.state = VehicleState.STOPPED

        self.steering_pid = PIDController(k_p=self.config['traffic.vehicle.steering_pid.kp'], k_i=self.config['traffic.vehicle.steering_pid.ki'], k_d=self.config['traffic.vehicle.steering_pid.kd'])

        # vehicle attributes
        self.vehicle_reference = vehicle_reference
        self.length = length
        self.width = width
        self.max_steering = self.config['traffic.vehicle.max_steering']

        self.throttle = 0
        self.brake = 0
        self.steering = 0
        # Human-readable diagnostic populated by VehicleManager.  Keeping the
        # reason separate from STOPPED matters because a red signal, a leading
        # actor, and a road obstacle require different recovery behaviour.
        self.stop_reason = None

    @classmethod
    def reset_id_counter(cls):
        """Reset the vehicle ID counter to zero."""
        cls._id_counter = 0

    def __str__(self):
        """Return a string representation of the vehicle."""
        return f'Vehicle(id={self.id}, position={self.position}, direction={self.direction}, yaw={self.yaw})'

    def __repr__(self):
        """Return a detailed string representation of the vehicle."""
        return f'Vehicle(id={self.id}, current_lane={self.current_lane.id}, position={self.position}, direction={self.direction}, yaw={self.yaw}, waypoints={self.waypoints})'

    def compute_control(self, waypoint, dt):
        """Compute throttle, brake, steering.

        Args:
            waypoint: Target waypoint to calculate control values.
            dt: Time delta for PID controller.

        Returns:
            tuple: Throttle, brake, steering values, and a boolean indicating control change.
        """
        target_x, target_y = waypoint.x, waypoint.y

        # Compute target yaw
        delta_x = target_x - self.position.x
        delta_y = target_y - self.position.y
        target_yaw = np.degrees(np.arctan2(delta_y, delta_x))

        # Compute yaw error
        yaw_error = target_yaw - self.yaw
        if yaw_error > 180:
            yaw_error -= 360
        elif yaw_error < -180:
            yaw_error += 360

        # Calculate normal distance to lane
        lane_direction = self.current_lane.direction
        vehicle_direction = self.direction
        lane_start = self.current_lane.start
        vehicle_to_lane = self.position - lane_start
        normal_distance = abs(vehicle_to_lane.cross(lane_direction))

        angle_diff = abs(np.degrees(np.arccos(lane_direction.dot(vehicle_direction))))

        # Consider both angle difference and normal distance
        if angle_diff < 3 and normal_distance < self.config['traffic.vehicle.lane_deviation']:
            steering = 0
            self.steering_pid.reset()
        else:
            if abs(yaw_error) < 1:
                yaw_error = 0
            steering = self.steering_pid.update(yaw_error, dt)
            steering_limit = self.max_steering
            if abs(yaw_error) >= 20 or normal_distance >= 100:
                steering_limit = max(steering_limit, 0.85)
            steering = np.clip(steering, -steering_limit, steering_limit)

        # Entering an intersection turn at the old fixed 0.4 throttle lets the
        # physics vehicle overshoot several Bezier samples before the next
        # control tick.  Scale the throttle with the actual heading error so a
        # sharp turn is tracked rather than cut across the kerb.
        abs_yaw_error = abs(yaw_error)
        # Lateral displacement matters even after most of the yaw error has
        # been corrected.  Accelerating back to 0.4 while still two metres off
        # the outgoing centreline was enough to carry a car toward curb trees.
        if abs_yaw_error >= 45 or normal_distance >= 200:
            throttle = 0.10
        elif abs_yaw_error >= 20 or normal_distance >= 100:
            throttle = 0.20
        elif steering != 0:
            throttle = 0.4
        else:
            throttle = 1
        brake = 0

        # if no changes, don't update
        if throttle == self.throttle and brake == self.brake and steering == self.steering:
            return 0, 0, 0, False

        self.throttle = throttle
        self.brake = brake
        self.steering = steering

        return throttle, brake, steering, True

    def advance_route_waypoints(
        self,
        reached_distance: float = 20.0,
        lookahead_distance: float = 200.0,
    ):
        """Advance to the spatially nearest remaining route segment.

        A vehicle can overshoot more than one densely sampled turn point in a
        physics tick.  Treating every point behind the vehicle as passed can
        then delete an entire curved route while the vehicle is laterally off
        it, leaving only a distant endpoint and causing a kerb-cut.  Anchor the
        remaining route at its nearest sample and select a short look-ahead
        point along the route.  This is a small pure-pursuit controller: it
        recovers toward the curve while always aiming forward along it instead
        of steering back to an already passed sample.
        """
        if not self.waypoints:
            return None

        distances = [self.position.distance(point) for point in self.waypoints]
        nearest_index = int(np.argmin(distances))
        if nearest_index:
            del self.waypoints[:nearest_index]

        while self.waypoints:
            to_waypoint = self.waypoints[0] - self.position
            distance = to_waypoint.length()
            if distance <= reached_distance:
                self.waypoints.pop(0)
                continue

            # A lone point behind the car is complete.  Multi-point routes keep
            # their nearest anchor; the look-ahead target below is deliberately
            # non-destructive so repeated control ticks cannot consume the path
            # while the physics actor has not moved.
            if (
                len(self.waypoints) == 1
                and self.direction.dot(to_waypoint.normalize()) < 0
            ):
                self.waypoints.pop(0)
            break

        if not self.waypoints:
            return None

        target_index = 0
        route_distance = 0.0
        while (
            target_index + 1 < len(self.waypoints)
            and route_distance < lookahead_distance
        ):
            route_distance += self.waypoints[target_index].distance(
                self.waypoints[target_index + 1]
            )
            target_index += 1
        return self.waypoints[target_index]

    def is_close_to_end(self):
        """Check if the vehicle is close to the end of the current lane.

        Returns:
            bool: True if vehicle is close to the end of the lane.
        """
        return self.position.distance(self.current_lane.end) < self.config['traffic.vehicle.distance_to_end'] + self.length / 2

    def _planned_path_proximity(self, point, planned_waypoints, lookahead):
        """Return ``(path_distance, lateral_distance)`` to a nearby point.

        Dynamic actors at a junction must be tested against the route the car
        will actually drive.  A wide heading cone labels cars waiting on an
        adjacent/conflicting approach as blockers even when their centers are
        several metres away from the planned turn, which can deadlock an
        otherwise signal-safe intersection.
        """
        path_points = [self.position]
        if planned_waypoints:
            path_points.extend(planned_waypoints)
        if len(path_points) == 1:
            path_points.append(self.position + self.direction * lookahead)

        best = None
        distance_before = 0.0
        remaining = lookahead
        for start, raw_end in zip(path_points, path_points[1:]):
            segment = raw_end - start
            segment_length = segment.length()
            if segment_length <= 1e-6:
                continue
            direction = segment.normalize()
            usable_length = min(segment_length, remaining)
            projection = max(
                0.0,
                min(usable_length, (point - start).dot(direction)),
            )
            nearest = start + direction * projection
            candidate = (
                distance_before + projection,
                point.distance(nearest),
            )
            if best is None or candidate[1] < best[1]:
                best = candidate
            distance_before += usable_length
            remaining -= usable_length
            if remaining <= 1e-6:
                break
        return best

    def is_close_to_object(self, vehicles, pedestrians, planned_waypoints=None):
        """Detect objects in the vehicle's path.

        Args:
            vehicles: List of vehicles to check for proximity.
            pedestrians: List of pedestrians to check for proximity.

        Returns:
            bool: True if the vehicle is close to any object.
        """
        PEDESTRIAN_DETECTION_DISTANCE = self.config['traffic.distance_between_objects'] + self.length / 2

        # Check vehicles
        for vehicle in vehicles:
            VEHICLE_DETECTION_DISTANCE = 1.5 * self.config['traffic.distance_between_objects'] + self.length / 2 + vehicle.length / 2
            if vehicle.id == self.id:
                continue

            distance = self.position.distance(vehicle.position)
            if distance > VEHICLE_DETECTION_DISTANCE:
                continue
            proximity = self._planned_path_proximity(
                vehicle.position,
                planned_waypoints,
                VEHICLE_DETECTION_DISTANCE,
            )
            if proximity is None:
                continue
            path_distance, lateral_distance = proximity
            swept_half_width = self.width / 2.0 + vehicle.width / 2.0 + 60.0
            if (
                path_distance > self.length / 2.0
                and lateral_distance <= swept_half_width
            ):
                return True

        # Check pedestrians
        for pedestrian in pedestrians:
            distance = self.position.distance(pedestrian.position)
            if distance > PEDESTRIAN_DETECTION_DISTANCE:
                continue
            proximity = self._planned_path_proximity(
                pedestrian.position,
                planned_waypoints,
                PEDESTRIAN_DETECTION_DISTANCE,
            )
            if proximity is None:
                continue
            path_distance, lateral_distance = proximity
            if (
                path_distance > self.length / 2.0
                and lateral_distance <= self.width / 2.0 + 100.0
            ):
                return True

        return False

    def nearest_static_obstacle_ahead(self, obstacles, planned_waypoints=None):
        """Return the closest mapped obstacle occupying this vehicle's path.

        The original traffic manager only sensed vehicles and pedestrians, so
        a managed car would continue driving into street furniture or debris.
        This detector works in lane coordinates: an object must be ahead of the
        bumper, within the configured look-ahead distance, and overlap the
        vehicle's swept width.  It deliberately does not treat objects behind
        or on the opposite sidewalk as hazards.
        """
        if not obstacles:
            return None

        lookahead = float(
            self.config.get(
                'traffic.vehicle.static_obstacle_detection_distance',
                800.0,
            )
        )
        lateral_margin = float(
            self.config.get(
                'traffic.vehicle.static_obstacle_lateral_margin',
                120.0,
            )
        )
        swept_half_width = self.width / 2.0 + lateral_margin
        # Follow the actual route rather than extending the current heading to
        # infinity.  The latter makes a turning car brake for curb furniture
        # that is visually ahead but not on its intended path.
        path_points = [self.position]
        if planned_waypoints:
            path_points.extend(planned_waypoints)
        if len(path_points) == 1:
            path_points.append(self.position + self.direction * lookahead)

        path_segments = []
        remaining = lookahead
        distance_from_vehicle = 0.0
        for start, end in zip(path_points, path_points[1:]):
            segment = end - start
            segment_length = segment.length()
            if segment_length <= 1e-6:
                continue
            if segment_length > remaining:
                end = start + segment.normalize() * remaining
                segment_length = remaining
            path_segments.append(
                (start, end, segment.normalize(), segment_length, distance_from_vehicle)
            )
            distance_from_vehicle += segment_length
            remaining -= segment_length
            if remaining <= 1e-6:
                break

        closest = None
        closest_path_distance = float('inf')
        for obstacle in obstacles:
            try:
                obstacle_position = Vector(
                    float(obstacle['x']),
                    float(obstacle['y']),
                )
            except (KeyError, TypeError, ValueError):
                continue
            for start, end, direction, segment_length, distance_before in path_segments:
                offset = obstacle_position - start
                projection = max(0.0, min(segment_length, offset.dot(direction)))
                nearest = start + direction * projection
                lateral_distance = obstacle_position.distance(nearest)
                path_distance = distance_before + projection
                if path_distance <= self.length / 2.0:
                    continue
                if lateral_distance > swept_half_width:
                    continue
                if path_distance < closest_path_distance:
                    closest_path_distance = path_distance
                    closest = obstacle
                break
        return closest

    def change_to_next_lane(self, next_lane):
        """Change the vehicle's current lane to the next lane.

        Args:
            next_lane: The lane to change to.
        """
        self.current_lane.vehicles.remove(self)
        self.current_lane = next_lane
        self.current_lane.vehicles.append(self)

        waypoints = cal_waypoints(next_lane.start, next_lane.end, self.config['traffic.gap_between_waypoints'])
        self.add_waypoint(waypoints)

    def add_waypoint(self, waypoint: list[Vector]):
        """Add waypoints to the vehicle's path.

        Args:
            waypoint: List of waypoint vectors to add.
        """
        self.waypoints.extend(waypoint)

    def pop_waypoint(self):
        """Remove and return the first waypoint.

        Returns:
            Vector: The first waypoint.
        """
        return self.waypoints.pop(0)

    def set_attributes(self, throttle: float, brake: float, steering: float):
        """Set the vehicle's control attributes.

        Args:
            throttle: Throttle value.
            brake: Brake value.
            steering: Steering value.
        """
        self.throttle = throttle
        self.brake = brake
        self.steering = steering

    def get_attributes(self):
        """Get the vehicle's current control attributes.

        Returns:
            tuple: Current throttle, brake, and steering values.
        """
        return self.throttle, self.brake, self.steering

    def completed_u_turn(self):
        """Check if the vehicle has completed a U-turn.

        Returns:
            bool: True if the U-turn is completed.
        """
        # Calculate angle between vehicle direction and lane direction
        angle_diff = abs(np.degrees(np.arccos(self.direction.dot(self.current_lane.direction))))
        return angle_diff < 15
