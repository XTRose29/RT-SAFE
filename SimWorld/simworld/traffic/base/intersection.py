"""Intersection module for representing and managing intersections in the simulation.

This module defines the intersection class which is responsible for managing the connections
between roads, lanes, sidewalks, and controlling traffic signals.
"""
import math
from typing import List

from simworld.traffic.base.crosswalk import Crosswalk
from simworld.traffic.base.road import Road
from simworld.traffic.base.traffic_lane import TrafficLane
from simworld.traffic.base.traffic_signal import (TrafficSignal,
                                                  TrafficSignalState)
from simworld.utils.vector import Vector


class Intersection:
    """Represents a road intersection in the simulation.

    Manages the connection of multiple roads, including lanes, sidewalks, and traffic signals.
    Handles the traffic flow logic at the intersection.
    """
    _id_counter = 0

    def __init__(self, center: Vector, roads: List[Road]):
        """Initialize an intersection.

        Args:
            center: The center point of the intersection.
            roads: List of roads connected to this intersection.
        """
        self.id = Intersection._id_counter
        Intersection._id_counter += 1
        self.center = center
        self.roads = roads  # roads connected to the intersection

        self.traffic_lights: List[TrafficSignal] = []  # [traffic_light]
        self.pedestrian_lights: List[TrafficSignal] = []  # [pedestrian_light]
        self.crosswalks = []  # [crosswalk]

        # number of sidewalks should be twice of the number of lanes
        self.lane_mapping = {}  # {incoming_lane: [outgoing_lane]}
        self.sidewalk_mapping = {}  # {sidewalk: [(sidewalk, crosswalk) or (sidewalk, None)]}

        # for traffic light cycle
        self.cycle_count = 0    # 0 for the pedestrian green light
        self.total_lights = 0
        self.vehicle_movement_groups: List[List[TrafficSignal]] = []
        self.active_vehicle_group_index = None
        self.active_vehicle_group_seen_green = False
        self.active_vehicle_group_elapsed_s = 0.0

        self.is_light_changing = False

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter for intersections.

        Used when resetting the simulation to ensure IDs start from 0.
        """
        cls._id_counter = 0

    def __repr__(self):
        """Return a string representation of the intersection.

        Returns:
            A string containing the intersection's attributes.
        """
        return f'Intersection(id={self.id}, center={self.center}, lanes={self.lane_mapping})'

    def init_intersection(self, config):
        """Initialize the intersection with configuration.

        Sets up lane connections, sidewalk connections, and traffic signals.

        Args:
            config: Configuration dictionary with simulation parameters.
        """
        self.connect_lanes(config)
        self.connect_sidewalks(config)
        self.add_traffic_signals(config)

    def add_traffic_signals(self, config):
        """Add traffic signals to the intersection.

        One signal for each road, controlling both lanes and crosswalks on that road.

        Args:
            config: Configuration dictionary with traffic signal parameters.

        Raises:
            ValueError: If there are no lanes to add traffic signals to.
        """
        if len(self.lane_mapping) == 0:
            raise ValueError('No lanes to add traffic light')

        if len(self.lane_mapping) == 1:  # u-turn intersection
            return

        # Sort incoming lanes clockwise based on their angle from the center
        sorted_lanes = sorted(
            self.lane_mapping.keys(),
            key=lambda lane: (
                -math.atan2(
                    lane.start.y - self.center.y,
                    lane.start.x - self.center.x
                ) % (2 * math.pi)
            )
        )

        # Add traffic lights in clockwise order
        for incoming_lane in sorted_lanes:
            # In UE, Y axis is inverted. The normal direction is on the left side of the direction vector
            incoming_lane_normal_direction = Vector(incoming_lane.direction.y, -incoming_lane.direction.x)
            traffic_light_center = self.center + (incoming_lane_normal_direction * config['traffic.traffic_signal.light_normal_offset']) + (incoming_lane.direction * config['traffic.traffic_signal.light_radial_offset'])

            dir_traffic_light = Vector(-incoming_lane.direction.x, -incoming_lane.direction.y)

            for crosswalk in self.crosswalks:
                if crosswalk.road_id == incoming_lane.road_id:
                    self.traffic_lights.append(TrafficSignal(traffic_light_center, dir_traffic_light, incoming_lane.id, crosswalk.id, 'both'))
                    break

        # Put a dedicated pedestrian head at both ends of every rendered zebra.
        #
        # The former three-way/two-way heuristics mirrored vehicle heads around
        # the intersection centre.  On a T junction that can leave a route
        # crosswalk with no nearby pedestrian head at all, while an unrelated
        # head tens of metres away is treated as its visual signal.  Endpoint
        # placement makes the rendered asset, crosswalk ID, and route geometry
        # describe the same physical crossing on every intersection shape.
        endpoint_offset = float(
            config.get(
                'traffic.traffic_signal.pedestrian_light_endpoint_offset',
                100,
            )
        )
        lateral_offset = float(
            config.get(
                'traffic.traffic_signal.pedestrian_light_lateral_offset',
                450,
            )
        )
        for crosswalk in sorted(self.crosswalks, key=lambda item: item.id):
            crossing_direction = (crosswalk.end - crosswalk.start).normalize()
            if crossing_direction == Vector(0, 0):
                continue
            crossing_normal = Vector(
                crossing_direction.y,
                -crossing_direction.x,
            )

            endpoint_heads = (
                (
                    (
                        crosswalk.start
                        - crossing_direction * endpoint_offset
                        + crossing_normal * lateral_offset
                    ),
                    crossing_direction,
                ),
                (
                    (
                        crosswalk.end
                        + crossing_direction * endpoint_offset
                        - crossing_normal * lateral_offset
                    ),
                    crossing_direction * -1,
                ),
            )
            for position, direction in endpoint_heads:
                self.pedestrian_lights.append(
                    TrafficSignal(
                        position,
                        direction,
                        None,
                        crosswalk.id,
                        'pedestrian',
                    )
                )

        # Count only 'both' type traffic lights
        self.total_lights = len(self.traffic_lights)
        self._build_vehicle_movement_groups()

    def _pedestrian_head_direction(self, position: Vector) -> Vector:
        """Face pedestrian heads toward sidewalk approaches."""
        outward = position - self.center
        if outward.length() <= 1e-9:
            return Vector(1.0, 0.0)
        return outward.normalize()

    def _build_vehicle_movement_groups(self, parallel_threshold=0.95):
        """Group non-conflicting, opposing approaches into vehicle phases.

        The packaged RT12 map exposes approach directions but no protected-turn
        arrows or detector plan.  Collinear heads therefore form one permissive
        two-way movement group, while perpendicular approaches stay in separate
        groups.  This mirrors a conventional two-way through phase without
        inventing protected turn phases that the map cannot render.
        """
        groups = []
        group_axes = []
        for light in self.traffic_lights:
            direction = light.direction.normalize()
            group_index = None
            for index, axis in enumerate(group_axes):
                if abs(direction.dot(axis)) >= parallel_threshold:
                    group_index = index
                    break
            if group_index is None:
                group_index = len(groups)
                groups.append([])
                group_axes.append(direction)
            light.vehicle_group = group_index
            groups[group_index].append(light)

        self.vehicle_movement_groups = groups

    def get_vehicle_movement_groups(self):
        """Return deterministic vehicle movement groups for this intersection."""
        if not self.vehicle_movement_groups and self.traffic_lights:
            self._build_vehicle_movement_groups()
        return self.vehicle_movement_groups

    def connect_lanes(self, config):
        """Connect lanes at the intersection.

        Creates mappings from incoming lanes to outgoing lanes.

        Args:
            config: Configuration dictionary with lane parameters.
        """
        # Make a U-turn at the intersection
        if len(self.roads) == 1:
            for i in range(config['traffic.num_lanes']):
                incoming_lane = self.roads[0].lanes['forward' + str(i+1)]
                outgoing_lane = self.roads[0].lanes['backward' + str(i+1)]
                self.lane_mapping[incoming_lane] = [outgoing_lane]
        else:
            # Forbid U-turn at the intersection
            # Handle multiple roads intersection
            for lane_level in range(config['traffic.num_lanes']):
                lane_num = lane_level + 1
                # Collect all incoming and outgoing lanes at this level
                incoming_lanes = []
                outgoing_lanes = []

                for road in self.roads:
                    # Add forward lane (incoming) if end is closer to intersection
                    forward_lane = road.lanes[f'forward{lane_num}']
                    if forward_lane.end.distance(self.center) < forward_lane.start.distance(self.center):
                        incoming_lanes.append((road.id, forward_lane))
                    else:
                        outgoing_lanes.append((road.id, forward_lane))

                    # Add backward lane (incoming) if end is closer to intersection
                    backward_lane = road.lanes[f'backward{lane_num}']
                    if backward_lane.end.distance(self.center) < backward_lane.start.distance(self.center):
                        incoming_lanes.append((road.id, backward_lane))
                    else:
                        outgoing_lanes.append((road.id, backward_lane))

                # Connect each incoming lane to all possible outgoing lanes from different roads
                for incoming_road_id, incoming_lane in incoming_lanes:
                    valid_outgoing_lanes = [
                        out_lane for out_road_id, out_lane in outgoing_lanes
                        if out_road_id != incoming_road_id  # Exclude lanes from the same road
                    ]
                    self.lane_mapping[incoming_lane] = valid_outgoing_lanes

    def connect_sidewalks(self, config):
        """Connect sidewalks at the intersection.

        Creates mappings between sidewalks and determines crosswalk connections.

        Args:
            config: Configuration dictionary with sidewalk parameters.

        Raises:
            ValueError: If a road has an unexpected number of crosswalks.
        """
        # Collect all sidewalks
        all_sidewalks = []
        for road in self.roads:
            all_sidewalks.extend([road.sidewalks['forward'], road.sidewalks['backward']])
            # Only add the crosswalk that is closer to this intersection
            if len(road.crosswalks) == 2:
                crosswalk1, crosswalk2 = road.crosswalks
                # Calculate the minimum distance of each crosswalk to the intersection center
                dist1 = min(crosswalk1.start.distance(self.center), crosswalk1.end.distance(self.center))
                dist2 = min(crosswalk2.start.distance(self.center), crosswalk2.end.distance(self.center))
                # Add the closer crosswalk
                self.crosswalks.append(crosswalk1 if dist1 < dist2 else crosswalk2)
            else:
                raise ValueError(f'Road {road.id} has {len(road.crosswalks)} crosswalks')

        # Calculate maximum connection distance
        max_distance = math.sqrt((2 * config['traffic.sidewalk_offset'])**2 + (2 * config['traffic.crosswalk_offset'])**2)

        # For each sidewalk, find its connections to other sidewalks and crosswalks
        for sidewalk in all_sidewalks:
            # Get the point closer to intersection center
            start_dist = sidewalk.start.distance(self.center)
            end_dist = sidewalk.end.distance(self.center)
            incoming_point = sidewalk.end if end_dist < start_dist else sidewalk.start

            # Find valid connections to other sidewalks and crosswalks
            valid_connections = []
            for other_sidewalk in all_sidewalks:
                if other_sidewalk == sidewalk:
                    continue

                # Get the point of other sidewalk closer to intersection center
                other_start_dist = other_sidewalk.start.distance(self.center)
                other_end_dist = other_sidewalk.end.distance(self.center)
                other_point = other_sidewalk.end if other_end_dist < other_start_dist else other_sidewalk.start

                if incoming_point == other_point:
                    valid_connections.append((other_sidewalk, None))
                    continue

                # If points are within range, add to valid connections with no crosswalk
                # 200 is a magic number to avoid error when calculating float
                if incoming_point.distance(other_point) < max_distance - 200:
                    for crosswalk in self.crosswalks:
                        if (crosswalk.start == incoming_point or crosswalk.end == incoming_point) and \
                           (crosswalk.start == other_point or crosswalk.end == other_point):
                            valid_connections.append((other_sidewalk, crosswalk))
                            break

            self.sidewalk_mapping[sidewalk] = valid_connections

    def all_traffic_lights_red(self):
        """Check if all traffic lights at the intersection are red.

        Returns:
            True if all traffic lights are red, False otherwise.
        """
        if len(self.traffic_lights) == 0:
            return False

        for traffic_light in self.traffic_lights:
            if not traffic_light.get_state() == (TrafficSignalState.VEHICLE_RED, TrafficSignalState.PEDESTRIAN_RED):
                return False
        return True

    def get_traffic_light_state(self, incoming_lane: TrafficLane):
        """Get the state of the traffic light for a specific lane.

        Args:
            incoming_lane: The lane to get the traffic light state for.

        Returns:
            The state of the traffic light for the lane.
        """
        for traffic_light in self.traffic_lights:
            if traffic_light.lane_id == incoming_lane.id:
                return traffic_light.get_state()
        return None

    def get_crosswalk_light_state(self, crosswalk: Crosswalk):
        """Get the state of the traffic light for a specific crosswalk.

        Args:
            crosswalk: The crosswalk to get the traffic light state for.

        Returns:
            A tuple of (state, left_time) for the crosswalk.
        """
        for traffic_light in self.traffic_lights:
            if traffic_light.crosswalk_id == crosswalk.id:
                state = traffic_light.get_state()[1]
                left_time = traffic_light.get_left_time()
                return state, left_time

        # If no traffic light is found, return the pedestrian green light
        return TrafficSignalState.PEDESTRIAN_GREEN, 20

    def has_completed_cycle(self):
        """Check if all traffic lights have cycled through green once.

        Returns:
            True if a complete cycle has been completed, False otherwise.
        """
        total_phases = (
            len(self.vehicle_movement_groups)
            if self.vehicle_movement_groups
            else self.total_lights
        )
        if self.cycle_count >= total_phases:
            return True
        return False

    def reset_cycle_count(self):
        """Reset the cycle count when a light cycle is completed."""
        self.cycle_count = 0
        self.active_vehicle_group_index = None
        self.active_vehicle_group_seen_green = False
        self.active_vehicle_group_elapsed_s = 0.0

    def increment_cycle_count(self):
        """Increment the cycle count when a light turns green."""
        self.cycle_count += 1

    @property
    def lanes(self):
        """Get the lane mapping for this intersection.

        Returns:
            Dictionary mapping incoming lanes to outgoing lanes.
        """
        return self.lane_mapping

    @property
    def sidewalks(self):
        """Get the sidewalk mapping for this intersection.

        Returns:
            Dictionary mapping sidewalks to their connections.
        """
        return self.sidewalk_mapping
