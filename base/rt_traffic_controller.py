from simworld.traffic.controller.traffic_controller import TrafficController
from simworld.config import Config
from simworld.utils.vector import Vector
from simworld.traffic.base.crosswalk import Crosswalk
from simworld.traffic.base.intersection import Intersection
from simworld.traffic.base.road import Road
from simworld.traffic.base.sidewalk import Sidewalk
from simworld.traffic.base.traffic_lane import TrafficLane
from simworld.traffic.base.traffic_signal import (
    TrafficSignal,
    TrafficSignalState,
)
from simworld.agent.pedestrian import Pedestrian
from simworld.agent.vehicle import Vehicle
from typing import List, Dict


class TaskRouteCrosswalk:
    """Authoritative task crosswalk absent from the road-owned zebra list."""

    def __init__(self, edge_index: int, start: Vector, end: Vector):
        self.id = f'task_edge_{edge_index}'
        self.start = start
        self.end = end
        self.road_id = None
        self.source = 'task_edge'


class RTTrafficController(TrafficController):
    def __init__(self, config: Config, map: str, seed: int = None,
                 dt: float = None, num_vehicles: int = 0,
                 num_pedestrians: int = 0):
        super().__init__(
            config,
            num_vehicles,
            num_pedestrians,
            map,
            seed,
            dt,
        )

        Vehicle.reset_id_counter()
        Pedestrian.reset_id_counter()
        TrafficSignal.reset_id_counter()
        Road.reset_id_counter()
        Intersection.reset_id_counter()
        TrafficLane.reset_id_counter()
        Sidewalk.reset_id_counter()
        Crosswalk.reset_id_counter()

    def update_states(self):
        """Refresh actor poses and make rendered RT heads authoritative."""
        super().update_states()
        for signal in self.traffic_signals:
            observed = self.communicator.get_traffic_signal_state(signal.id)
            if observed['vehicle_green']:
                state = (
                    TrafficSignalState.VEHICLE_GREEN,
                    TrafficSignalState.PEDESTRIAN_RED,
                )
            elif observed['pedestrian_walk']:
                state = (
                    TrafficSignalState.VEHICLE_RED,
                    TrafficSignalState.PEDESTRIAN_GREEN,
                )
            else:
                state = (
                    TrafficSignalState.VEHICLE_RED,
                    TrafficSignalState.PEDESTRIAN_RED,
                )
            signal.set_state(state)
            remaining = observed.get('remaining_time_s')
            if remaining is not None:
                signal.set_left_time(remaining)

    def get_intersections_from_task(self, task_intersections: List[List[float]], tolerance: float = 1.0):
        """
        Find Intersection objects from TrafficController that match the task's intersection coordinates.
        
        Args:
            task_intersections: List of [x, y] coordinates from task['route_info']['intersections']
            tolerance: Distance tolerance for matching intersections (default: 1.0)
            
        Returns:
            List of Intersection objects that match the task's intersections
            
        Example:
            >>> task = json.load(open('data/all_tasks.json'))['tasks'][0]
            >>> task_intersections = task['route_info']['intersections']
            >>> intersections = controller.get_intersections_from_task(task_intersections)
        """
        matched_intersections = []
        
        for task_coord in task_intersections:
            # Convert task coordinates (which are in map units) to controller units
            # Note: TrafficController multiplies by 100 when loading roads
            task_x = round(task_coord[0])
            task_y = round(task_coord[1])
            task_point = Vector(task_x, task_y)
            
            # Find matching intersection in controller's intersections
            for intersection in self.intersections:
                distance = (intersection.center - task_point).length()
                if distance <= tolerance:
                    matched_intersections.append(intersection)
                    break
        
        return matched_intersections
    
    def get_intersections_info(self, task_intersections: List[List[float]], tolerance: float = 1.0) -> List[Dict]:
        """
        Get detailed information about intersections matching the task's coordinates.
        
        Args:
            task_intersections: List of [x, y] coordinates from task['route_info']['intersections']
            tolerance: Distance tolerance for matching intersections (default: 1.0)
            
        Returns:
            List of dictionaries containing intersection information:
            - id: Intersection ID
            - center: [x, y] coordinates
            - num_connected_roads: Number of roads connected to this intersection
            - connected_road_ids: List of road IDs
            
        Example:
            >>> task = json.load(open('data/all_tasks.json'))['tasks'][0]
            >>> task_intersections = task['route_info']['intersections']
            >>> info = controller.get_intersections_info(task_intersections)
            >>> for i in info:
            ...     print(f"Intersection {i['id']} at {i['center']} has {i['num_connected_roads']} roads")
        """
        matched_intersections = self.get_intersections_from_task(task_intersections, tolerance)
        
        info_list = []
        for intersection in matched_intersections:
            info = {
                'id': intersection.id,
                'center': [intersection.center.x, intersection.center.y],
                'num_connected_roads': len(intersection.roads),
                'connected_road_ids': [road.id for road in intersection.roads],
                'has_traffic_lights': len(intersection.traffic_lights) > 0,
                'num_traffic_lights': len(intersection.traffic_lights),
                'num_pedestrian_lights': len(intersection.pedestrian_lights)
            }
            info_list.append(info)
        
        return info_list

    def get_route_crosswalks(
        self,
        path_points: List[List[float]],
        tolerance: float = 50.0,
        task_edges=None,
    ):
        """Resolve every authoritative crossing on an ordered task route.

        Task ``edges`` are not guaranteed to contain the crosswalk selected by
        the shortest path (RT12 Task5 is one example). The route itself is the
        reliable source: crossing segments have endpoints that match a generated
        map ``Crosswalk`` in either direction. Conversely, task generation also
        labels short connectors between adjacent road endpoints as crosswalks;
        those do not exist in the road-owned crosswalk list and must retain their
        task geometry so input, feedback, and evaluation score the same corridor.
        """
        if not path_points or len(path_points) < 2:
            return []

        path = [
            point if isinstance(point, Vector) else Vector(point[0], point[1])
            for point in path_points
        ]
        matched = []
        matched_ids = set()

        if task_edges is not None:
            candidates = [
                (
                    edge_index,
                    Vector(edge['node1'][0], edge['node1'][1]),
                    Vector(edge['node2'][0], edge['node2'][1]),
                )
                for edge_index, edge in enumerate(task_edges)
                if str(edge.get('type', '')).lower() == 'crosswalk'
            ]
        else:
            candidates = [
                (edge_index, route_start, route_end)
                for edge_index, (route_start, route_end) in enumerate(
                    zip(path, path[1:])
                )
            ]

        for edge_index, route_start, route_end in candidates:
            best_crosswalk = None
            best_error = float('inf')
            for crosswalk in self.crosswalks:
                direct_error = (
                    route_start.distance(crosswalk.start)
                    + route_end.distance(crosswalk.end)
                )
                reverse_error = (
                    route_start.distance(crosswalk.end)
                    + route_end.distance(crosswalk.start)
                )
                endpoint_error = min(direct_error, reverse_error)
                if endpoint_error < best_error:
                    best_error = endpoint_error
                    best_crosswalk = crosswalk

            if (
                best_crosswalk is not None
                and best_error <= tolerance * 2
                and best_crosswalk.id not in matched_ids
            ):
                matched.append(best_crosswalk)
                matched_ids.add(best_crosswalk.id)
            elif task_edges is not None:
                synthetic = TaskRouteCrosswalk(
                    edge_index,
                    route_start,
                    route_end,
                )
                matched.append(synthetic)
                matched_ids.add(synthetic.id)

        return matched

    def _get_route_crosswalk_intersection(self, crosswalk):
        """Resolve native and task-edge crossings to their rendered junction."""
        for intersection in self.intersections:
            if any(
                native.id == crosswalk.id
                for native in intersection.crosswalks
            ):
                return intersection
        if not self.intersections:
            return None
        midpoint = (crosswalk.start + crosswalk.end) * 0.5
        return min(
            self.intersections,
            key=lambda intersection: intersection.center.distance(midpoint),
        )

    def get_crosswalk_signal_groups(self, crosswalks) -> Dict[int, List[TrafficSignal]]:
        """Map each route crosswalk to all signals in its UE intersection.

        The RT intersection Blueprint switches its pedestrian phase as a group.
        Reading the group allows the agent to use the same rendered state even
        when the closest visible asset is a pedestrian-only signal without a
        Python ``crosswalk_id``.
        """
        groups = {}
        for crosswalk in crosswalks or []:
            intersection = RTTrafficController._get_route_crosswalk_intersection(
                self,
                crosswalk,
            )
            if intersection is not None:
                groups[crosswalk.id] = list(
                    intersection.traffic_lights
                ) + list(intersection.pedestrian_lights)
        return groups

    def get_crosswalk_intersection_names(self, crosswalks) -> Dict[int, str]:
        """Map route crosswalk IDs to their native UE controller actors."""
        names = {}
        for crosswalk in crosswalks or []:
            intersection = RTTrafficController._get_route_crosswalk_intersection(
                self,
                crosswalk,
            )
            if intersection is not None:
                names[crosswalk.id] = f"RT_Intersection_{intersection.id}"
        return names
