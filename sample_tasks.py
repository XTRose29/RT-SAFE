#!/usr/bin/env python3
"""
Task Sampling Script

Samples 4 unique tasks per map with constraints:
- No duplicate trajectories (start/end edges must be different)
- Exactly 1 task without crosswalk, 3 tasks must pass through crosswalk
- Fixed hop counts: 2, 3, 4, 4

Output task structure (tasks.json / all_tasks.json):
{
    "task_id": int,                    # Global unique task ID
    "start_point": [x, y],            # Agent start position
    "end_point": [x, y],              # Agent destination position
    "start_edge": {                    # Edge on which the start_point lies
        "node1": [x, y],
        "node2": [x, y],
        "type": "sidewalk" | "crosswalk"
    },
    "end_edge": {                      # Edge on which the end_point lies
        "node1": [x, y],
        "node2": [x, y],
        "type": "sidewalk" | "crosswalk"
    },
    "edges": [                         # Ordered list of edges forming the route
        {
            "node1": [x, y],
            "node2": [x, y],
            "type": "sidewalk" | "crosswalk"
        },
        ...
    ],
    "total_hops": int,                 # Total number of edges in the route
    "crosswalk_hops": int,             # Number of crosswalk edges in the route
    "map_path": str,                   # Path to the map roads.json file
    "route_info": {
        "shortest_path": [[x, y], ...],   # Shortest path nodes from start to end
        "intersections": [[x, y], ...]     # Intersection points along the route
    },
    "pedestrian_routes": [             # Pedestrian routes on roads along the route
        {
            "route_id": int,
            "route": [[x, y], ...],    # Ordered list of waypoint positions
            "is_loop": bool,           # Whether the route forms a loop
            "has_crosswalk": bool      # Whether the route includes a crosswalk
        },
        ...
    ],
    "pedestrian_routes_num": int,      # Number of pedestrian routes
    "scooter_routes": [                # Scooter loop routes on roads along the route
        {
            "route_id": int,
            "route": [[x, y], ...],    # Ordered list of waypoint positions
            "is_loop": bool,           # Whether the route forms a loop
            "has_crosswalk": bool      # Whether the route includes a crosswalk
        },
        ...
    ],
    "scooter_routes_num": int,         # Number of scooter routes
    "irregular_routes": [              # Irregular routes sampled from walker waypoints
        {
            "route_id": int,
            "sidewalk_id": int,        # ID of the sidewalk this route belongs to
            "route": [[x, y], ...]     # Ordered waypoints alternating far/near road positions
        },
        ...
    ],
    "irregular_routes_num": int,       # Number of irregular routes
    "falling_objects": [               # Falling object obstacle IDs (pre-generated in obstacles.json)
        "GEN_RT_FO_RT_Falling_xxx_N",  # Each element is an obstacle ID string
        ...
    ],
    "falling_objects_num": int         # Number of falling objects
}
"""

import json
import random
import copy
import os
from datetime import datetime
from simworld.config import Config
from simworld.map.map import Map, Edge, Node
from simworld.utils.vector import Vector
from sample.task_sampler import TaskSampler
from sample.pedestrian_sampler import PedestrianSampler
from sample.scooter_sampler import ScooterSampler


# Map configurations: (map_path, output_path)
MAPS = [
    ('data/map1_10roads/roads.json', 'data/map1_10roads/task_routes.json'),
    ('data/map2_12roads/roads.json', 'data/map2_12roads/task_routes.json'),
    ('data/map3_15roads/roads.json', 'data/map3_15roads/task_routes.json'),
    ('data/map4_18roads/roads.json', 'data/map4_18roads/task_routes.json'),
    ('data/map5_20roads/roads.json', 'data/map5_20roads/task_routes.json'),
]

CONFIG_PATH = 'config.yaml'
TASKS_PER_MAP = 8
# Fixed hop requirements per map: [2, 3, 4, 4]
HOP_REQUIREMENTS = [2, 2, 2, 3, 3, 3, 4, 4]
# Exactly 1 task without crosswalk, 3 tasks must pass through crosswalk
MAX_NO_CROSSWALK_TASKS = 4
SEED = 42
PEDESTRIAN_ROUTES_MULTIPLIER = 3  # Number of pedestrian routes = total_hops * this multiplier
IRREGULAR_ROUTES_PER_SIDEWALK = 10  # Number of irregular routes to sample per sidewalk
FINAL_OUTPUT_PATH = 'data/all_tasks.json'  # Combined output file
USE_EXISTING_TASKS = False  # If True, load tasks from existing task_routes.json files; if False, generate new tasks


def get_intersections_from_roads(roads):
    """
    Extract intersection points from a set of roads.
    
    Args:
        roads: Set of road objects
        
    Returns:
        List of intersection points (as Vector objects with x, y coordinates)
    """
    from collections import Counter
    
    # Collect all road endpoints
    endpoints = []
    for road in roads:
        # Round to 2 decimal places to handle floating point precision
        endpoints.append((round(road.start.x, 2), round(road.start.y, 2)))
        endpoints.append((round(road.end.x, 2), round(road.end.y, 2)))
    
    # Find points that appear more than once (true intersections)
    endpoint_counts = Counter(endpoints)
    intersections = [
        Vector(point[0], point[1]) 
        for point, count in endpoint_counts.items() 
        if count > 1
    ]
    
    return intersections


def get_route_info(map: Map, task):
    """
    Get the route info for the task
    
    Args:
        map: Map instance
        task: a dictionary containing the start_point and end_point
        
    Returns:
        a dictionary containing the shortest path, roads, the start and end points, and the route from edges
    """
    start_point = task['start_point']
    end_point = task['end_point']
    start_edge = task['start_edge']
    end_edge = task['end_edge']
    
    # create a new graph
    new_map = copy.deepcopy(map)
    
    # create the start and end nodes
    start_node = Node(start_point, Vector(0, 0), 'sidewalk')
    end_node = Node(end_point, Vector(0, 0), 'sidewalk')
    
    new_map.add_node(start_node)
    new_map.add_node(end_node)
    new_map.add_edge(Edge(start_node, start_edge.node1, 'sidewalk'))
    new_map.add_edge(Edge(start_node, start_edge.node2, 'sidewalk'))
    new_map.add_edge(Edge(end_node, end_edge.node1, 'sidewalk'))
    new_map.add_edge(Edge(end_node, end_edge.node2, 'sidewalk'))

    new_map.edges.remove(start_edge)
    new_map.edges.remove(end_edge)
    
    shortest_path = new_map.get_shortest_path(start_node, end_node)
    
    # Get roads from shortest path
    roads = set()
    for node in shortest_path:
        road = map.get_road_at_position(node.position)
        roads.add(road)
    
    # Get intersections from roads
    intersections = get_intersections_from_roads(roads)
    
    # Build route from task edges
    route = None
    if 'edges' in task and len(task['edges']) > 0:
        edges = task['edges']
        if len(edges) == 1:
            route = [edges[0].node1, edges[0].node2]
        else:
            route = []
            # Start with the first edge
            if edges[0].node1 == edges[1].node1 or edges[0].node1 == edges[1].node2:
                route.append(edges[0].node2)
                route.append(edges[0].node1)
            else:
                route.append(edges[0].node1)
                route.append(edges[0].node2)
            
            # Process remaining edges
            for i in range(1, len(edges)):
                current_edge = edges[i]
                prev_end_node = route[-1]
                
                # Check which node of current edge connects to the previous end
                if current_edge.node1 == prev_end_node:
                    route.append(current_edge.node2)
                elif current_edge.node2 == prev_end_node:
                    route.append(current_edge.node1)
    
    result = {
        "start_point": start_point,
        "end_point": end_point,
        "shortest_path": shortest_path,
        "roads": roads,
        "intersections": intersections,
    }
    
    if route:
        result['default_route'] = route
    
    return result


def get_edge_key(edge):
    """Create a unique key for an edge based on its endpoints."""
    p1 = (round(edge.node1.position.x, 2), round(edge.node1.position.y, 2))
    p2 = (round(edge.node2.position.x, 2), round(edge.node2.position.y, 2))
    return tuple(sorted([p1, p2]))


def load_tasks_from_json(json_path):
    """Load tasks from existing task_routes.json file and reconstruct objects.
    
    Args:
        json_path: Path to the task_routes.json file
        
    Returns:
        List of task dictionaries with reconstructed Edge and Node objects
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tasks = []
    for task_data in data['tasks']:
        # Reconstruct start_point and end_point as Vector objects
        start_point = Vector(task_data['start_point'][0], task_data['start_point'][1])
        end_point = Vector(task_data['end_point'][0], task_data['end_point'][1])
        
        # Reconstruct start_edge and end_edge
        start_edge_data = task_data['start_edge']
        start_edge = Edge(
            Node(Vector(start_edge_data['node1'][0], start_edge_data['node1'][1]), Vector(0, 0)),
            Node(Vector(start_edge_data['node2'][0], start_edge_data['node2'][1]), Vector(0, 0)),
            start_edge_data['type']
        )
        
        end_edge_data = task_data['end_edge']
        end_edge = Edge(
            Node(Vector(end_edge_data['node1'][0], end_edge_data['node1'][1]), Vector(0, 0)),
            Node(Vector(end_edge_data['node2'][0], end_edge_data['node2'][1]), Vector(0, 0)),
            end_edge_data['type']
        )
        
        # Reconstruct edges list
        edges = []
        for edge_data in task_data['edges']:
            edge = Edge(
                Node(Vector(edge_data['node1'][0], edge_data['node1'][1]), Vector(0, 0)),
                Node(Vector(edge_data['node2'][0], edge_data['node2'][1]), Vector(0, 0)),
                edge_data['type']
            )
            edges.append(edge)
        
        task = {
            'start_point': start_point,
            'end_point': end_point,
            'start_edge': start_edge,
            'end_edge': end_edge,
            'edges': edges,
            'total_hops': task_data['total_hops'],
            'crosswalk_hops': task_data['crosswalk_hops']
        }
        tasks.append(task)
    
    print(f"  Loaded {len(tasks)} tasks from {json_path}")
    return tasks


def load_walker_waypoints(map_dir):
    """Load walker waypoints from the map directory.
    
    Args:
        map_dir: Path to the map directory (e.g., 'data/map1_10roads')
        
    Returns:
        List of sidewalk waypoint dicts, or None if file not found.
        Each sidewalk has: sidewalk_id, start, end, far_road, near_road (same-length lists).
    """
    waypoints_path = os.path.join(map_dir, 'walker_waypoints.json')
    if not os.path.exists(waypoints_path):
        print(f"  Warning: {waypoints_path} not found, skipping irregular routes")
        return None
    
    with open(waypoints_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['sidewalks']


def match_sidewalks_to_task(task, sidewalks):
    """Match task's sidewalk edges to sidewalk waypoints by comparing endpoints.
    
    Args:
        task: Task dict with 'edges' list (Edge objects)
        sidewalks: List of sidewalk waypoint dicts from walker_waypoints.json
        
    Returns:
        List of matched sidewalk waypoint dicts (one per sidewalk edge in the task)
    """
    matched = []
    
    for edge in task['edges']:
        if edge.type != 'sidewalk':
            continue
        
        edge_n1 = (round(edge.node1.position.x, 2), round(edge.node1.position.y, 2))
        edge_n2 = (round(edge.node2.position.x, 2), round(edge.node2.position.y, 2))
        
        for sw in sidewalks:
            sw_start = (round(sw['start']['x'], 2), round(sw['start']['y'], 2))
            sw_end = (round(sw['end']['x'], 2), round(sw['end']['y'], 2))
            
            if (edge_n1 == sw_start and edge_n2 == sw_end) or \
               (edge_n1 == sw_end and edge_n2 == sw_start):
                matched.append(sw)
                break
    
    return matched


def sample_irregular_routes(sidewalk, num_routes, min_route_len=2, max_route_len=6):
    """Sample irregular routes on a sidewalk by alternating far/near waypoints.
    
    Each route alternates between far_road and near_road points.
    Adjacency rule: if current point is far[i], the next point can be
    near[i-1], near[i], or near[i+1] (and vice versa).
    The first point is chosen randomly; no duplicate points within a route,
    no duplicate routes returned.
    
    Args:
        sidewalk: Sidewalk waypoint dict with far_road and near_road lists
        num_routes: Number of routes to sample
        min_route_len: Minimum number of waypoints per route (default 2)
        max_route_len: Maximum number of waypoints per route (default 6)
        
    Returns:
        List of route dicts, each with:
            - 'sidewalk_id': int
            - 'route': list of [x, y] coordinate pairs
    """
    far = sidewalk['far_road']
    near = sidewalk['near_road']
    n = len(far)  # far and near have the same length
    
    if n == 0:
        return []
    
    routes = []
    seen_routes = set()
    max_attempts = num_routes * 30
    attempts = 0
    
    while len(routes) < num_routes and attempts < max_attempts:
        attempts += 1
        route_len = random.randint(min_route_len, max_route_len)
        
        # Pick a random starting point (random side, random index)
        is_far = random.choice([True, False])
        idx = random.randint(0, n - 1)
        
        route = []
        used = set()  # track (side, idx) to avoid revisiting
        
        if is_far:
            route.append([far[idx]['x'], far[idx]['y']])
            used.add(('far', idx))
            current_side = 'far'
        else:
            route.append([near[idx]['x'], near[idx]['y']])
            used.add(('near', idx))
            current_side = 'near'
        current_idx = idx
        
        # Extend the route by alternating sides
        for _ in range(route_len - 1):
            next_side = 'near' if current_side == 'far' else 'far'
            next_points = near if next_side == 'near' else far
            
            # Candidate indices: current_idx - 1, current_idx, current_idx + 1
            candidates = []
            for di in [-1, 0, 1]:
                ni = current_idx + di
                if 0 <= ni < n and (next_side, ni) not in used:
                    candidates.append(ni)
            
            if not candidates:
                break
            
            next_idx = random.choice(candidates)
            route.append([next_points[next_idx]['x'], next_points[next_idx]['y']])
            used.add((next_side, next_idx))
            current_side = next_side
            current_idx = next_idx
        
        # Only accept routes with at least min_route_len points
        if len(route) < min_route_len:
            continue
        
        # Deduplicate by route content
        route_key = tuple(tuple(p) for p in route)
        if route_key in seen_routes:
            continue
        
        seen_routes.add(route_key)
        routes.append({
            'sidewalk_id': sidewalk['sidewalk_id'],
            'route': route
        })
    
    return routes


def task_to_dict_simple(task, task_id):
    """Convert task object to serializable dict (without pedestrian routes)."""
    return {
        'task_id': task_id,
        'start_point': [task['start_point'].x, task['start_point'].y],
        'end_point': [task['end_point'].x, task['end_point'].y],
        'start_edge': {
            'node1': [task['start_edge'].node1.position.x, task['start_edge'].node1.position.y],
            'node2': [task['start_edge'].node2.position.x, task['start_edge'].node2.position.y],
            'type': task['start_edge'].type
        },
        'end_edge': {
            'node1': [task['end_edge'].node1.position.x, task['end_edge'].node1.position.y],
            'node2': [task['end_edge'].node2.position.x, task['end_edge'].node2.position.y],
            'type': task['end_edge'].type
        },
        'edges': [
            {
                'node1': [edge.node1.position.x, edge.node1.position.y],
                'node2': [edge.node2.position.x, edge.node2.position.y],
                'type': edge.type
            } for edge in task['edges']
        ],
        'total_hops': task['total_hops'],
        'crosswalk_hops': task['crosswalk_hops']
    }


def task_to_dict_full(task, task_id, map_path):
    """Convert task object to serializable dict (with pedestrian routes and map info)."""
    result = task_to_dict_simple(task, task_id)
    result['map_path'] = map_path
    
    # Add route info
    if 'route_info' in task:
        result['route_info'] = {
            'shortest_path': [
                [node.position.x, node.position.y] for node in task['route_info']['shortest_path']
            ]
        }
        
        # Add intersections
        if 'intersections' in task['route_info']:
            result['route_info']['intersections'] = [
                [intersection.x, intersection.y] for intersection in task['route_info']['intersections']
            ]
    
    # Add pedestrian routes
    if 'pedestrian_routes' in task:
        result['pedestrian_routes'] = [
            {
                'route_id': i,
                'route': [[node.position.x, node.position.y] for node in route['route']],
                'is_loop': route['is_loop'],
                'has_crosswalk': route['has_crosswalk'],
            } for i, route in enumerate(task['pedestrian_routes'])
        ]
        result['pedestrian_routes_num'] = len(task['pedestrian_routes'])
    
    # Add scooter routes
    if 'scooter_routes' in task:
        result['scooter_routes'] = [
            {
                'route_id': i,
                'route': [[node.position.x, node.position.y] for node in route['route']],
                'is_loop': route['is_loop'],
                'has_crosswalk': route['has_crosswalk'],
            } for i, route in enumerate(task['scooter_routes'])
        ]
        result['scooter_routes_num'] = len(task['scooter_routes'])
    
    # Add irregular routes (already serialized as [x, y] lists)
    if 'irregular_routes' in task:
        result['irregular_routes'] = [
            {
                'route_id': i,
                'sidewalk_id': route['sidewalk_id'],
                'route': route['route'],
            } for i, route in enumerate(task['irregular_routes'])
        ]
        result['irregular_routes_num'] = len(task['irregular_routes'])
    
    # Add falling object IDs (pre-generated in obstacles.json)
    if 'falling_objects' in task:
        result['falling_objects'] = task['falling_objects']  # list of obstacle IDs
        result['falling_objects_num'] = len(task['falling_objects'])
    
    return result


def main():
    random.seed(SEED)
    config = Config(CONFIG_PATH)
    
    all_tasks_full = []  # For combined output with pedestrian routes
    global_task_id = 0
    
    for map_path, output_path in MAPS:
        print(f"\nProcessing {map_path}...")
        
        try:
            # Initialize map, pedestrian sampler, and scooter sampler (needed for both modes)
            city_map = Map(config, None)
            city_map.initialize_map_from_file(map_path, config['traffic.sidewalk_offset'], False)
            pedestrian_sampler = PedestrianSampler(config['traffic.sidewalk_offset'])
            scooter_sampler = ScooterSampler(config['traffic.sidewalk_offset'])
            
            # Load existing tasks or generate new ones
            if USE_EXISTING_TASKS:
                if os.path.exists(output_path):
                    print(f"  Loading existing tasks from {output_path}...")
                    tasks = load_tasks_from_json(output_path)
                else:
                    print(f"  File {output_path} not found, generating new tasks...")
                    tasks = sample_tasks_for_map(config, map_path, TASKS_PER_MAP, ped_multiplier=0)  # Don't generate ped routes yet
                    
                    # Save simple tasks (without pedestrian routes) per map
                    simple_output_data = {
                        'generated_at': datetime.now().isoformat(),
                        'map_path': map_path,
                        'total_tasks': len(tasks),
                        'tasks': [task_to_dict_simple(t, i) for i, t in enumerate(tasks)]
                    }
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(simple_output_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Saved {len(tasks)} simple tasks to {output_path}")
                    
                    # Clear pedestrian routes since we set multiplier to 0
                    for task in tasks:
                        if 'pedestrian_routes' in task:
                            del task['pedestrian_routes']
                        if 'route_info' in task:
                            del task['route_info']
            else:
                # Generate new tasks from scratch (without pedestrian routes yet)
                print("  Generating new tasks...")
                task_sampler = TaskSampler(city_map)
                
                tasks = []
                used_edge_pairs = set()
                
                def try_add_task(task):
                    if task is None:
                        return False
                    start_edge_key = get_edge_key(task['start_edge'])
                    end_edge_key = get_edge_key(task['end_edge'])
                    edge_pair = tuple(sorted([start_edge_key, end_edge_key]))
                    if edge_pair in used_edge_pairs:
                        return False
                    used_edge_pairs.add(edge_pair)
                    tasks.append(task)
                    print(f"  Task {len(tasks)}: {task['total_hops']} hops, {task['crosswalk_hops']} crosswalks")
                    return True
                
                # Sample 4 tasks: hops [2, 3, 4, 4], exactly 1 without crosswalk
                # Step 1: Try to sample 1 task with 0 crosswalks for one of the hop values
                no_crosswalk_hop = None
                for try_hop in [2, 3, 4]:
                    print(f"  Sampling task with {try_hop} hops, 0 crosswalks...")
                    for _ in range(200):
                        task = task_sampler.sample_task(try_hop, try_hop, crosswalk_hops=0, max_trials=200)
                        if try_add_task(task):
                            no_crosswalk_hop = try_hop
                            break
                    if no_crosswalk_hop is not None:
                        break
                
                if no_crosswalk_hop is None:
                    raise RuntimeError(f"Could not find a no-crosswalk task for any hop in [2,3,4] on {map_path}")
                
                # Step 2: Sample remaining tasks.
                # Hard constraint: routes with hop <= 2 must NOT pass through crosswalk.
                remaining_hops = list(HOP_REQUIREMENTS)
                remaining_hops.remove(no_crosswalk_hop)  # remove first occurrence only
                for hop in remaining_hops:
                    if hop <= 2:
                        print(f"  Sampling task with {hop} hops, no crosswalk...")
                        for _ in range(200):
                            task = task_sampler.sample_task(hop, hop, crosswalk_hops=0, max_trials=200)
                            if task is not None and task['crosswalk_hops'] == 0 and try_add_task(task):
                                break
                        else:
                            raise RuntimeError(f"Could not find a no-crosswalk task with {hop} hops on {map_path}")
                    else:
                        print(f"  Sampling task with {hop} hops, with crosswalk...")
                        for _ in range(200):
                            task = task_sampler.sample_task(hop, hop, crosswalk_hops=-1, max_trials=200)
                            if task is not None and task['crosswalk_hops'] >= 1 and try_add_task(task):
                                break
                        else:
                            raise RuntimeError(f"Could not find a crosswalk task with {hop} hops on {map_path}")
                
                # Save simple tasks (without pedestrian routes) per map
                simple_output_data = {
                    'generated_at': datetime.now().isoformat(),
                    'map_path': map_path,
                    'total_tasks': len(tasks),
                    'tasks': [task_to_dict_simple(t, i) for i, t in enumerate(tasks)]
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_output_data, f, indent=2, ensure_ascii=False)
                
                print(f"  Saved {len(tasks)} simple tasks to {output_path}")
            
            # Generate pedestrian routes and scooter routes for all tasks (whether loaded or newly generated)
            print(f"  Generating pedestrian and scooter routes...")
            for task in tasks:
                route_info = get_route_info(city_map, task)
                num_ped_routes = task['total_hops'] * PEDESTRIAN_ROUTES_MULTIPLIER
                # All pedestrian routes can use crosswalks
                pedestrian_routes = pedestrian_sampler.sample_routes(
                    roads=route_info['roads'], 
                    num_routes=num_ped_routes,
                    no_crosswalk_routes=0
                )
                
                # Add non-crosswalk edges as individual pedestrian routes
                for edge in task['edges']:
                    if edge.type != 'crosswalk':
                        pedestrian_routes.append({
                            'route': [edge.node1, edge.node2],
                            'is_loop': False,
                            'has_crosswalk': False
                        })
                
                # Deduplicate pedestrian routes based on route nodes
                def route_to_key(route_dict):
                    """Create a unique key for a route based on its nodes."""
                    route = route_dict['route']
                    # Convert route to tuple of positions for hashing
                    positions = tuple(
                        (round(node.position.x, 2), round(node.position.y, 2)) 
                        for node in route
                    )
                    # Keep forward and reverse directions as different routes
                    return positions
                
                seen_routes = set()
                unique_routes = []
                for route_dict in pedestrian_routes:
                    route_key = route_to_key(route_dict)
                    if route_key not in seen_routes:
                        seen_routes.add(route_key)
                        unique_routes.append(route_dict)

                # Add agent's default route if available
                if 'default_route' in route_info:
                    unique_routes.append({
                        'route': route_info['default_route'], 
                        'is_loop': False,
                        'has_crosswalk': task['crosswalk_hops'] > 0
                    })
                
                task['pedestrian_routes'] = unique_routes
                
                # Generate scooter routes (one loop per road)
                scooter_routes = scooter_sampler.sample_routes(roads=route_info['roads'])
                task['scooter_routes'] = scooter_routes
                
                task['route_info'] = route_info
            
            # Generate irregular routes from walker waypoints
            map_dir = os.path.dirname(map_path)
            sidewalks = load_walker_waypoints(map_dir)
            if sidewalks is not None:
                print(f"  Generating irregular routes and collecting falling objects...")
                for task in tasks:
                    matched_sidewalks = match_sidewalks_to_task(task, sidewalks)
                    all_irregular = []
                    task_route_keys = set()  # deduplicate across sidewalks within a task
                    all_falling_objects = []  # list of obstacle IDs
                    for sw in matched_sidewalks:
                        sw_routes = sample_irregular_routes(
                            sw, IRREGULAR_ROUTES_PER_SIDEWALK,
                            min_route_len=2, max_route_len=6
                        )
                        for r in sw_routes:
                            route_key = tuple(tuple(p) for p in r['route'])
                            if route_key not in task_route_keys:
                                task_route_keys.add(route_key)
                                all_irregular.append(r)
                        
                        # Collect pre-generated falling object IDs from this sidewalk
                        fo_ids = sw.get('falling_objects', [])
                        all_falling_objects.extend(fo_ids)
                    
                    task['irregular_routes'] = all_irregular
                    task['falling_objects'] = all_falling_objects
                    print(f"    Task: {len(matched_sidewalks)} sidewalks matched, "
                          f"{len(all_irregular)} irregular routes, "
                          f"{len(all_falling_objects)} falling objects")
            
            # Collect full tasks for combined output
            map_tasks_full = []
            for task in tasks:
                task_dict = task_to_dict_full(task, global_task_id, map_path)
                all_tasks_full.append(task_dict)
                map_tasks_full.append(task_dict)
                global_task_id += 1
            
            # Save full tasks to the map's own directory
            map_dir = os.path.dirname(map_path)
            per_map_output_path = os.path.join(map_dir, 'tasks.json')
            per_map_output_data = {
                'generated_at': datetime.now().isoformat(),
                'map_path': map_path,
                'total_tasks': len(map_tasks_full),
                'tasks': map_tasks_full
            }
            with open(per_map_output_path, 'w', encoding='utf-8') as f:
                json.dump(per_map_output_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(map_tasks_full)} full tasks to {per_map_output_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined output with all tasks and pedestrian routes
    combined_output_data = {
        'generated_at': datetime.now().isoformat(),
        'total_tasks': len(all_tasks_full),
        'tasks': all_tasks_full
    }
    
    with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(combined_output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(all_tasks_full)} full tasks to {FINAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
