import random
from typing import List, Dict, Any, Set, Optional, Union
from simworld.map.map import Map, Edge, Node
from simworld.utils.vector import Vector


class PedestrianSampler:
    """Samples pedestrian routes on road networks."""
    
    def __init__(self, sidewalk_offset: int):
        """Initialize the pedestrian sampler.
        
        Args:
            sidewalk_offset: Distance offset for sidewalks from road center
        """
        self.sidewalk_offset = sidewalk_offset
    
    def sample_routes(self, 
                     roads: Set,
                     num_routes: int = 10,
                     no_crosswalk_routes: int = 0,
                     min_steps: int = 4,
                     max_steps: int = 10) -> List[Dict[str, Any]]:
        """Sample pedestrian routes from the roads.
        
        Args:
            roads: Set of roads to sample from
            num_routes: Number of routes to generate
            no_crosswalk_routes: Number of routes that cannot use crosswalks (0 to num_routes)
                                The remaining routes can use crosswalks.
            min_steps: Minimum number of steps for random walk
            max_steps: Maximum number of steps for random walk
            
        Returns:
            List of route dictionaries containing route nodes and loop info
        """
        if no_crosswalk_routes < 0 or no_crosswalk_routes > num_routes:
            raise ValueError(f"no_crosswalk_routes ({no_crosswalk_routes}) must be between 0 and num_routes ({num_routes})")
        
        # Build connectivity graph from sidewalks and crosswalks
        connectivity_graph = self._build_connectivity_graph(roads)
        
        # Create crosswalk settings: first (num_routes - no_crosswalk_routes) can use crosswalks
        # Last no_crosswalk_routes cannot use crosswalks
        crosswalk_allowed_count = num_routes - no_crosswalk_routes
        crosswalk_settings = [True] * crosswalk_allowed_count + [False] * no_crosswalk_routes
        
        # Shuffle to randomize which routes can/cannot use crosswalks
        random.shuffle(crosswalk_settings)
        
        routes = []
        for i in range(num_routes):
            start_node = random.choice(list(connectivity_graph.nodes))
            steps = random.randint(min_steps, max_steps)
            
            # Perform random walk with individual allow_crosswalk setting
            route = self._random_walk(
                connectivity_graph, 
                start_node, 
                steps, 
                crosswalk_settings[i]
            )
            routes.append(route)
        
        return routes
    
    def _build_connectivity_graph(self, roads: Set) -> Map:
        """Build connectivity graph from sidewalks and crosswalks using offset.
        
        Args:
            roads: Set of roads to process
            
        Returns:
            Map object representing the connectivity graph
        """
        connectivity_graph = Map(None)
        
        sidewalk_offset = self.sidewalk_offset
        
        # Process each road to create sidewalk and crosswalk nodes
        for road in roads:
            # Calculate sidewalk and crosswalk positions using offset
            normal = Vector(road.direction.y, -road.direction.x)
            offset = sidewalk_offset
            
            # Sidewalk points (parallel to road)
            p1 = road.start - normal * offset + road.direction * offset
            p2 = road.end - normal * offset - road.direction * offset
            p3 = road.end + normal * offset - road.direction * offset
            p4 = road.start + normal * offset + road.direction * offset
            
            # Create nodes
            sidewalk_node1 = Node(p1, road.direction, 'intersection')
            sidewalk_node2 = Node(p2, road.direction, 'intersection')
            sidewalk_node3 = Node(p3, road.direction, 'intersection')
            sidewalk_node4 = Node(p4, road.direction, 'intersection')
            
            # Add nodes to graph
            all_nodes = [sidewalk_node1, sidewalk_node2, sidewalk_node3, sidewalk_node4]
            for node in all_nodes:
                connectivity_graph.add_node(node)

            connectivity_graph.add_edge(Edge(all_nodes[0], all_nodes[1], 'sidewalk'))
            connectivity_graph.add_edge(Edge(all_nodes[2], all_nodes[3], 'sidewalk'))
            connectivity_graph.add_edge(Edge(all_nodes[0], all_nodes[3], 'crosswalk'))
            connectivity_graph.add_edge(Edge(all_nodes[1], all_nodes[2], 'crosswalk'))
        
        return connectivity_graph
    
    def _random_walk(self, 
                    connectivity_graph: Map, 
                    start_node: Node, 
                    max_steps: int, 
                    allow_crosswalk: bool = True) -> Dict[str, Any]:
        """Perform random walk on the connectivity graph.
        
        Args:
            connectivity_graph: The connectivity graph
            start_node: Starting node for the walk
            max_steps: Maximum number of steps
            allow_crosswalk: Whether to allow crossing crosswalks
            
        Returns:
            Dictionary with 'route' (list of nodes), 'is_loop' (bool), and 'has_crosswalk' (bool)
        """
        route = [start_node]
        current_node = start_node
        visited_edges = set()
        has_crosswalk = False  # Track if route uses any crosswalk
        
        for step in range(max_steps):
            # Get available neighbors
            available_neighbors = []
            for neighbor in connectivity_graph.adjacency_list[current_node]:
                # Check if we haven't visited this edge before
                edge = Edge(current_node, neighbor)  # type doesn't matter for equality
                if edge not in visited_edges:
                    # Find the actual edge in the graph to check its type
                    actual_edge = None
                    for e in connectivity_graph.edges:
                        if (e.node1 == current_node and e.node2 == neighbor) or \
                           (e.node2 == current_node and e.node1 == neighbor):
                            actual_edge = e
                            break
                    
                    # Filter out crosswalk edges if not allowed
                    if not allow_crosswalk and actual_edge and actual_edge.type == 'crosswalk':
                        continue
                    
                    available_neighbors.append(neighbor)
            
            # If no available neighbors, stop
            if not available_neighbors:
                break
            
            # Randomly select next node
            next_node = random.choice(available_neighbors)
            
            # Check if the edge to next_node is a crosswalk
            for e in connectivity_graph.edges:
                if (e.node1 == current_node and e.node2 == next_node) or \
                   (e.node2 == current_node and e.node1 == next_node):
                    if e.type == 'crosswalk':
                        has_crosswalk = True
                    break
            
            # Check if this creates a cycle (route already contains this node)
            if next_node in route:
                # Found a cycle, stop here
                route.append(next_node)
                break
            
            # Add to route
            route.append(next_node)
            
            # Mark edge as visited
            edge = Edge(current_node, next_node)  # type doesn't matter for equality
            visited_edges.add(edge)
            
            # Move to next node
            current_node = next_node
        
        # Check if route forms a loop
        is_loop = route[0] == route[-1]
        
        # If not a loop, try to add a sidewalk edge from the last node
        if not is_loop:
            last_node = route[-1]
            # Find a sidewalk edge connected to the last node
            for edge in connectivity_graph.edges:
                if edge.type == 'sidewalk':
                    if edge.node1 == last_node and edge.node2 not in route:
                        route.append(edge.node2)
                        break
                    elif edge.node2 == last_node and edge.node1 not in route:
                        route.append(edge.node1)
                        break
        else:
            route.pop()
        
        return {
            'route': route,
            'is_loop': is_loop,
            'has_crosswalk': has_crosswalk,
        }

