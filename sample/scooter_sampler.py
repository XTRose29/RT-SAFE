import random
from typing import List, Dict, Any, Set
from simworld.map.map import Map, Edge, Node
from simworld.utils.vector import Vector


class ScooterSampler:
    """Samples scooter routes on road networks."""
    
    def __init__(self, sidewalk_offset: int):
        """Initialize the scooter sampler.
        
        Args:
            sidewalk_offset: Distance offset for sidewalks from road center
        """
        self.sidewalk_offset = sidewalk_offset
    
    def sample_routes(self, roads: Set) -> List[Dict[str, Any]]:
        """Sample scooter routes from the roads.
        
        For each road, create a loop route using its 4 nodes (2 sidewalks + 2 crosswalks).
        Each road naturally forms a loop: p1 -> p2 -> p3 -> p4 -> p1
        
        Args:
            roads: Set of roads to sample from
            
        Returns:
            List of route dictionaries, one loop per road
        """
        routes = []
        
        for road in roads:
            # Calculate sidewalk and crosswalk positions using offset (same as PedestrianSampler)
            normal = Vector(road.direction.y, -road.direction.x)
            offset = self.sidewalk_offset + 150
            
            # Sidewalk points (parallel to road)
            p1 = road.start - normal * offset + road.direction * offset
            p2 = road.end - normal * offset - road.direction * offset
            p3 = road.end + normal * offset - road.direction * offset
            p4 = road.start + normal * offset + road.direction * offset
            
            # Create nodes
            node1 = Node(p1, road.direction, 'intersection')
            node2 = Node(p2, road.direction, 'intersection')
            node3 = Node(p3, road.direction, 'intersection')
            node4 = Node(p4, road.direction, 'intersection')
            
            # Create a loop route: p1 -> p2 -> p3 -> p4 -> back to p1
            # Randomly choose starting point and direction (clockwise or counter-clockwise)
            nodes = [node1, node2, node3, node4]
            start_idx = random.randint(0, 3)
            clockwise = random.choice([True, False])
            
            if clockwise:
                # Rotate to start from start_idx, going clockwise
                route = [nodes[(start_idx + i) % 4] for i in range(4)]
            else:
                # Rotate to start from start_idx, going counter-clockwise
                route = [nodes[(start_idx - i) % 4] for i in range(4)]
            
            routes.append({
                'route': route,
                'is_loop': True,
                'has_crosswalk': True,  # Always has crosswalk since it's a complete loop
                'road': road  # Keep reference to original road
            })
        
        return routes
