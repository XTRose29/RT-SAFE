import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from simworld.map.map import Node
from rt_map import RTMap
from simworld.utils.vector import Vector
from simworld.config import Config

# 占比参数：far=树, middle=RT, near=其余
TREE_PROPORTION = 0.4
RT_PROPORTION = 0.4
OTHER_PROPORTION = 0.2

class ObstacleSampler:
    """Samples obstacles on map nodes based on configuration and element data."""
    
    def __init__(self, config: Config):
        """Initialize the obstacle sampler.
        
        Args:
            config: Configuration object containing sampling parameters
        """
        self.config = config
        self.obstacles = []
        
    def load_elements_bbox(self, json_path: str) -> Dict[str, Any]:
        """Load element bounding box data from JSON file.
        
        Args:
            json_path: Path to the bbox.json file
            
        Returns:
            Dictionary containing element data with bounding boxes
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['elements']  # Only return the elements section
    
    def sample_obstacles(self, map: RTMap, bbox_path: str, element_number_per_road: int = 2) -> List[Dict[str, Any]]:
        """Sample obstacles on the map.
        
        Args:
            map: Map instance
            bbox_path: Path to bbox.json file
            element_number_per_road: Number of elements per road
            
        Returns:
            List of obstacle placement dictionaries
        """
        # 1. Load elements bbox data
        elements_bbox = self.load_elements_bbox(bbox_path)
        element_names = list(elements_bbox.keys())
        
        # 2. Get different types of nodes (far / middle / near)
        sidewalk_far_nodes = map.get_sidewalk_far_road_nodes()
        sidewalk_middle_nodes = map.get_sidewalk_middle_nodes()
        sidewalk_near_nodes = map.get_sidewalk_near_road_nodes()

        # 3. Categorize element names: far=树, middle=RT开头, near=剩下
        tree_names = [name for name in element_names if name.startswith('BP_Tree')]
        rt_names = [name for name in element_names if name.startswith('RT_')]
        other_names = [name for name in element_names if not name.startswith('BP_Tree') and not name.startswith('RT_')]

        # 4. Decouple middle from element_number_per_road:
        #    - middle: use all nodes
        #    - far/near: split element budget as evenly as possible
        num_elements = element_number_per_road * len(map.roads)
        num_trees = num_elements // 2
        num_others = num_elements - num_trees

        # 5. Sample nodes per zone:
        #    - far/near 根据预算随机采样（均分预算）
        #    - middle 使用所有可用点位
        sampled_tree_nodes = random.sample(
            sidewalk_far_nodes, min(len(sidewalk_far_nodes), num_trees)
        ) if sidewalk_far_nodes else []
        sampled_rt_nodes = list(sidewalk_middle_nodes) if sidewalk_middle_nodes else []
        sampled_other_nodes = random.sample(
            sidewalk_near_nodes, min(len(sidewalk_near_nodes), num_others)
        ) if sidewalk_near_nodes else []

        # 6. Mark nodes as obstacles
        all_sampled_nodes = sampled_tree_nodes + sampled_rt_nodes + sampled_other_nodes
        for node in all_sampled_nodes:
            node.obstacle = True

        # 7. Randomly select element names per zone
        sampled_tree_names = random.choices(tree_names, k=len(sampled_tree_nodes)) if tree_names else []
        sampled_rt_names = random.choices(rt_names, k=len(sampled_rt_nodes)) if rt_names else []
        sampled_other_names = random.choices(other_names, k=len(sampled_other_nodes)) if other_names else []

        # 8. Build placements: far=tree, middle=rt, near=other
        placements = []
        for name, node in zip(sampled_tree_names, sampled_tree_nodes):
            placements.append({
                'element': name,
                'node': node,
                'bbox': elements_bbox[name]['bbox'],
                'type': 'tree'
            })
        for name, node in zip(sampled_rt_names, sampled_rt_nodes):
            placements.append({
                'element': name,
                'node': node,
                'bbox': elements_bbox[name]['bbox'],
                'type': 'rt'
            })
        for name, node in zip(sampled_other_names, sampled_other_nodes):
            placements.append({
                'element': name,
                'node': node,
                'bbox': elements_bbox[name]['bbox'],
                'type': 'other'
            })
        
        self.obstacles = placements
        return placements
    
    def _point_to_segment_distance(self, p: Vector, a: Vector, b: Vector) -> float:
        """Calculate distance from point to line segment.
        
        Args:
            p: Point
            a: Start of line segment
            b: End of line segment
            
        Returns:
            Distance from point to line segment
        """
        ab = b - a
        ap = p - a
        ab_len2 = ab.x ** 2 + ab.y ** 2
        if ab_len2 == 0:
            return ap.length()
        t = max(0, min(1, (ap.x * ab.x + ap.y * ab.y) / ab_len2))
        proj = a + ab * t
        return (p - proj).length()

    def generate_progen_world_json(self, elements, output_world_path):
        """Generate a complete progen_world.json file directly.
        
        Args:
            elements: List of obstacle elements to include
            output_world_path: Path where to save the generated JSON file
        """

        base_map_config = {
            "name": "map_1",
            "env_bin": "gym_citynav\\Binaries\\Win64\\gym_citynav.exe",
            "width": 1000,
            "height": 1000
        }
        
        # Generate nodes from elements
        nodes = []
        for i, element in enumerate(elements):
            nodes.append(self.element_to_node_dict(element, i))
        
        # Create the complete world data structure
        world_data = {
            "base_map": base_map_config,
            "nodes": nodes
        }
        
        # Write to file
        with open(output_world_path, 'w') as f:
            json.dump(world_data, f, indent=2)
    
    def element_to_node_dict(self, element, idx):
        node = element['node']
        name = element['element']
        return {
            "id": f"GEN_RT_{'_'.join(name.split('_')[:-1])}_{idx}",
            "instance_name": name,
            "properties": {
                "location": {
                    "x": node.position.x + random.uniform(-5, 5),
                    "y": node.position.y + random.uniform(-5, 5),
                    "z": 2
                },
                "orientation": {
                    "pitch": 0,
                    "yaw": random.uniform(0, 2 * math.pi),
                    "roll": 0
                },
                "scale": {
                    "x": 1.0,
                    "y": 1.0,
                    "z": 1.0
                }
            }
        }


# Example usage
if __name__ == "__main__":
    random.seed(42)
    config = Config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
    
    # Create obstacle sampler
    sampler = ObstacleSampler(config)
    
    # Sample obstacles for all 5 maps
    maps = [
        "map1_10roads",
        "map2_12roads",
        "map3_15roads",
        "map4_18roads",
        "map5_20roads"
    ]
    
    for map_name in maps:
        print(f"Processing {map_name}...")
        
        # Create new sampler for each map
        sampler = ObstacleSampler(config)
        
        # Initialize map
        map = RTMap(config)
        roads_path = rf"E:\Projects\SimWorld-RBench\data\{map_name}\roads.json"
        map.initialize_map_from_file(roads_path, config['traffic.sidewalk_offset'], True, waypoints_distance=600, waypoints_normal_distance=120)
        
        # Sample obstacles
        obstacles = sampler.sample_obstacles(map, "data/bbox.json", element_number_per_road=30)
        
        # Generate obstacles.json file
        output_path = rf"data\{map_name}\obstacles.json"
        sampler.generate_progen_world_json(obstacles, output_path)
        # map.visualize_obstacles(obstacles)
        
        print(f"  - Generated {len(obstacles)} obstacles for {map_name}")
        print(f"  - Saved to {output_path}")
    
    print("\nAll maps processed successfully!")
