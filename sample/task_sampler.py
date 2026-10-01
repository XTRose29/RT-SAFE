import random
import copy
from collections import deque
from pathlib import Path
from simworld.map.map import Map, Edge, Node
from simworld.utils.vector import Vector

class TaskSampler:
    def __init__(self, city_map: Map):
        """
        Initializes the sampler with a pre-built map.
        """
        self.map = city_map
        if not self.map.edges:
            raise ValueError("Map has no edges to sample from.")

    def sample_task(self, min_hops: int, max_hops: int, crosswalk_hops: int = -1, max_trials: int = 10):
        """
        Samples a navigation task with constraints on hop count and crosswalks.
        This is a convenience method that calls sample_task_with_retries with default max_trials.

        Args:
            min_hops: The minimum number of edges (total) the path should have.
            max_hops: The maximum number of edges (total) the path should have.
            crosswalk_hops: The exact number of crosswalks the path should cross.

        Returns:
            A dictionary containing the start point, end point, and path details,
            or None if no suitable path is found.
        """
        if crosswalk_hops == -1:
            return self.sample_task_simple(min_hops, max_hops, max_trials=max_trials)
        else:
            return self.sample_task_with_retries(min_hops, max_hops, crosswalk_hops, max_trials=max_trials)

    def sample_task_with_retries(self, min_hops: int, max_hops: int, crosswalk_hops: int = -1, max_trials: int = 10):
        """
        Samples a navigation task with multiple trials to increase success rate.
        
        Args:
            min_hops: The minimum number of edges (total) the path should have.
            max_hops: The maximum number of edges (total) the path should have.
            crosswalk_hops: The exact number of crosswalks the path should cross.
            max_trials: Maximum number of trials with different starting edges.
            
        Returns:
            A dictionary containing the start point, end point, and path details,
            or None if no suitable path is found after all trials.
        """
        # get all valid start edges
        valid_start_edges = [edge for edge in self.map.edges if edge.type != 'crosswalk']
        if not valid_start_edges:
            return None
        random.shuffle(valid_start_edges)
        
        for trial in range(min(max_trials, len(valid_start_edges))):
            start_edge = valid_start_edges[trial]
            
            result = self._try_sample_with_start_edge(start_edge, min_hops, max_hops, crosswalk_hops)
            
            if result is not None:
                return result
        
        return None
    
    def _try_sample_with_start_edge(self, start_edge: Edge, min_hops: int, max_hops: int, crosswalk_hops: int = -1):
        """
        Internal method to try sampling a task with a specific starting edge.
        """
        start_position = self._get_random_point_on_edge(start_edge)

        # if min_hops=1, allow start and end on the same edge
        if min_hops == 1 and max_hops == 1:
            end_position = self._get_random_point_on_edge(start_edge)
            while (end_position.x == start_position.x and end_position.y == start_position.y):
                end_position = self._get_random_point_on_edge(start_edge)
            
            return {
                "start_point": start_position,
                "end_point": end_position,
                "start_edge": start_edge,
                "end_edge": start_edge,
                "edges": [start_edge],
                "total_hops": 1,
                "crosswalk_hops": 0
            }

        # otherwise, find multi-hop paths
        candidate_paths = self._find_paths_with_bfs(start_edge, min_hops, max_hops, crosswalk_hops)

        if not candidate_paths:
            return None

        # prioritize paths with end edge not being crosswalk
        valid_candidate_paths = []
        for path in candidate_paths:
            end_edge = path[-1]
            if end_edge.type != 'crosswalk':
                valid_candidate_paths.append(path)
        
        # if no valid path, consider backtracking strategy
        if not valid_candidate_paths:
            for path in candidate_paths:
                # try to find the last non-crosswalk edge as the end edge
                for edge in reversed(path):
                    if edge.type != 'crosswalk':
                        valid_candidate_paths.append(path)
                        break
        
        if not valid_candidate_paths:
            return None
        
        # randomly select from valid candidate paths
        chosen_path = random.choice(valid_candidate_paths)
        end_edge = chosen_path[-1]
        
        # if the end edge is crosswalk, need to backtrack to the last non-crosswalk edge
        if end_edge.type == 'crosswalk':
            for edge in reversed(chosen_path):
                if edge.type != 'crosswalk':
                    end_edge = edge
                    break
            
        chosen_path = chosen_path[:chosen_path.index(end_edge) + 1]
        
        end_position = self._get_random_point_on_edge(end_edge)

        return {
            "start_point": start_position,
            "end_point": end_position,
            "start_edge": start_edge,
            "end_edge": end_edge,
            "edges": chosen_path,
            "total_hops": len(chosen_path),
            "crosswalk_hops": sum(1 for edge in chosen_path if edge.type == 'crosswalk')
        }

    def sample_task_simple(self, min_hops: int, max_hops: int, max_trials: int = 10):
        """
        Samples a navigation task with only hop count constraints.
        Ensures start and end edges are not crosswalks, but doesn't constrain crosswalk count in between.

        Args:
            min_hops: The minimum number of edges (total) the path should have.
            max_hops: The maximum number of edges (total) the path should have.
            max_trials: Maximum number of trials with different starting edges.

        Returns:
            A dictionary containing the start point, end point, and path details,
            or None if no suitable path is found.
        """
        # get all valid start edges (non-crosswalk)
        valid_start_edges = [edge for edge in self.map.edges if edge.type != 'crosswalk']
        if not valid_start_edges:
            return None
        random.shuffle(valid_start_edges)
        
        for trial in range(min(max_trials, len(valid_start_edges))):
            start_edge = valid_start_edges[trial]
            
            result = self._try_sample_simple_with_start_edge(start_edge, min_hops, max_hops)
            
            if result is not None:
                return result
        
        return None
    
    def _try_sample_simple_with_start_edge(self, start_edge: Edge, min_hops: int, max_hops: int):
        """
        Internal method to try sampling a simple task with a specific starting edge.
        """
        start_position = self._get_random_point_on_edge(start_edge)

        # if min_hops=1, allow start and end on the same edge
        if min_hops == 1 and max_hops == 1:
            end_position = self._get_random_point_on_edge(start_edge)
            while (end_position.x == start_position.x and end_position.y == start_position.y):
                end_position = self._get_random_point_on_edge(start_edge)
            
            return {
                "start_point": start_position,
                "end_point": end_position,
                "start_edge": start_edge,
                "end_edge": start_edge,
                "edges": [start_edge],
                "total_hops": 1,
                "crosswalk_hops": 0
            }

        # otherwise, find multi-hop paths
        candidate_paths = self._find_paths_simple_bfs(start_edge, min_hops, max_hops)

        if not candidate_paths:
            return None

        # prioritize paths with end edge not being crosswalk
        valid_candidate_paths = []
        for path in candidate_paths:
            end_edge = path[-1]
            if end_edge.type != 'crosswalk':
                valid_candidate_paths.append(path)
        
        # if no valid path, consider backtracking strategy
        if not valid_candidate_paths:
            for path in candidate_paths:
                # try to find the last non-crosswalk edge as the end edge
                for edge in reversed(path):
                    if edge.type != 'crosswalk':
                        valid_candidate_paths.append(path)
                        break
        
        if not valid_candidate_paths:
            return None
        
        # randomly select from valid candidate paths
        chosen_path = random.choice(valid_candidate_paths)
        end_edge = chosen_path[-1]
        
        # if the end edge is crosswalk, need to backtrack to the last non-crosswalk edge
        if end_edge.type == 'crosswalk':
            for edge in reversed(chosen_path):
                if edge.type != 'crosswalk':
                    end_edge = edge
                    break
            
        chosen_path = chosen_path[:chosen_path.index(end_edge) + 1]
        
        end_position = self._get_random_point_on_edge(end_edge)

        return {
            "start_point": start_position,
            "end_point": end_position,
            "start_edge": start_edge,
            "end_edge": end_edge,
            "edges": chosen_path,
            "total_hops": len(chosen_path),
            "crosswalk_hops": sum(1 for edge in chosen_path if edge.type == 'crosswalk')
        }

    def _find_paths_simple_bfs(self, start_edge: Edge, min_hops: int, max_hops: int):
        """
        Performs a BFS to find all paths that satisfy only hop constraints.
        """
        candidate_paths = []
        
        # The queue will store tuples of: (current_node, path_of_edges_so_far)
        q = deque()
        
        # BFS needs to start from both ends of the initial edge
        # the initial path should contain the start edge
        q.append((start_edge.node1, [start_edge]))
        q.append((start_edge.node2, [start_edge]))
        
        # Visited set is crucial to prevent cycles and redundant work.
        # We store (node, hop_count) to allow revisiting nodes via different length paths.
        visited = set()
        
        while q:
            current_node, path = q.popleft()

            # Pruning: If path is already too long, stop exploring this branch
            if len(path) >= max_hops:
                continue
            
            # Explore neighbors
            for neighbor_node in self.map.adjacency_list[current_node]:
                # find the edge connecting the current node and the neighbor node
                neighbor_edge = None
                for edge in self.map.edges:
                    if (edge.node1 == current_node and edge.node2 == neighbor_node) or \
                       (edge.node1 == neighbor_node and edge.node2 == current_node):
                        neighbor_edge = edge
                        break
                
                if neighbor_edge is None:
                    continue
                
                # avoid duplicate edges in the path
                if neighbor_edge in path:
                    continue
                
                # Determine the next node in the path
                next_node = neighbor_node
                
                # Create the new path
                new_path = path + [neighbor_edge]
                
                # Check visit status to avoid cycles
                visit_key = (next_node, len(new_path))
                if visit_key in visited:
                    continue
                visited.add(visit_key)
                
                # Check constraints - only hop count matters
                current_hops = len(new_path)
                
                if current_hops >= min_hops:
                    candidate_paths.append(new_path)
                    
                # Continue the search
                q.append((next_node, new_path))
                
        return candidate_paths

    def _get_random_point_on_edge(self, edge: Edge) -> Vector:
        """Helper to get a random position along an edge's line segment."""
        fraction = random.random()
        start_pos = edge.node1.position
        end_pos = edge.node2.position

        interpolated_x = start_pos.x + (end_pos.x - start_pos.x) * fraction
        interpolated_y = start_pos.y + (end_pos.y - start_pos.y) * fraction
        
        return Vector(interpolated_x, interpolated_y)

    def _find_paths_with_bfs(self, start_edge: Edge, min_hops: int, max_hops: int, required_crosswalks: int = -1):
        """
        Performs a BFS to find all paths that satisfy the hop and crosswalk constraints.
        """
        candidate_paths = []
        
        # The queue will store tuples of: (current_node, path_of_edges_so_far)
        q = deque()
        
        # BFS needs to start from both ends of the initial edge
        # the initial path should contain the start edge
        q.append((start_edge.node1, [start_edge]))
        q.append((start_edge.node2, [start_edge]))
        
        # Visited set is crucial to prevent cycles and redundant work.
        # We store (node, hop_count) to allow revisiting nodes via different length paths.
        visited = set()
        
        while q:
            current_node, path = q.popleft()

            # Pruning: If path is already too long, stop exploring this branch
            if len(path) >= max_hops:
                continue
            
            # Explore neighbors
            for neighbor_node in self.map.adjacency_list[current_node]:
                # find the edge connecting the current node and the neighbor node
                neighbor_edge = None
                for edge in self.map.edges:
                    if (edge.node1 == current_node and edge.node2 == neighbor_node) or \
                       (edge.node1 == neighbor_node and edge.node2 == current_node):
                        neighbor_edge = edge
                        break
                
                if neighbor_edge is None:
                    continue
                
                # avoid duplicate edges in the path
                if neighbor_edge in path:
                    continue
                
                # Determine the next node in the path
                next_node = neighbor_node
                
                # Create the new path
                new_path = path + [neighbor_edge]
                
                # Check visit status to avoid cycles
                visit_key = (next_node, len(new_path))
                if visit_key in visited:
                    continue
                visited.add(visit_key)
                
                # Check constraints
                current_hops = len(new_path)
                current_crosswalks = sum(1 for e in new_path if e.type == 'crosswalk')
                
                if current_hops >= min_hops and (required_crosswalks == -1 or current_crosswalks == required_crosswalks):
                    candidate_paths.append(new_path)
                    
                # Continue the search
                q.append((next_node, new_path))
                
        return candidate_paths


if __name__ == "__main__":
    from simworld.config import Config
    random.seed(0)
    config = Config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
    map = Map(config, None)
    map.initialize_map_from_file(r"E:\Projects\SimWorld-RBench\data\roads.json", config['traffic.sidewalk_offset'], False)
    task_sampler = TaskSampler(map)
    task = task_sampler.sample_task(3, 3)
    print(task)
    map.visualize_path([Node(task['start_point'], Vector(0, 0)), Node(task['end_point'], Vector(0, 0))])
