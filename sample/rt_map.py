from simworld.map.map import Map, Edge, Node
from simworld.utils.vector import Vector
from simworld.config import Config
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget
import sys
import random

class RTMap(Map):
    def __init__(self, config: Config, traffic_signals: list = None):
        super().__init__(config, traffic_signals)
    
    def visualize_obstacles(self, obstacles=None):
        """Visualize the map with obstacles highlighted.
        
        Args:
            obstacles: List of obstacle dictionaries, if None will find all obstacle nodes
        """
        if obstacles is None:
            # Find all nodes with obstacles
            obstacle_nodes = [node for node in self.nodes if hasattr(node, 'obstacle') and node.obstacle]
            obstacles = [{'node': node, 'type': 'obstacle'} for node in obstacle_nodes]
        
        class ObstacleViewer(QWidget):
            def __init__(self, nodes, edges, obstacles):
                super().__init__()
                self.nodes = nodes
                self.edges = edges
                self.obstacles = obstacles
                self.setMinimumSize(800, 800)
                self.setWindowTitle('Map Visualization with Obstacles')
                self._set_bounds()
                
                self.scale = 1.0
                self.offset_x = 0
                self.offset_y = 0
                self.last_mouse_pos = None

            def _set_bounds(self):
                self.min_x = min(node.position.x for node in self.nodes)
                self.max_x = max(node.position.x for node in self.nodes)
                self.min_y = min(node.position.y for node in self.nodes)
                self.max_y = max(node.position.y for node in self.nodes)

            def paintEvent(self, event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                width, height, margin = self.width(), self.height(), 50
                scale_x = (width - 2 * margin) / (self.max_x - self.min_x) if self.max_x > self.min_x else 1
                scale_y = (height - 2 * margin) / (self.max_y - self.min_y) if self.max_y > self.min_y else 1
                base_scale = min(scale_x, scale_y) * self.scale

                painter.translate(self.offset_x, self.offset_y)

                # Draw legend
                self._draw_legend(painter, margin)

                # Draw edges by type
                for edge in self.edges:
                    edge_type = getattr(edge, 'type', 'sidewalk')
                    if edge_type == 'sidewalk':
                        color = QColor(100, 200, 100)
                        width = 2
                    elif edge_type == 'crosswalk':
                        color = QColor(100, 100, 255)
                        width = 3
                    else:
                        color = QColor(200, 200, 200)
                        width = 1

                    painter.setPen(QPen(color, width))
                    x1 = margin + (edge.node1.position.x - self.min_x) * base_scale
                    y1 = margin + (edge.node1.position.y - self.min_y) * base_scale
                    x2 = margin + (edge.node2.position.x - self.min_x) * base_scale
                    y2 = margin + (edge.node2.position.y - self.min_y) * base_scale
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

                # Draw all non-obstacle nodes (same color)
                for node in self.nodes:
                    if not (hasattr(node, 'obstacle') and node.obstacle):
                        painter.setPen(QPen(Qt.gray, 4))
                        x = margin + (node.position.x - self.min_x) * base_scale
                        y = margin + (node.position.y - self.min_y) * base_scale
                        painter.drawPoint(int(x), int(y))

                # Draw obstacle nodes (highlighted)
                for obstacle in self.obstacles:
                    node = obstacle['node']
                    obstacle_type = obstacle.get('type', 'obstacle')
                    
                    # Choose color based on obstacle type
                    if obstacle_type == 'tree':
                        color = QColor(0, 150, 0)  # Dark green for trees
                        size = 8
                    elif obstacle_type == 'other':
                        color = QColor(0, 0, 255)  # Blue for other obstacles
                        size = 8
                    else:
                        color = QColor(255, 0, 0)  # Red for unknown obstacles
                        size = 6
                    
                    painter.setPen(QPen(color, size))
                    x = margin + (node.position.x - self.min_x) * base_scale
                    y = margin + (node.position.y - self.min_y) * base_scale
                    painter.drawPoint(int(x), int(y))

            def wheelEvent(self, event):
                angle = event.angleDelta().y()
                factor = 1.15 if angle > 0 else 0.85
                old_scale = self.scale
                self.scale *= factor
                mouse_pos = event.pos()
                dx = mouse_pos.x() - self.offset_x
                dy = mouse_pos.y() - self.offset_y
                self.offset_x -= dx * (self.scale - old_scale) / old_scale
                self.offset_y -= dy * (self.scale - old_scale) / old_scale
                self.update()

            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = event.pos()

            def mouseMoveEvent(self, event):
                if self.last_mouse_pos is not None:
                    delta = event.pos() - self.last_mouse_pos
                    self.offset_x += delta.x()
                    self.offset_y += delta.y()
                    self.last_mouse_pos = event.pos()
                    self.update()

            def mouseReleaseEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.last_mouse_pos = None

            def _draw_legend(self, painter, margin):
                """Draw a legend showing obstacles."""
                # Set font for legend text
                font = painter.font()
                font.setPointSize(10)
                painter.setFont(font)

                # Legend position (top-right corner)
                legend_x = self.width() - 200
                legend_y = margin + 20

                # Draw legend background
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setBrush(QColor(255, 255, 255, 200))
                painter.drawRect(legend_x - 10, legend_y - 10, 190, 100)

                # Draw legend title
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x, legend_y, 'Legend')

                # Draw node types
                legend_y += 25

                # Regular nodes
                painter.setPen(QPen(Qt.gray, 4))
                painter.drawPoint(legend_x + 10, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Regular Nodes')

                # Draw obstacle types
                legend_y += 25

                # Tree obstacle
                painter.setPen(QPen(QColor(0, 150, 0), 8))
                painter.drawPoint(legend_x + 10, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Tree Obstacle')

                # Other obstacle
                legend_y += 20
                painter.setPen(QPen(QColor(255, 165, 0), 6))
                painter.drawPoint(legend_x + 10, legend_y)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(legend_x + 25, legend_y + 5, 'Other Obstacle')

        app = QApplication.instance() or QApplication(sys.argv)
        viewer = ObstacleViewer(list(self.nodes), list(self.edges), obstacles)
        viewer.show()
        app.exec_()


    def _interpolate_nodes(self, num_waypoints_normal: int, waypoints_distance: float, waypoints_normal_distance: float, sidewalk_offset: float):
        """
        Interpolate normal nodes between existing nodes along each edge.
        For each edge, insert a node every waypoints_distance.
        Each interpolation point is classified as crosswalk or sidewalk based on the length of the edge.
        Sidewalk points are further classified as near_road, middle, or far_road.

        Side effect: populates self.sidewalk_groups — a list of dicts, one per
        sidewalk edge, each containing:
            {
                "start": Vector,   # intersection start position
                "end": Vector,     # intersection end position
                "far_road": [Node, ...],
                "middle":   [Node, ...],
                "near_road":[Node, ...],
            }
        """
        self.sidewalk_groups = []  # reset every time

        # Copy the current edges to avoid modifying the set during iteration
        original_edges = list(self.edges)

        for edge in original_edges:
            start = edge.node1.position
            end = edge.node2.position
            direction = (end - start).normalize()
            length = start.distance(end)
            num_points = int(length // waypoints_distance)

            is_crosswalk = abs(length - 2 * sidewalk_offset) < 1e-3  
            node_type = 'crosswalk' if is_crosswalk else 'sidewalk'

            # The first and last nodes are intersections (only one node)
            intersection_start = edge.node1
            intersection_end = edge.node2

            # Store layers of nodes for connection
            layers = []

            # Add the first intersection node as the first layer (single node)
            layers.append([intersection_start])

            # If the edge is too short, insert only one node in the middle
            if num_points < 2:
                pos = start + direction * (length / 2)
                normal = Vector(-direction.y, direction.x)
                # Only 3 points: near, middle, far with random interpolation
                offsets = [
                    2 * waypoints_normal_distance,  # far (left side)
                    0,  # middle (random interpolation)
                    -2 * waypoints_normal_distance     # near (right side)
                ]
                nodes = [Node(pos + normal * offset, direction, node_type) for offset in offsets]
                if node_type == 'sidewalk':
                    # Assign types directly based on position
                    nodes[0].type = 'sidewalk_far_road'    # far (left side)
                    nodes[1].type = 'sidewalk_middle'      # middle (random)
                    nodes[2].type = 'sidewalk_near_road'   # near (right side)
                for node in nodes:
                    self.add_node(node)
                layers.append(nodes)
            else:
                for i in range(1, num_points):
                    pos = start + direction * (i * waypoints_distance)
                    normal = Vector(-direction.y, direction.x)
                    if node_type == 'sidewalk':
                        # Only 3 points: near, middle, far with random interpolation
                        offsets = [
                            2 * waypoints_normal_distance,  # far (left side)
                            random.uniform(-1.5, 1.5) * waypoints_normal_distance,  # middle (random interpolation)
                            -2 * waypoints_normal_distance     # near (right side)
                        ]
                        nodes = [Node(pos + normal * offset, direction, node_type) for offset in offsets]
                        # Assign types directly based on position
                        nodes[0].type = 'sidewalk_far_road'    # far (left side)
                        nodes[1].type = 'sidewalk_middle'      # middle (random)
                        nodes[2].type = 'sidewalk_near_road'   # near (right side)
                    else:
                        # For non-sidewalk, keep the same 3-point structure with random interpolation
                        offsets = [
                            2 * waypoints_normal_distance,  # far (left side)
                            0,  # middle (random interpolation)
                            -2 * waypoints_normal_distance     # near (right side)
                        ]
                        nodes = [Node(pos + normal * offset, direction, node_type) for offset in offsets]
                    for node in nodes:
                        self.add_node(node)
                    layers.append(nodes)

            # Add the last intersection node as the last layer (single node)
            layers.append([intersection_end])

            # ── Record sidewalk group (three ordered lists) ──
            if node_type == 'sidewalk':
                group = {
                    "start": start,
                    "end": end,
                    "far_road": [],
                    "middle": [],
                    "near_road": [],
                }
                # layers[0] and layers[-1] are single-node intersection layers;
                # the middle layers each have 3 nodes: [far, middle, near]
                for layer in layers[1:-1]:
                    if len(layer) == 3:
                        group["far_road"].append(layer[0])
                        group["middle"].append(layer[1])
                        group["near_road"].append(layer[2])
                self.sidewalk_groups.append(group)

            # Connect nodes between consecutive layers
            for i in range(len(layers) - 1):
                current_layer = layers[i]
                next_layer = layers[i + 1]
                for node_a in current_layer:
                    for node_b in next_layer:
                        # Avoid duplicate edges
                        if not self.has_edge(Edge(node_a, node_b)):
                            self.add_edge(Edge(node_a, node_b))

            # Remove the original long edge
            if edge in self.edges:
                self.edges.remove(edge)
                # Optionally, also remove from adjacency_list if needed
                if edge.node2 in self.adjacency_list.get(edge.node1, []):
                    self.adjacency_list[edge.node1].remove(edge.node2)
                if edge.node1 in self.adjacency_list.get(edge.node2, []):
                    self.adjacency_list[edge.node2].remove(edge.node1)

    def get_sidewalk_groups(self):
        """Return sidewalk groups recorded during interpolation.

        Each group is a dict:
            {
                "start": Vector,
                "end": Vector,
                "far_road": [Node, ...],
                "middle":   [Node, ...],
                "near_road":[Node, ...],
            }
        """
        return getattr(self, 'sidewalk_groups', [])

    def get_sidewalk_near_road_nodes(self):
        return [node for node in self.nodes if getattr(node, 'type', None) == 'sidewalk_near_road']
    
    def get_sidewalk_middle_nodes(self):
        return [node for node in self.nodes if getattr(node, 'type', None) == 'sidewalk_middle']
    
    def get_sidewalk_far_road_nodes(self):
        return [node for node in self.nodes if getattr(node, 'type', None) == 'sidewalk_far_road']
    
    def get_normal_nodes(self):
        return [node for node in self.nodes if getattr(node, 'type', None) == 'normal']
    
    def get_crosswalk_nodes(self):
        return [node for node in self.nodes if getattr(node, 'type', None) == 'crosswalk']

    def get_intersection_nodes(self):
        return [node for node in self.nodes if getattr(node, 'type', None) == 'intersection']
    

    def _point_to_segment_distance(self, p, a, b):
        ab = b - a
        ap = p - a
        ab_len2 = ab.x ** 2 + ab.y ** 2
        if ab_len2 == 0:
            return ap.length()
        t = max(0, min(1, (ap.x * ab.x + ap.y * ab.y) / ab_len2))
        proj = a + ab * t
        return (p - proj).length()

    def _nearest_road_distance(self, node):
        return min(self._point_to_segment_distance(node.position, road.start, road.end) for road in self.roads)