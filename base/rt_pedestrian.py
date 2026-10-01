from simworld.agent.pedestrian import Pedestrian
from simworld.utils.vector import Vector

class RTPedestrian(Pedestrian):
    def __init__(self, position: Vector, direction: Vector, waypoints: list[Vector]):
        super().__init__(position, direction)

        self.waypoints = waypoints

    def __str__(self):
        return f"Pedestrian(position={self.position}, direction={self.direction}, waypoints={self.waypoints})"
    
    def __repr__(self):
        return self.__str__()

    def set_speed(self, speed: float):
        self.speed = speed
