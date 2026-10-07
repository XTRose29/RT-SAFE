from simworld.agent.scooter import Scooter
from simworld.utils.vector import Vector

class RTScooter(Scooter):
    def __init__(self, position: Vector, direction: Vector, path: list[Vector]):
        super().__init__(position, direction)

        self.path = path

    def __str__(self):
        return f"Scooter(position={self.position}, direction={self.direction}, path={self.path})"
    
    def __repr__(self):
        return self.__str__()