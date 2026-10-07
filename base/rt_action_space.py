import json
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from simworld.utils.logger import Logger

# Action types
MOVE_TO = "move_to"
TURN_AROUND = "turn_around"
WAIT = "wait"

# Valid parameters for each action type
MOVE_TO_PARAMS = ['1', '2', '3', '4', '5', '6', '7']
TURN_AROUND_PARAMS = ['L30', 'L60', 'L90', 'R30', 'R60', 'R90']
WAIT_PARAMS = ['1', '2', '3']


class RTActionSpace(BaseModel):
    """
    Action space for the RT agent.
    
    Three action types, each with a specific parameter:
    - move_to: param in ['1','2','3','4','5','6','7'] (waypoint index)
    - turn_around: param in ['L30','L60','L90','R30','R60','R90']
    - wait: param in ['1','2','3'] (seconds)
    """
    action_type: Optional[str] = None
    action_param: Optional[str] = None
    reasoning: Optional[str] = None

    def __str__(self):
        output = f'Action: {self.action_type}, Param: {self.action_param}'
        if self.reasoning:
            output += f', Reasoning: {self.reasoning}'
        return output
    
    def __repr__(self):
        return self.__str__()

    def is_valid(self) -> bool:
        """Check if the action is valid."""
        if not self.action_type or not self.action_param:
            return False
        if self.action_type == MOVE_TO:
            return self.action_param in MOVE_TO_PARAMS
        if self.action_type == TURN_AROUND:
            return self.action_param in TURN_AROUND_PARAMS
        if self.action_type == WAIT:
            return self.action_param in WAIT_PARAMS
        return False

    @classmethod
    def from_json(cls, json_str):
        """Parse the action space from a json string."""
        if isinstance(json_str, str):
            try:
                json_str = json.loads(json_str)
            except Exception as e:
                Logger.get_logger('RTActionSpace').warning(f'Parse action space from json failed: {e}')
                return cls()
        try:
            action_type = json_str.get('action_type')
            action_param = json_str.get('action_param')
            reasoning = json_str.get('reasoning')
            return cls(action_type=action_type, action_param=action_param, reasoning=reasoning)
        except Exception as e:
            Logger.get_logger('RTActionSpace').warning(f'Parse action space from json failed: {e}')
            return cls()
