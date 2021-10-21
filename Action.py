from typing import Callable

class Action:
    def __init__(self, name: str, index: int, action_function: Callable) -> None:
        self.name = name
        self.index = index
        self.action_function = action_function
        
    def Call(self) -> None: # Calls the action function
        self.action_function()