from typing import List, Callable
from Action import Action
from State import State

class Enviorment:
    def __init__(self, reward_function: Callable, reset_function: Callable, terminal_point_function: Callable, states: List[State], actions: List[Action]) -> None:
        self.reward_func = reward_function
        self.reset_func = reset_function
        self.terminal_func = terminal_point_function
        self.states = states
        self.discrete_states = [state for state in states if state.discrete]
        self.continuous_states = [state for state in states if not state.discrete]
        self.actions = actions
        self.last_discrete_state = 0
        self.last_continous_state = []
        self.n_discrete = sum([state.n_discrete for state in self.discrete_states])
    
    def Run(self, action: int) -> None: # Calls given action and updates last state
        self.last_discrete_state = self.Discrete_State()
        self.last_continous_state = self.Continous_State()
        self.actions[action].Call()
        
    def Discrete_State(self) -> int: # Returns the current discrete state
        def tally(_states: List[State]):
            if len(_states) == 1:
                return _states[0].State()
            else:
                l = 1
                for state in _states[0:-1]:
                    l *= state.n_discrete
                return (l * _states[-1].State()) + tally(_states[0:-1])
                    
        return tally(self.discrete_states)
    
    def Continous_State(self) -> List[float]: # Returns the current continous state
        c_state = []
        for state in self.continuous_states:
            c_state.append(state.State())
        return c_state
    
    def Reward(self) -> float: # Returns the reward for achieving the current state
        return self.reward_func()
    
    def Terminal_Point(self) -> bool: # Returns if agent has reached a terminal point
        return self.terminal_func()
    
    def Reset(self) -> None:
        self.reset_func()