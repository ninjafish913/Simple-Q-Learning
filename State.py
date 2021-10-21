from typing import List, Callable

class State:
    def __init__(self, name: str, index: int, state_function: Callable, discrete: bool = True, n_discrete: int = 0) -> None:
        self.name = name
        self.index = index
        self.discrete = discrete
        self.n_discrete = n_discrete
        self.state_func = state_function
    
    def State(self): # Returns the value of the state
        return self.state_func()