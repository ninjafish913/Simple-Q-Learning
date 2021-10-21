from typing import List
import random
from Enviorment import Enviorment

class QLearning:
    def __init__(self, enviorment: Enviorment, discount_factor: float = 0.99, learning_rate: float = 1, epsilon_greedy_strategy: bool = True, epsilon: float = 0.2, save_rewards: bool = True, epsilon_falloff: bool = True, save_actions: bool = True) -> None:
        self.env = enviorment
        self.Q = self.Initialize_QTable()
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.epsilon_greedy_strategy = epsilon_greedy_strategy
        self.epsilon = epsilon
        self.generation = 0
        self.save_rewards = save_rewards
        self.gen_rewards = []
        self.save_actions = save_actions
        self.gen_actions = []
        self.epsilon_falloff = epsilon_falloff
    
    def Initialize_QTable(self) -> List[List[float]]: # Returns a blank Q-Table
        QTable = []
        for state in range(self.env.n_discrete):
            QTable.append([])
            for _ in self.env.actions:
                QTable[state].append(0.0)
        return QTable
    
    def TD(self, s: int, a: int) -> float: # Calculates the temportal difference of an action taken at a previous state
        return self.env.Reward() + (self.gamma * max(self.Q[self.env.Discrete_State()])) - self.Q[s][a]
    
    def newQ(self, s: int, a: int) -> float: # Calculates the new Q value for an action taken at a previous state using the Bellman Equation
        return self.Q[s][a] + (self.alpha * self.TD(s,a))
    
    def Exploit(self) -> int: # Selects and performs action by selecting action with highest Q-Value for current state. Returns the selected action
        env = self.env
        action = self.Q[env.Discrete_State()].index(max(self.Q[env.Discrete_State()]))
        env.Run(action)
        return action
    
    def Explore(self) -> int: # Selects and performs action at random. Returns the selected action
        env = self.env
        action = random.randint(0, len(env.actions)-1)
        env.Run(action)
        return action

    def Epsilon_Action(self, iteration, max_iterations) -> int: # Selects and performs action by choosing to exploit or explore. Returns the selected action
        epsilon = self.epsilon
        if self.epsilon_falloff:
            if iteration > max_iterations / 2:
                epsilon = self.epsilon / 2
        
        action = 0
        if random.uniform(0, 1) < epsilon:
            action = self.Explore()
        else:
            action = self.Exploit()
        return action
    
    def Update_QValue(self, s: int, a: int) -> None: # Updates Q value for action taken a previous state
        self.Q[s][a] = self.newQ(s,a)
    
    def Save_Step(self): # Runs all save methods associated with the action training loop
        if self.save_rewards:
            self.Save_Reward()
        
        if self.save_actions:
            self.Save_Action()
    
    def Save_Reward(self): # Saves the accumulated reward of each generation
        self.gen_rewards[self.generation - 1] += self.env.Reward()
    
    def Save_Action(self): # Saves the total actions of each generation
        self.gen_actions[self.generation - 1] += 1
        
    def Training_Mode(self, max_iterations: int = 1000) -> None: # Updates Q-Values until system is fully trained
        env = self.env
        iteration = 1
        while iteration <= max_iterations:
            self.generation += 1
            
            if self.save_rewards:
                self.gen_rewards.append(0)
                
            if self.save_actions:
                self.gen_actions.append(0)
                
            while not env.Terminal_Point():
                action = 0
                if self.epsilon_greedy_strategy:
                    action = self.Epsilon_Action(iteration, max_iterations)
                else:
                    action = self.Exploit()
                    
                self.Update_QValue(env.last_discrete_state, action)
                self.Save_Step()
            
            self.env.Reset()
            iteration += 1 
    
    def Inference_Mode(self) -> None: # Runs through simulation without updating Q-Values
        env = self.env
        while not env.Terminal_Point():
            self.Exploit()