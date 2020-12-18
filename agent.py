from abc import abstractmethod
from random import randint

import numpy as np

class Agent:
    def __init__(self, epsilon, alpha, gamma, num_state, num_actions, action_space):
        """
        Contructor
        Args:
            epsilon: The degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
            action_space: To call the random action
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions

        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space

    """
    The Base class that is implemented by
    other classes to avoid the duplicate 'choose_action'
    method
    """
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space[randint(0, self.num_actions-1)]
        else:
            action = self.action_space[np.argmax(self.Q[state, :])]
        return action

    @abstractmethod
    def update(self, prev_state, next_state, reward, prev_action, next_action):
        pass


class QLearningAgent(Agent):
    def update(self, prev_state, next_state, reward, prev_action, next_action):
        """
        Update the action value function using the Q-Learning update.
        Q(S_t, A_t) = Q(S_t, A_t) + alpha(reward + (gamma * Max Q(S_t+1, *) - Q(S_t, A_t))
        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        predict = self.Q[prev_state, prev_action]
        target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[prev_state, prev_action] += self.alpha * (target - predict)
