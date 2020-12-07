from .agent import Agent
import numpy as np

np.set_printoptions(threshold=np.inf)


class QLearningAgent(Agent):
    # learning rate
    alpha = 0.1
    # how often random move
    epsilon = 0.1
    # discount future rewards
    discount = 0.6

    def __init__(self, action_space, observation_space):
        super().__init__(action_space)

        # initialize empty Q-table
        self.Q = np.zeros((observation_space.n, action_space.n))

        # initialize state, action and reward
        self.s = self.a = self.r = None

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random_sample() < self.epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[state])

        return action

    def update(self, state, action, next_state, reward):
        # update Q-values when taking action in state leading to next_state with reward
        Q = self.Q
        s, a, s1, r = state, action, next_state, reward
        alpha, epsilon, discount = self.alpha, self.epsilon, self.discount

        Q[s, a] = Q[s, a] + alpha * (r + discount * (np.max(Q[s1])) - Q[s, a])

    def get_greedy_action(self, state):
        return np.argmax(self.Q[state])

    def printq(self):
        print(self.Q)

    def save_table(self):
        np.savetxt("table.csv", self.Q, delimiter=",")

    def load_table(self):
        self.Q = np.loadtxt(open("table.csv"), delimiter=",")
