from .agent import Agent


class RandomAgent(Agent):

    def get_action(self, state):
        # Choose random action
        return self.action_space.sample()

    def get_greedy_action(self, state):
        return self.action_space.sample()
