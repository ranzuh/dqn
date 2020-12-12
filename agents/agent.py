class Agent:

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def get_action(self, state):
        """
        Returns some action for given state
        :param state: current state
        :return action
        """
        pass

    def get_greedy_action(self, state):
        """
        Returns greedy action according to policy
        :param state: current state
        :return action
        """
        pass