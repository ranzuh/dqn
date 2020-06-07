class Agent:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        """
        Returns some action for given state
        :param state: current state
        :return action
        """
        pass

    def get_policy(self, state):
        """
        Returns action according to policy
        :param state: current state
        :return action
        """
        pass