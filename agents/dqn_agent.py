from .agent import Agent
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
import random


class DQNAgent(Agent):
    # learning rate
    alpha = 0.1
    # how often random move
    epsilon = 1
    epsilon_decay = 0.9999
    # discount future rewards
    discount = 0.
    replay_start_size = 1000
    replay_memory_size = 10000

    def __init__(self, action_space, observation_space):
        super().__init__(action_space)

        # Initialize replay memory D to capacity N

        self.replay_memory = []

        # Initialize action-value function Q with random weights theta

        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=observation_space.shape))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_space.n, activation='linear'))

        #optimizer = keras.optimizers.RMSprop()

        self.model.compile(optimizer='rmsprop',
                      loss='mse',
                      metrics=['accuracy'])

        # Initialize target action-value function ^Q with weights theta- = theta

        self.model_target = keras.Sequential()
        self.model_target.add(keras.Input(shape=observation_space.shape))
        self.model_target.add(Dense(16, activation='relu'))
        self.model_target.add(Dense(16, activation='relu'))
        self.model_target.add(Dense(16, activation='relu'))
        self.model_target.add(Dense(action_space.n, activation='linear'))
        self.model_target.set_weights(self.model.get_weights())


    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random_sample() < self.epsilon:
            # With probability e select a random action a
            action = self.action_space.sample()
        else:
            # Otherwise
            action = np.argmax(self.model.predict(np.array([state]))[0])

        return action

    def observe(self, state, action, reward, next_state, done):
        # store transition (state, action, reward, next_state, done) in replay memory D
        self.replay_memory.append((state, action, reward, next_state, done))

        if len(self.replay_memory) > self.replay_start_size:
            self.replay()
            self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

        #print(self.epsilon)

    def update_target(self):
        print(self.epsilon)
        self.model_target.set_weights(self.model.get_weights())

    def replay(self):
        # Sample random minibatch of transitions from replay memory D

        batch = random.sample(self.replay_memory, 32)

        X = []
        Y = []

        for state, action, reward, next_state, done in batch:
            y = self.model.predict_on_batch(np.array([state]))
            if done:
                y[0][action] = reward
            else:
                y[0][action] = reward + self.discount * np.amax(self.model_target.predict_on_batch(np.array([next_state]))[0])
            X.append(state)
            Y.append(y)

        # perform a gradient descent step on (y - Q)^2 with respect to the network parameters theta

        self.model.fit(np.array(X), np.squeeze(np.array(Y)), batch_size=32, epochs=1, verbose=0)

        # every C steps reset Q^ = Q

    def get_policy(self, state):
        return np.argmax(self.model.predict(np.array([state]))[0])


