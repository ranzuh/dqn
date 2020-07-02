from .agent import Agent
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
import random


class DQNAgent(Agent):
    # learning rate
    learning_rate = 0.001
    # gradient momentum for RMSprop
    momentum = 0
    # how often random move
    epsilon = 1
    epsilon_decay = 0.99995
    # discount future rewards
    discount = 0.99
    replay_start_size = 1000
    replay_memory_size = 100000

    def __init__(self, action_space, observation_space):
        super().__init__(action_space)

        # Initialize replay memory D to capacity N
        self.replay_memory = []

        # Initialize action-value function Q with random weights theta
        self.model = self.create_model(action_space, observation_space)

        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate, momentum=self.momentum)
        self.model.compile(
            optimizer=optimizer,
            loss='huber_loss',
            metrics=['accuracy']
        )

        #self.model.summary()

        # Initialize target action-value function ^Q with weights theta- = theta
        self.model_target = self.create_model(action_space, observation_space)
        self.model_target.set_weights(self.model.get_weights())

    def create_model(self, action_space, observation_space):
        model = keras.Sequential()
        model.add(keras.Input(shape=observation_space.shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(action_space.n, activation='linear'))
        return model

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random_sample() < self.epsilon:
            # With probability e select a random action a
            action = self.action_space.sample()
        else:
            # Otherwise
            action = np.argmax(self.model.predict(np.array([state]))[0])

        return action

    def observe(self, state, action, reward, next_state, done, timesteps):
        # store transition (state, action, reward, next_state, done) in replay memory D
        self.replay_memory.append((state, action, reward, next_state, done))

        if len(self.replay_memory) > self.replay_start_size:
            self.replay()
            self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

        if timesteps % 1000 == 0:
            print("total timesteps:", timesteps)
            self.update_target()

        #print(self.epsilon)

    def update_target(self):
        print("Updated target. current epsilon:", self.epsilon)
        self.model_target.set_weights(self.model.get_weights())

    def replay(self):
        # Sample random minibatch of transitions from replay memory D

        batch = random.sample(self.replay_memory, 32)

        X = []
        Y = []

        states = np.array([i[0] for i in batch])
        next_states = np.array([i[3] for i in batch])

        y = self.model.predict_on_batch(states)
        target_next = self.model_target.predict_on_batch(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            else:
                y[i][action] = reward + self.discount * np.amax(target_next[i])
            X.append(state)
            Y.append(y[i])

        # perform a gradient descent step on (y - Q)^2 with respect to the network parameters theta

        self.model.fit(np.array(X), np.array(Y), verbose=0)

        # every C steps reset Q^ = Q

    def get_policy(self, state):
        return np.argmax(self.model.predict(np.array([state]))[0])


