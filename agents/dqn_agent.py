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
    epsilon_annealing_steps = 30000
    epsilon_min = 0.1
    epsilon_decay = (epsilon - epsilon_min) / epsilon_annealing_steps
    # discount future rewards
    discount = 0.99
    replay_start_size = 1000
    replay_memory_size = 1000000
    batch_size = 32
    target_update_steps = 2000
    use_double_dqn = True

    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

        # Initialize replay memory D to capacity N
        self.replay_memory = []

        # Initialize action-value function Q with random weights theta
        self.model = self.create_model(action_space, observation_space)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # huber_loss or mse
            metrics=['accuracy']
        )

        # self.model.summary()

        # Initialize target action-value function ^Q with weights theta- = theta
        self.model_target = self.create_model(action_space, observation_space)
        self.model_target.set_weights(self.model.get_weights())

    def create_model(self, action_space, observation_space):
        model = keras.Sequential()
        model.add(keras.Input(shape=observation_space.shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        model.add(Dense(action_space.n, activation='linear'))
        model.summary()
        return model

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random_sample() < self.epsilon:
            # With probability e select a random action a
            action = self.action_space.sample()
        else:
            # Otherwise
            action = np.argmax(self.model.predict_on_batch(np.array([state]))[0])

        return action

    def observe(self, state, action, reward, next_state, done, timesteps):
        # store transition (state, action, reward, next_state, done) in replay memory D
        self.replay_memory.append((state, action, reward, next_state, done))

        if len(self.replay_memory) > self.replay_start_size:
            self.replay()
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

        if timesteps % self.target_update_steps == 0:
            # print("total timesteps:", timesteps)
            self.update_target()
        if timesteps % 10000 == 0:
            print("\n epsilon", self.epsilon, "timesteps", timesteps)

    def update_target(self):
        # print("Updated target. current epsilon:", self.epsilon)
        self.model_target.set_weights(self.model.get_weights())

    def replay(self):
        # Sample random minibatch of transitions from replay memory D

        batch = random.sample(self.replay_memory, self.batch_size)

        X = []
        Y = []

        states = np.array([i[0] for i in batch])
        next_states = np.array([i[3] for i in batch])

        y = self.model.predict_on_batch(states)
        target_next = self.model_target.predict_on_batch(next_states)

        y_next = None
        if self.use_double_dqn:
            y_next = self.model.predict_on_batch(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            elif self.use_double_dqn:
                y[i][action] = reward + self.discount * target_next[i][np.argmax(y_next[i])]
            else:
                y[i][action] = reward + self.discount * np.amax(target_next[i])

            X.append(state)
            Y.append(y[i])

        # perform a gradient descent step on (y - Q)^2 with respect to the network parameters theta

        self.model.train_on_batch(np.array(X), np.array(Y))

        # every C steps reset Q^ = Q

    def get_greedy_action(self, state):
        # 0 or 0.05 or 0.01
        if np.random.random_sample() < 0:
            # With probability e select a random action a
            action = self.action_space.sample()
        else:
            # Otherwise
            action = np.argmax(self.model.predict_on_batch(np.array([state]))[0])

        return action

    def save_model(self, filename="dqn_model.h5"):
        self.model.save(filename)

    def load_model(self, filename="dqn_model.h5"):
        self.model = keras.models.load_model(filename)
