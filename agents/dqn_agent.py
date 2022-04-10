from torch.nn.modules import linear
from torch.nn.modules.activation import ReLU
from .agent import Agent
import numpy as np
import torch
from torch import nn
import random
import copy

class DQNAgent(Agent):
    # learning rate
    # for cartpole 0.000125
    learning_rate = 0.000250
    # gradient momentum for RMSprop
    momentum = 0
    epsilon = 1
    epsilon_annealing_steps = 5000
    epsilon_min = 0.1
    epsilon_decay = (epsilon - epsilon_min) / epsilon_annealing_steps
    discount = 0.99
    replay_start_size = 500
    replay_memory_size = 1000000
    batch_size = 32
    target_update_steps = 1000
    use_double_dqn = False

    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

        # Initialize replay memory D to capacity N
        self.replay_memory = []

        # Initialize action-value function Q with random weights theta
        self.Q = self.create_model(action_space, observation_space)

        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.learning_rate)

        # Initialize target action-value function ^Q with weights theta- = theta
        self.target_Q = copy.deepcopy(self.Q)

    def create_model(self, action_space, observation_space):
        hidden = 64
        model = nn.Sequential( 
            nn.Linear(observation_space.shape[0], hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_space.n)
        )
        return model

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random_sample() < self.epsilon:
            # With probability e select a random action a
            action = self.action_space.sample()
        else:
            # Otherwise
            with torch.no_grad():
                action = torch.argmax(self.Q(torch.FloatTensor(state))).numpy().item()

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
        self.target_Q.load_state_dict(self.Q.state_dict())

    def replay(self):
        # Sample random minibatch of transitions from replay memory D

        batch = random.sample(self.replay_memory, self.batch_size)

        state      = torch.FloatTensor([i[0] for i in batch])
        action     = torch.tensor([i[1] for i in batch])
        reward     = torch.FloatTensor([i[2] for i in batch])
        next_state = torch.FloatTensor([i[3] for i in batch])
        not_done   = torch.FloatTensor([float(i[4] == False) for i in batch])
        
        output = self.Q(state)

        q_value = torch.gather(output, 1, action.view(-1,1)).squeeze()

        target = reward + not_done * self.discount * torch.amax(self.target_Q(next_state), 1)

        # perform a gradient descent step on (y - Q)^2 with respect to the network parameters theta

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)
        self.optimizer.step()
        

    def get_greedy_action(self, state):
        # 0 or 0.05 or 0.01
        if np.random.random_sample() < 0:
            # With probability e select a random action a
            action = self.action_space.sample()
        else:
            # Otherwise
            action = np.argmax(self.model.predict_on_batch(np.array([state]))[0])

        return action

    #def save_model(self, filename="dqn_model.h5"):
    #    self.model.save(filename)

    #def load_model(self, filename="dqn_model.h5"):
    #    self.model = keras.models.load_model(filename)
