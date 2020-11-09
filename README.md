# Deep Q-Network in OpenAI Gym LunarLander environment

Reinforcement learning agent that learns to land a rocket optimally!

![landing.gif](https://user-images.githubusercontent.com/13645811/88183990-d9a72200-cc3a-11ea-9e87-319f46316169.gif)

## Features

* Off-policy model free Q-learning with deep neural network
* Experience replay and separate target network for more stable learning
* Able to solve LunarLander environment (LunarLander-v2 is considered "solved" when the agent obtains an average reward of at least 200 over 100 consecutive episodes.)

## Usage

Run main.py to start training, you will get a plot if you ctrl-c.

Run example.py to see pre-trained agent in action.


### Additional info
Based on
"Human-level control through deep reinforcement learning." by Mnih, Volodymyr, et al. (2015)

Environment
https://gym.openai.com/envs/LunarLander-v2/

### TODO
* DQN extensions
  * Double-DQN
  * Prioritized experience replay
  * Dueling-DQN
* Refactoring and better documentation
