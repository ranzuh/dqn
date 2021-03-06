# Deep Q-Network in OpenAI Gym LunarLander environment

Reinforcement learning agent that learns to land a rocket optimally!

![landing.gif](https://user-images.githubusercontent.com/13645811/88183990-d9a72200-cc3a-11ea-9e87-319f46316169.gif)

## Motivation

I made this because I wanted to learn about deep reinforcement learning. I had already implemented Q-learning algorithm for Taxi environment and I wanted to tackle more challening environment. Lunar lander wasn't an easy environment but it seemed fun and I also like rockets!

## Features

* Off-policy model free Q-learning with deep neural network
* Experience replay and separate target network for more stable learning
* Able to solve LunarLander environment (LunarLander-v2 is considered "solved" when the agent obtains an average reward of at least 200 over 100 consecutive episodes.)
* Uses Double DQN target for better performance

## Technologies
* Python 3.7
* OpenAI Gym 0.17.3
* Numpy 1.19.3
* matplotlib 3.3.2
* Tensorflow 1.15.4
* tqdm 4.51.0
* box2d 2.3.10


## Usage

Install the requirements
```
pip install -r requirements.txt
```
Run main.py to start training, you will get a plot if you abort the training with Ctrl-C.
```
python main.py
```
Run example.py to see pre-trained agent in action.
```
python example.py
```

## Additional info
Based on Deepmind's DQN paper:
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." (2015)

Double DQN from:
Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." (2015)

Environment
https://gym.openai.com/envs/LunarLander-v2/

## TODO
* DQN extensions
  * Prioritized experience replay
  * Dueling-DQN
* Refactoring and better documentation
