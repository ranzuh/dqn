import gym
from agents.random_agent import RandomAgent
from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
import numpy as np


def train(env, agent, episodes=10001):
    """
    Train agent on environment env for given amount of episodes
    :param env: a gym environment
    :param agent: an agent from /agents
    :param episodes: integer
    """
    # These are for statistics
    total_penalties = 0
    completions = 0
    total_timesteps = 0

    for episode in range(episodes):
        if episode % 100 == 0:
            print("Episode:", episode)

        state = env.reset()
        reward = 0
        done = False
        penalties = 0
        total_reward = 0
        timesteps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done)

            env.render()

            state = next_state
            total_reward += reward

            if done:
                print("Episode {} finished after {} timesteps".format(episode, timesteps + 1))

            timesteps += 1

            if (total_timesteps + timesteps) % 100 == 0:
                agent.update_target()

        total_penalties += penalties
        total_timesteps += timesteps

    print()
    print("Training complete after", episodes, "episodes")
    print("Average penalties over episode", total_penalties / episodes)
    print("Completions", completions)
    print()


def evaluate(env, agent, episodes=100):
    """
    Evaluate agent in environment env for given amount of episodes.
    :param env: a gym environment
    :param agent: an agent from /agents
    :param episodes: integer
    :return: (x,y,z) tuple where
                x: mean for rewards per episode
                y: mean for penalties per episode
                z: mean for timesteps per episode
    """

    # these are for statistics
    completions = 0
    rewards_per_episode = []
    timesteps_per_episode = []
    penalties_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        reward = 0
        done = False
        penalties = 0
        total_reward = 0
        timesteps = 0

        while not done:
            action = agent.get_policy(state)
            next_state, reward, done, info = env.step(action)

            env.render()

            state = next_state

            total_reward += reward
            timesteps += 1

            if reward == -10:
                penalties += 1

            if reward == 20:
                completions += 1

        penalties_per_episode.append(penalties)
        rewards_per_episode.append(total_reward)
        timesteps_per_episode.append(timesteps)

    return np.mean(rewards_per_episode), np.mean(penalties_per_episode), np.mean(timesteps_per_episode)


if __name__ == '__main__':
    # Initialize the taxi environment
    env = gym.make('LunarLander-v2')

    # Initialize and train DQN agent
    agent = DQNAgent(env.action_space, env.observation_space)
    train(env, agent, 101)

    #Evaluate trained DQN agent
    print("DQN agent")
    rewards, penalties, timesteps = evaluate(env, agent, 10)
    print("Average rewards", rewards)
    print("Average penalties", penalties)
    print("Average timesteps", timesteps)
    print()

    # Evaluate random agent for comparison
    print("Random agent")
    rand_agent = RandomAgent(env.action_space)
    rewards, penalties, timesteps = evaluate(env, rand_agent, 5)
    print("Average rewards", rewards)
    print("Average penalties", penalties)
    print("Average timesteps", timesteps)
