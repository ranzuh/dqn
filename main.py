import gym
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import atexit
from tqdm import tqdm

test_scores = []
rewards_per_episode = []
moving_avg = []

moving_avg_number = 100
eval_freq = 20

def exit_handler():
    global test_scores
    plt.plot(rewards_per_episode)
    plt.plot([i for i in range(0, len(test_scores) * eval_freq, eval_freq)], test_scores)
    plt.plot([i for i in range(moving_avg_number, moving_avg_number + len(moving_avg))], moving_avg)
    plt.ylabel('Average reward')
    plt.xlabel('Episodes')
    plt.show()

atexit.register(exit_handler)

def train(env, agent, episodes=10001):
    """
    Train agent on environment env for given amount of episodes
    :param env: a gym environment
    :param agent: an agent from /agents
    :param episodes: integer
    """
    # These are for statistics
    global test_scores
    global rewards_per_episode
    global moving_avg
    total_timesteps = 0

    for episode in tqdm(range(episodes)):
        if episode % eval_freq == 0:
            print("Episode:", episode)
            evaluation = evaluate(env, agent, 5)
            test_scores.append(evaluation[0])
            print("evaluation", evaluation)
        if episode > moving_avg_number:
            moving_avg.append(np.mean(rewards_per_episode[-moving_avg_number:]))
            # print("mean of last 100 eps", np.mean(rewards_per_episode[-100:]))

        state = env.reset()
        reward = 0
        done = False
        total_reward = 0
        timesteps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done, total_timesteps)
            #env.render()

            state = next_state
            total_reward += reward
            timesteps += 1
            total_timesteps += 1

            if done:
                tqdm.write("Episode {} finished after {} timesteps and total reward was {}. Last 100 episode mean {}".format(episode, timesteps, round(total_reward, 2), round(moving_avg[-1], 2) if moving_avg != [] else 0))

        rewards_per_episode.append(total_reward)

    print()
    print("Training complete after", episodes, "episodes")
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
    rewards_per_episode = []
    timesteps_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        reward = 0
        done = False
        total_reward = 0
        timesteps = 0

        while not done:
            action = agent.get_policy(state)
            next_state, reward, done, info = env.step(action)

            env.render()

            state = next_state

            total_reward += reward
            timesteps += 1
            if done:
                print("Episode {} finished after {} timesteps and total reward was {}".format(episode, timesteps, round(total_reward, 2)))

        rewards_per_episode.append(total_reward)
        timesteps_per_episode.append(timesteps)

    return np.mean(rewards_per_episode), np.mean(timesteps_per_episode)


if __name__ == '__main__':
    # Initialize the LunarLander environment
    env = gym.make('LunarLander-v2')
    # Initialize and train DQN agent
    agent = DQNAgent(env.action_space, env.observation_space)
    train(env, agent, 600)

    # Save model as dqn_model.h5
    agent.save_model()

    #Evaluate trained DQN agent
    print("DQN agent")
    rewards, timesteps = evaluate(env, agent, 10)
    print("Average rewards", rewards)
    print("Average timesteps", timesteps)
    print()

    # Evaluate random agent for comparison
    print("Random agent")
    rand_agent = RandomAgent(env.action_space)
    rewards, timesteps = evaluate(env, rand_agent, 10)
    print("Average rewards", rewards)
    print("Average timesteps", timesteps)


