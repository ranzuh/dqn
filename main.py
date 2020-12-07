import gym
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
import numpy as np
import atexit
from tqdm import tqdm
import math
import os
from plot import plot

test_scores = []
rewards_per_episode = []
moving_avg = []

moving_avg_number = 100
eval_freq = 10

def create_next_folder(folder_name="data"):
    i = 0
    while True:
        try:
            os.mkdir(folder_name + "/run" + str(i))
        except FileExistsError:
            i += 1
            continue
        else:
            break

    return "data/run" + str(i)

def log_stats(folder_name="data"):
    np.savetxt(folder_name + "/episode_rewards.csv", rewards_per_episode, delimiter=", ", fmt="%d")
    np.savetxt(folder_name + "/test_scores.csv", test_scores, delimiter=", ", fmt="%d")
    np.savetxt(folder_name + "/moving_avg.csv", moving_avg, delimiter=", ", fmt="%d")


def exit_handler():
    run_folder = create_next_folder("data")
    log_stats(run_folder)
    plot(run_folder, eval_freq, moving_avg_number)

atexit.register(exit_handler)


def train(env, agent, episodes=10001, render=False, eval=True):
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
        if eval and episode % eval_freq == 0 and episode > 0:
            print("\nEvaluation - Episode:", episode)
            max_score = max(test_scores, default=-math.inf)
            evaluation = evaluate(env, agent, 10, render=False)
            if evaluation[0] > max_score:
                print("new max score", evaluation[0], max_score)
                agent.save_model(filename="dqn_model_best.h5")
            test_scores.append(evaluation[0])
            print("evaluation", evaluation)
        if episode > moving_avg_number:
            moving_avg.append(np.mean(rewards_per_episode[-moving_avg_number:]))
            # print("mean of last 100 eps", np.mean(rewards_per_episode[-100:]))

        state = env.reset()
        done = False
        total_reward = 0
        timesteps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done, total_timesteps)

            if render:
                env.render()

            state = next_state
            total_reward += reward
            timesteps += 1
            total_timesteps += 1

            if done:
                tqdm.write(
                    "Episode {} finished after {} timesteps and total reward was {}. Last 100 episode mean {}".format(
                        episode, timesteps, round(total_reward, 2),
                        round(moving_avg[-1], 2) if moving_avg != [] else 0))

        rewards_per_episode.append(total_reward)
        agent.save_model()

    print()
    print("Training complete after", episodes, "episodes")
    print()


def evaluate(env, agent, episodes=100, render=False):
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
        done = False
        total_reward = 0
        timesteps = 0

        while not done:
            action = agent.get_greedy_action(state)
            next_state, reward, done, info = env.step(action)

            if render:
                env.render()

            state = next_state

            total_reward += reward
            timesteps += 1
            if done:
                print("Episode {} finished after {} timesteps and total reward was {}".format(episode, timesteps,
                                                                                              round(total_reward, 2)))

        rewards_per_episode.append(total_reward)
        timesteps_per_episode.append(timesteps)

    return np.mean(rewards_per_episode), np.mean(timesteps_per_episode)


if __name__ == '__main__':
    # Initialize the LunarLander environment
    #env = gym.make('CartPole-v1')
    env = gym.make('LunarLander-v2')
    # Initialize and train DQN agent
    agent = DQNAgent(env.action_space, env.observation_space)
    train(env, agent, 1000)

    # Save model as dqn_model.h5
    agent.save_model()

    # Evaluate trained DQN agent
    file_name = "dqn_model_best.h5"
    agent.load_model(file_name)
    print("DQN agent")
    rewards, timesteps = evaluate(env, agent, 10, render=True)
    print("Average rewards", rewards)
    print("Average timesteps", timesteps)
    print()

    # Evaluate random agent for comparison
    print("Random agent")
    rand_agent = RandomAgent(env.action_space)
    rewards, timesteps = evaluate(env, rand_agent, 10, render=True)
    print("Average rewards", rewards)
    print("Average timesteps", timesteps)
