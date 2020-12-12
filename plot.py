import matplotlib.pyplot as plt
import numpy as np

def plot(folder_name, eval_freq, moving_avg_number):
    rewards_per_episode = np.genfromtxt(folder_name + "/episode_rewards.csv")
    test_scores = np.genfromtxt(folder_name + "/test_scores.csv")
    moving_avg = np.genfromtxt(folder_name + "/moving_avg.csv")

    plt.plot(rewards_per_episode)
    plt.plot([i for i in range(eval_freq, test_scores.size * eval_freq + 1, eval_freq)], test_scores)
    plt.plot([i for i in range(moving_avg_number, moving_avg_number + moving_avg.size)], moving_avg)
    plt.ylabel('Average reward')
    plt.xlabel('Episodes')
    plt.show()

if __name__ == '__main__':
    plot("data/run15", 20, 100)
