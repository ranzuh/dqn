from main import evaluate
from agents.dqn_agent import DQNAgent
import gym

env = gym.make('CartPole-v1')
#env = gym.make('LunarLander-v2')
agent = DQNAgent(env.action_space, env.observation_space)

file_name = "dqn_model.h5"
agent.load_model(file_name)
print(file_name)
rewards, timesteps = evaluate(env, agent, 10, render=True)
print("Average rewards", rewards)
print("Average timesteps", timesteps)

#Leaderboards

#paras epsilon 0.05
#Average rewards 238.65008710894563
#Average timesteps 367.88

#paras 100eps greedy epsilon 0
#Average rewards 243.51289742601924
#Average timesteps 390.0

#best3 100eps greedy epsilon 0
#Average rewards 239.666817204311
#Average timesteps 302.73

