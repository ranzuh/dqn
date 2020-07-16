from main import evaluate
from agents.dqn_agent import DQNAgent
import gym

env = gym.make('LunarLander-v2')
agent = DQNAgent(env.action_space, env.observation_space)
agent.load_model("dqn_model_2.h5")



agent.load_model("dqn_model_3.h5")
print("DQN agent dqn_model_3.h5")
rewards, timesteps = evaluate(env, agent, 100, render=True)
print("Average rewards", rewards)
print("Average timesteps", timesteps)
#Average rewards 238.65008710894563
#Average timesteps 367.88


print("DQN agent dqn_model_2.h5")
rewards, timesteps = evaluate(env, agent, 100, render=False)
print("Average rewards", rewards)
print("Average timesteps", timesteps)
#Average rewards 223.55018486981305
#Average timesteps 452.3