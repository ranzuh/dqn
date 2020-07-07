from main import evaluate
from agents.dqn_agent import DQNAgent
import gym

env = gym.make('LunarLander-v2')
agent = DQNAgent(env.action_space, env.observation_space)
agent.load_model("dqn_model_paras.h5")

print("DQN agent")
rewards, timesteps = evaluate(env, agent, 10)
print("Average rewards", rewards)
print("Average timesteps", timesteps)