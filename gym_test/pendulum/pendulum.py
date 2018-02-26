import numpy as np
import json

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Create an OpenAIgym environment
env = OpenAIGym('Pendulum-v0', visualize=False)

network_path = './pendulum_ppo_network.json'
agent_path = './pendulum_ppo.json'
with open(network_path, 'r') as fp:
    network_spec = json.load(fp=fp)
with open(agent_path, 'r') as fp:
    agent_config = json.load(fp=fp)
agent = Agent.from_spec(
    spec=agent_config,
    kwargs=dict(
        states=env.states,
        actions=env.actions,
        network=network_spec
    )
)

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=5000, max_episode_timesteps=200, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
