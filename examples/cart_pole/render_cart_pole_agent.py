"""Run a CartPole agent and render the episode on screen.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python render_cart_pole_agent.py path/to/solution.pickle
"""

import pickle
import argparse

import gym
from pyneat.nn.recurrent import RNN

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument('agent', type=str, help="The (pickled) genome of the agent to evaluate.")
args = parser.parse_args()

# Build environment
env = gym.make('CartPole-v1')

# Build network
genome = pickle.load(open(args.agent, 'rb'))
network = RNN.create(genome)

outputs = []
actions = []

for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        env.render()
        output = network.forward(observation)[0]
        action = round(output)
        actions.append(action)
        outputs.append(output)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
