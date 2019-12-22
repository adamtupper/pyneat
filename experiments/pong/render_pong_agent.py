"""Run a Pong agent and render the game on screen.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python render_pong_agent.py path/to/solution.pickle
"""

import pickle
import argparse

import numpy as np
import gym
from atariari.benchmark.wrapper import AtariARIWrapper
from custom_neat.nn.recurrent import RNN

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument('agent', type=str, help="The (pickled) genome of the agent to evaluate.")
args = parser.parse_args()

# Build environment
env = AtariARIWrapper(gym.make('PongNoFrameskip-v4'))

# Build network
genome = pickle.load(open(args.agent, 'rb'))
network = RNN.create(genome)

output_to_action = {
        0: 0,  # NOOP
        1: 2,  # RIGHT
        2: 3  # LEFT
    }

for i_episode in range(1):
    img = env.reset()
    img, reward, done, state = env.step(0)
    network.reset()

    while not done:
        env.render()

        obs = [state['labels']['player_y'],
               state['labels']['enemy_y'],
               state['labels']['ball_x'],
               state['labels']['ball_y']]

        output = network.forward(obs)
        action = output_to_action[np.argmax(output)]
        print(action)

        img, reward, done, state = env.step(action)

env.close()
