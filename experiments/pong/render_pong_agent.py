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
env = AtariARIWrapper(gym.make('Pong-v4'))

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

        obs = [state['labels']['player_x'],
               state['labels']['player_y'],
               state['labels']['enemy_x'],
               state['labels']['enemy_y'],
               state['labels']['ball_x'],
               state['labels']['ball_y']]

        # Normalise inputs in the range [0, 1]
        obs[0] = (obs[0] - 0) / (205 - 0)  # player x
        obs[1] = (obs[1] - 38) / (203 - 38)  # player y
        obs[2] = (obs[2] - 0) / (205 - 0)  # enemy x
        obs[3] = (obs[3] - 38) / (203 - 38)  # enemy y
        obs[4] = (obs[4] - 0) / (205 - 0)  # ball x
        obs[5] = (obs[5] - 44) / (207 - 44)  # ball y

        output = network.forward(obs)
        action = output_to_action[np.argmax(output)]

        img, reward, done, state = env.step(action)

env.close()
