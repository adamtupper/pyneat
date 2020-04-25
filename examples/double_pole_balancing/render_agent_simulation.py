"""Render a movie of a single episode for a pole balancing agent.

Example:
    Ensure that the following is executed in the `peal` conda environment.
        $ python render_agent_simulation.py path/to/solution.pickle
"""
import pickle
import argparse
import math

from pyneat.nn.recurrent import RNN

from cart_pole2 import CartPole
from movie import make_movie

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument('agent', type=str, help="The (pickled) genome of the agent to evaluate.")
args = parser.parse_args()

# Build environment
env = CartPole(population=None, markov=False)

# Print initial conditions
x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot = env.get_state()
print()
print("Initial conditions:")
print("          x = {0:.4f}".format(x))                            # Cart position
print("      x_dot = {0:.4f}".format(x_dot))                        # Cart velocity
print("    theta_1 = {0:.4f}".format(theta_1 * 180 / math.pi))      # Long pole angle
print("theta_1_dot = {0:.4f}".format(theta_1_dot * 180 / math.pi))  # Long pole velocity
print("    theta_2 = {0:.4f}".format(theta_2 * 180 / math.pi))      # Short pole angle
print("theta_2_dot = {0:.4f}".format(theta_2_dot * 180 / math.pi))  # Short pole velocity
print()

# Build network
genome = pickle.load(open(args.agent, 'rb'))
network = RNN.create(genome)

# Store system state for each time step
X = [x]
X_DOT = [x_dot]
THETA_1 = [theta_1]
THETA_1_DOT = [theta_1_dot]
THETA_2 = [theta_2]
THETA_2_DOT = [theta_2_dot]

# Run the given simulation for up to 1000 time steps
steps = 0
while steps < 1000:
    obs = env.get_scaled_state()  # Note: Velocities are not normalised!
    obs = [obs[0], obs[2], obs[3]]  # Remove velocities
    action = network.forward(obs)[0]

    if steps < 1 or steps % 2 == 0:
        action = 0.5

    print(f'Action: {action} \t Force: {(action - 0.5) * 10.0 * 2.0}')

    # Apply action to the simulated cart-pole
    env.step(action)

    # Updated stored system state
    x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot = env.get_state()
    X.append(x)
    X_DOT.append(x_dot)
    THETA_1.append(theta_1)
    THETA_1_DOT.append(theta_1_dot)
    THETA_2.append(theta_2)
    THETA_2_DOT.append(theta_2_dot)

    # Stop if the network fails to keep the cart within the position or angle limits.
    # The per-run fitness is the number of time steps the network can balance the pole
    # without exceeding these limits.
    if env.outside_bounds():
        break

    steps += 1

# Display results
print('Pole balanced for {0} of 1000 steps'.format(steps))

x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot = env.get_state()
print()
print("Final conditions:")
print("          x = {0:.4f}".format(x))
print("      x_dot = {0:.4f}".format(x_dot))
print("    theta_1 = {0:.4f}".format(theta_1 * 180 / math.pi))
print("theta_1_dot = {0:.4f}".format(theta_1_dot * 180 / math.pi))
print("    theta_2 = {0:.4f}".format(theta_2 * 180 / math.pi))
print("theta_2_dot = {0:.4f}".format(theta_2_dot * 180 / math.pi))
print()

# Create representative movie (different evaluation)
print("Making movie...")
make_movie(network, 15.0, "solution.mp4")
