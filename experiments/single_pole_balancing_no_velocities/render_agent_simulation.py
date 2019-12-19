"""Render a movie of a single episode for a pole balancing agent.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python render_agent_simulation.py path/to/solution.pickle
"""
import pickle
import argparse

from custom_neat.nn.recurrent import RNN

from cart_pole import CartPole, continuous_actuator_force
from movie import make_movie

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument('agent', type=str, help="The (pickled) genome of the agent to evaluate.")
args = parser.parse_args()

# Build environment
env = CartPole()

# Print initial conditions
x, x_dot, theta, theta_dot = env.get_state()
print()
print("Initial conditions:")
print("        x = {0:.4f}".format(x))
print("    x_dot = {0:.4f}".format(x_dot))
print("    theta = {0:.4f}".format(theta))
print("theta_dot = {0:.4f}".format(theta_dot))
print()

# Build network
genome = pickle.load(open(args.agent, 'rb'))
network = RNN.create(genome)

# Store system state for each time step
X = [x]
DX = [x_dot]
THETA = [theta]
DTHETA = [theta_dot]

# Run the given simulation for up to 120 seconds.
balance_time = 0.0
while env.t < 120.0:
    observation = env.get_scaled_state()
    observation = [observation[0], observation[2]]  # Remove velocities
    action = network.forward(observation)

    # Apply action to the simulated cart-pole
    force = continuous_actuator_force(action)
    env.step(force)

    # Updated stored system state
    x, x_dot, theta, theta_dot = env.get_state()
    X.append(x)
    DX.append(x)
    THETA.append(x)
    DTHETA.append(x)

    # Stop if the network fails to keep the cart within the position or angle limits.
    # The per-run fitness is the number of time steps the network can balance the pole
    # without exceeding these limits.
    if abs(env.x) >= env.position_limit or abs(env.theta) >= env.angle_limit_radians:
        break

    balance_time = env.t

# Display results
print('Pole balanced for {0:.1f} of 120.0 seconds'.format(balance_time))

x, x_dot, theta, theta_dot = env.get_state()
print()
print("Final conditions:")
print("        x = {0:.4f}".format(x))
print("    x_dot = {0:.4f}".format(x_dot))
print("    theta = {0:.4f}".format(theta))
print("theta_dot = {0:.4f}".format(theta_dot))
print()
print("Making movie...")

# Create representative movie (different evaluation
make_movie(network, continuous_actuator_force, 15.0, "solution.mp4")
