"""Evolve an agent to solve the pole balancing problem without velocities.

The agent is provided with two inputs (position of cart and the angle of pole).
It must learn one output (0 to move the cart left, 1 to move the cart right).
The problem is considered solved if the agent can achieve an average reward of
>= 475 over 100 consecutive episodes.

This problem is harder than the standard pole balancing problem as the agent is
required to learn the speeds, which requires recurrence.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python cart_pole_no_velocities.py path/to/config.ini

    To safely stop an evolutionary run early, interrupt the program as the
    population statistics are reported for a generation.
"""
import os
import pickle
import argparse
import signal
import sys
from collections import deque

import ray
import neat

from custom_neat.population import Population
from custom_neat.nn.recurrent import RNN
from custom_neat.genome import Genome
from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet
from custom_neat.config import CustomConfig

import visualize
# import cart_pole

# start cart_pole.py ###########################################################
"""General settings and implementation of the single-pole cart system dynamics.

Adapted from the example provided with NEAT-Python.
"""

from math import cos, pi, sin
import random

class CartPole(object):
    gravity = 9.8  # Acceleration due to gravity, positive is downward, m/sec^2
    mcart = 1.0  # Cart mass in kg
    mpole = 0.1  # Pole mass in kg
    lpole = 0.5  # Half the pole length in meters
    time_step = 0.01  # Time step size in seconds

    def __init__(self, x=None, dx=None, theta=None, dtheta=None,
                 position_limit=4.8, angle_limit=36):
        """Initialise the system.

        Args:
            x: The cart position in meters.
            dx: The cart velocity in meters per second.
            theta: The pole angle in degrees.
            dtheta: The pole angular velocity in degrees per second.
            position_limit: The cart position limit in meters.
            angle_limit: The pole angle limit in degrees.
        """
        self.position_limit = position_limit
        self.angle_limit = angle_limit

        # Initialise system, randomly if starting state not given
        if x is None:
            x = random.uniform(-0.5 * self.position_limit, 0.5 * self.position_limit)

        if theta is None:
            theta = random.uniform(-0.5 * self.angle_limit, 0.5 * self.angle_limit)

        if dx is None:
            dx = random.uniform(-1.0, 1.0)

        if dtheta is None:
            dtheta = random.uniform(-1.0, 1.0)

        self.t = 0.0
        self.x = x
        self.dx = dx
        self.theta = theta
        self.dtheta = dtheta

        self.xacc = 0.0
        self.tacc = 0.0

        # Convert angular measurements to radians for dynamics calculations
        self.angle_limit_radians = angle_limit * (pi / 180)
        self.theta = self.theta * (pi / 180)
        self.dtheta = self.dtheta * (pi / 180)

    def step(self, force):
        """Update the system state using leapfrog integration.

        Equations:

            x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
            v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt

        Args:
            force (float): The force applied by the agent on the cart. Measured
                in newtons.
        """
        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole
        dt = self.time_step

        # Remember acceleration from previous step.
        tacc0 = self.tacc
        xacc0 = self.xacc

        # Update position/angle.
        self.x += dt * self.dx + 0.5 * xacc0 * dt ** 2
        self.theta += dt * self.dtheta + 0.5 * tacc0 * dt ** 2

        # Compute new accelerations as given in "Correct equations for the dynamics of the cart-pole system"
        # by Razvan V. Florian (http://florian.io).
        # http://coneural.org/florian/papers/05_cart_pole.pdf
        st = sin(self.theta)
        ct = cos(self.theta)
        tacc1 = (g * st + ct * (-force - mp * L * self.dtheta ** 2 * st) / mt) / (L * (4.0 / 3 - mp * ct ** 2 / mt))
        xacc1 = (force + mp * L * (self.dtheta ** 2 * st - tacc1 * ct)) / mt

        # Update velocities.
        self.dx += 0.5 * (xacc0 + xacc1) * dt
        self.dtheta += 0.5 * (tacc0 + tacc1) * dt

        # Remember current acceleration for next step.
        self.tacc = tacc1
        self.xacc = xacc1
        self.t += dt

    def get_scaled_state(self):
        """Get system state, scaled into (approximately) [-1, 1].

        Returns:
            list: The scaled system state [x, dx, theta, dtheta].
        """
        return [self.x / self.position_limit,
                self.dx / 4.0,  # Assuming max velocity = 4.0 m/s
                (self.theta + self.angle_limit_radians) / self.angle_limit_radians,  # TODO: Check max angular velocity
                self.dtheta / self.angle_limit_radians]

    def get_angle_limit(self):
        """Return the angle limit in degrees.

        Returns:
            float: The angle limit in degrees.
        """
        return self.angle_limit_radians * (180 / pi)

    def get_state(self):
        """Return the state of the system.

        Units:
            - x (the cart position) is measured in meters.
            - dx (the cart velocity) is measured in meters per second.
            - theta (the pole angle) is measure in degrees.
            - dtheta (the pole angular velocity) is measured in degrees per
              second.

        Returns:
            tuple: The state of the system (x, dx, theta, dtheta)
        """
        return self.x, self.dx, self.theta * (180 / pi), self.dtheta * (180 / pi)


def continuous_actuator_force(action):
    """Convert the network output to a continuous force to be applied to the
    cart.

    Args:
        action (list): A scalar float vector in the range [-1, 1].

    Returns:
        float: The force to be applied to the cart in the range [-10, 10] N.
    """
    return 10.0 * action[0]


def noisy_continuous_actuator_force(action):
    """

    # TODO: Complete function docstring.
    # TODO: Check that the implementation conforms to my requirements.

    Args:
        action:

    Returns:

    """
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0


def discrete_actuator_force(action):
    """Convert the network action to a discrete force applied to the cart.

    Args:
        action ([float]): The action of the agent. Must be a scalar value in the
            range [-1, 1].

    Returns:
        float: A force of either 5 N (move cart right) or -5 N (move cart left)
            to be applied to the cart.
    """
    return 5.0 if action[0] > 0.0 else -10.0


def noisy_discrete_actuator_force(action):
    """

    # TODO: Complete function docstring.
    # TODO: Check that the implementation conforms to my requirements.

    Args:
        action:

    Returns:

    """
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0

# end cart_pole.py #############################################################

# Keep track of best genome and run statistics
best = None
stats = None


def parse_args(args):
    """Parse command line parameters.

    Args:
      args ([str]): command line parameters as list of strings.

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Experiment configuration file")
    parser.add_argument('results_dir', type=str, help="Directory to save results")

    return parser.parse_args(args)


def save_object(genome, filename):
    """Save pickled genome object.

    Args:
        genome (Genome): The genome object to save.
        filename (str): The name of the file to save the pickled genome to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(genome, file)


def signal_handler(sig, frame):
    """Gracefully exit after receiving interrupt signal.

    TODO: Complete function docstring.

    Args:
        sig:
        frame:
    """
    global best
    global stats

    print('Early termination by user.')

    if best:
        print('Saving best genome...')
        save_object(best, 'solution.pickle')
        save_object(stats, 'stats_reporter.pickle')

    ray.shutdown()
    sys.exit()


def calc_fitness(t, max_t, x, dx, theta, dtheta):
    """Calculate genome fitness. This fitness function is designed to minimise
    pole oscillations.

    Args:
        t (int): The number of time steps the agent balanced the pole for.
        max_t (int): The maximum number of time steps (solution threshold).
        x (iterable): The cart position in meters.
        dx (iterable): The cart velocity in meters per second.
        theta (iterable): The pole angle from vertical in degrees.
        dtheta (iterable): The pole velocity in degrees per second.

    Returns:
        float: The fitness of the genome.
    """
    f1 = t / max_t

    if t < 100:
        f2 = 0
    else:
        f2 = 0.75 / sum([abs(x[i]) + abs(dx[i]) + abs(theta[i]) + abs(dtheta[i]) for i in range(100)])

    return 0.1 * f1 + 0.9 * f2


@ray.remote(num_cpus=1)
def evaluate_network(network, config, basic=False):
    """Evaluate the fitness of a network by running a pole balancing simulation.

    Args:
        network (RNN): The recurrent neural network to be evaluated.
        config (Config): The experiment configuration
        basic (bool): True if fitness is to be measured solely by balance time,
            False otherwise.

    Returns:
        float: The fitness of the network.
    """
    episode_fitnesses = []
    for i in range(config.num_episodes):  # Average over multiple episodes
        sim = CartPole(x=0.0, dx=0.0, theta=1.0, dtheta=0.0)
        network.reset()
        time_steps = 0

        X = deque()
        DX = deque()
        THETA = deque()
        DTHETA = deque()

        while time_steps < 60000:  # 60,000 time steps = 10 mins balance time
            # Store the system state for the previous 100 time steps
            if len(X) == 100:
                X.popleft()
                DX.popleft()
                THETA.popleft()
                DTHETA.popleft()

            x, dx, theta, dtheta = sim.get_state()
            X.append(x)
            DX.append(dx)
            THETA.append(theta)
            DTHETA.append(dtheta)

            # Get normalised inputs in the range [-1, 1]
            observation = sim.get_scaled_state()
            observation = [observation[0], observation[2]]  # Remove velocities

            output = network.forward(observation)
            action = continuous_actuator_force(output)

            sim.step(action)
            sim.step(0)  # Skip every 2nd time step

            # Stop if network fails to keep the cart within the position or
            # angle limits
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            time_steps += 1

        if basic:
            fitness = time_steps
        else:
            fitness = calc_fitness(time_steps, config.fitness_threshold,
                                   X, DX, THETA, DTHETA)
        episode_fitnesses.append(fitness)

    return sum(episode_fitnesses) / len(episode_fitnesses)


def evaluate_genomes(genomes, config, basic=False):
    """Setup the parallel evaluation of the given genomes.

    Modifies only the fitness member of each genome.

    Args:
        genomes (list):  A list of (genome_id, genome) pairs of the genomes to
            be evaluated.
        config (Config): The experiment configuration file.
        basic (bool): True if fitness is to be measured solely by balance time,
            False otherwise.
    """
    remaining_ids = []
    job_id_mapping = {}

    for _, genome in genomes:
        # Build network
        network = RNN.create(genome)

        # Create job for evaluating the fitness
        job_id = evaluate_network.remote(network, config, basic)
        remaining_ids.append(job_id)
        job_id_mapping[job_id] = genome

        while remaining_ids:
            # Use ray.wait to get the job ID of the first task that completes
            # There is only one returned result by default
            done_ids, remaining_ids = ray.wait(remaining_ids)
            result_id = done_ids[0]
            fitness = ray.get(result_id)
            genome = job_id_mapping[result_id]
            genome.fitness = fitness


def run(config, base_dir):
    """Performs a single evolutionary run.

    Args:
        config (Config): The experiment configuration file.
        base_dir (str): The base directory to store the results for this run.
    """
    global best
    global stats

    # Configure algorithm
    population = Population(config)

    # Add reporters
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(generation_interval=config.checkpoint_interval,
                                              time_interval_seconds=None,
                                              filename_prefix=base_dir + 'checkpoints/' + 'neat-checkpoint-'))

    # Set generation limit
    max_generations = config.max_generations
    generation = 0

    while generation < max_generations:
        batch_size = 1
        best = population.run(fitness_function=evaluate_genomes, n=batch_size)

        visualize.plot_stats(stats, ylog=False, view=False, filename=base_dir + "fitness.svg")

        mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
        print("Average mean fitness over last 5 generations: {0}".format(mfs))

        mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
        print("Average min fitness over last 5 generations: {0}".format(mfs))

        # Check if a solution has been found
        evaluate_genomes([(0, best)], config, True)
        print(f'Current best genome lasted {best.fitness} time steps.')
        # if best.fitness >= config.fitness_threshold:
        #     # Solved
        #     break

        # Save current best
        save_object(best, base_dir + f'solution_{generation}.pickle')

        generation += batch_size

    # Save best genome and stats reporter
    if best:
        print('Saving best genome...')
        save_object(best, base_dir + 'solution.pickle')
        save_object(stats, base_dir + 'stats_reporter.pickle')

        # visualize.plot_stats(stats, ylog=True, view=True, filename=base_dir + "fitness.svg")
        # visualize.plot_species(stats, view=True, filename=base_dir + "speciation.svg")
        #
        # node_names = {0: 'x', 1: 'theta', 2: 'out'}
        # visualize.draw_net(config, best, True, node_names=node_names)
        #
        # visualize.draw_net(config, best, view=True, node_names=node_names,
        #                    filename=base_dir + "best-rnn.gv")
        # visualize.draw_net(config, best, view=True, node_names=node_names,
        #                    filename=base_dir + "best-rnn-enabled.gv", show_disabled=False)
        # visualize.draw_net(config, best, view=True, node_names=node_names,
        #                    filename=base_dir + "best-rnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)


def main():
    """The main entry point to the program. Performs multiple evolutionary runs.
    """
    args = parse_args(sys.argv[1:])
    signal.signal(signal.SIGINT, signal_handler)

    if not args.config:
        print('No experiment configuration file provided.')
    else:
        # Initialise Ray
        ray.init()
        print('Ray Configuration:')
        print(f'Available resources: {ray.available_resources()}')
        print()

        # Load the experiment configuration file
        config = CustomConfig(Genome,
                              Reproduction,
                              SpeciesSet,
                              neat.DefaultStagnation,
                              args.config)

        for i in range(config.num_runs):
            print(f'Starting run {i+1}/{config.num_runs}')

            results_dir = args.results_dir + f'/run_{i+1}/'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                os.makedirs(results_dir + 'checkpoints/')

            run(config, results_dir)

        ray.shutdown()


if __name__ == '__main__':
    main()
