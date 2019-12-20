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
import pickle
import argparse
import signal
import sys
from collections import deque

import ray
import neat

from custom_neat.nn.recurrent import RNN
from custom_neat.genome import Genome
from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet
from custom_neat.config import CustomConfig

import visualize
import cart_pole

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Experiment configuration file")
args = parser.parse_args()

# Initialise Ray
ray.init()
print('Ray Configuration:')
print(f'Available resources: {ray.available_resources()}')
print()

# Keep track of best genome
best = None


def save_genome(genome, filename):
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

    print('Early termination by user.')

    if best:
        print('Saving best genome...')
        save_genome(best, 'solution.pickle')

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
        sim = cart_pole.CartPole(x=0.0, dx=0.0, theta=1.0, dtheta=0.0)
        network.reset()
        time_steps = 0

        X = deque()
        DX = deque()
        THETA = deque()
        DTHETA = deque()

        while time_steps < config.fitness_threshold:  # Considered solved fitness exceeds some threshold
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
            action = cart_pole.continuous_actuator_force(output)

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


def run():
    """The entry point for the program, runs the experiment.
    """
    global best

    if not args.config:
        print('No experiment configuration file provided.')
    else:
        # Load the experiment configuration file
        config = CustomConfig(Genome,
                              Reproduction,
                              SpeciesSet,
                              neat.DefaultStagnation,
                              args.config)

        # Configure algorithm
        population = neat.Population(config)

        # Add reporters
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(generation_interval=config.checkpoint_interval,
                                                  time_interval_seconds=None))

        # Set generation limit
        max_generations = config.max_generations
        generation = 0

        while generation < max_generations:
            batch_size = 1
            best = population.run(fitness_function=evaluate_genomes, n=batch_size)

            visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Check if a solution has been found
            evaluate_genomes([(0, best)], config, True)
            print(f'Current best genome lasted {best.fitness} time steps.')
            if best.fitness >= config.fitness_threshold:
                # Solved
                break

            # Save current best
            save_genome(best, f'solution_{generation}.pickle')

            generation += batch_size

        # Save best genome
        if best:
            print('Saving best genome...')
            save_genome(best, 'solution.pickle')

            visualize.plot_stats(stats, ylog=True, view=True, filename="fitness.svg")
            visualize.plot_species(stats, view=True, filename="speciation.svg")

            node_names = {0: 'x', 1: 'theta', 2: 'out'}
            visualize.draw_net(config, best, True, node_names=node_names)

            visualize.draw_net(config, best, view=True, node_names=node_names,
                               filename="best-rnn.gv")
            visualize.draw_net(config, best, view=True, node_names=node_names,
                               filename="best-rnn-enabled.gv", show_disabled=False)
            visualize.draw_net(config, best, view=True, node_names=node_names,
                               filename="best-rnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    ray.shutdown()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    run()
