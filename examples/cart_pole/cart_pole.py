"""Test NEAT implementation using the OpenAI Gym CartPole-v1 Environment.

The agent is provided with four inputs (position of cart, velocity of cart,
angle of pole and rotation rate of pole) and must learn one output (0 to move
the cart left, 1 to move the cart right). The problem is considered solved if
the agent can achieve an average reward of >= 475 over 100 consecutive episodes.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python cart_pole.py path/to/config.ini
"""
import os
import sys
import pickle
import argparse
import time

import gym
import ray
import neat

from pyneat.nn.recurrent import RNN
from pyneat.genome import Genome
from pyneat.reproduction import Reproduction
from pyneat.species import SpeciesSet
from pyneat.config import CustomConfig
from pyneat.population import Population

# # Initialise argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('config', type=str, help="Experiment configuration file")
# args = parser.parse_args()

# # Initialise Ray
# ray.init()
# print('Ray Configuration:')
# print(f'Available resources: {ray.available_resources()}')
# print()

# TODO: Put fitness evaluation functions in a class to store progress over time


def parse_args(args):
    """Parse command line parameters.

    Args:
      args ([str]): command line parameters as list of strings.

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Experiment configuration file")
    parser.add_argument('results_dir', type=str, help='Directory to save results')

    return parser.parse_args(args)


@ray.remote(num_cpus=1)
def compute_fitness(network):
    """Evaluate the fitness of a network in the gym environment.

    Args:
        network (RNN): The recurrent neural network to be evaluated.

    Returns:
        float: The fitness of the network.
    """
    # Create Gym environment
    env = gym.make('CartPole-v1')

    episode_fitnesses = []
    for i in range(100):  # average over 100 episodes
        observation = env.reset()
        network.reset()
        for t in range(500):  # considered solved if able to survive 500 time steps
            action = round(network.forward(observation)[0])
            observation, reward, done, info = env.step(action)

            if done:
                episode_fitnesses.append(t)
                break

    env.close()
    return sum(episode_fitnesses) / len(episode_fitnesses)


def evaluate_genomes(genomes, config):
    """Setup the parallel evaluation of the given genomes.

    Modifies only the fitness member of each genome.

    Args:
        genomes (list):  A list of (genome_id, genome) pairs of the genomes to
            be evaluated.
        config (Config): The experiment configuration file.
    """
    remaining_ids = []
    job_id_mapping = {}

    for _, genome in genomes:
        # Build network
        network = RNN.create(genome)

        # Create job for evaluating the fitness
        job_id = compute_fitness.remote(network)
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

    # Perform evolutionary run
    population = Population(config)
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(generation_interval=config.checkpoint_interval,
                                              time_interval_seconds=None,
                                              filename_prefix=base_dir + 'checkpoints/' + 'neat-checkpoint-'))

    start = time.time()
    solution = population.run(fitness_function=evaluate_genomes)
    end = time.time()

    elapsed = end - start
    print(f'Time elapsed: {time.strftime("%H:%M:%S", time.gmtime(elapsed))}')
    print(f'Fitness achieved: {solution.fitness}')
    print()

    # Save best genome
    with open(base_dir + 'solution.pickle', 'wb') as file:
        pickle.dump(solution, file)


def main():
    """The main entry point to the program. Performs multiple evolutionary runs.
    """
    args = parse_args(sys.argv[1:])

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

            results_dir = args.results_dir + f'run_{i+1}/'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                os.makedirs(results_dir + 'checkpoints/')

            run(config, results_dir)

        ray.shutdown()


if __name__ == '__main__':
    main()
