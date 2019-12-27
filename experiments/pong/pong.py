"""Evolve an agent to play Pong using the Atari Annotated RAM Interface
(AtariARI).

The agent is provided with two inputs (position of cart and the angle of pole).
It must learn one output (0 to move the cart left, 1 to move the cart right).
The problem is considered solved if the agent can achieve an average reward of
>= 475 over 100 consecutive episodes.

This problem is harder than the standard pole balancing problem as the agent is
required to learn the speeds, which requires recurrence.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python learn_pong.py path/to/config.ini

    To safely stop an evolutionary run early, interrupt the program as the
    population statistics are reported for a generation.
"""
import os
import pickle
import argparse
import signal
import sys
from collections import deque

import numpy as np
import ray
import neat
import gym
from atariari.benchmark.wrapper import AtariARIWrapper

from custom_neat.nn.recurrent import RNN
from custom_neat.genome import Genome
from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet
from custom_neat.config import CustomConfig

import visualize

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
    parser.add_argument('results_dir', type=str, help='Directory to save results')

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


@ray.remote(num_cpus=1)
def evaluate_network(network, config):
    """Evaluate the fitness of a network by playing through a number of games.

    Args:
        network (RNN): The recurrent neural network to be evaluated.
        config (Config): The experiment configuration

    Returns:
        float: The fitness of the network.
    """
    env = AtariARIWrapper(gym.make('Pong-v4'))

    output_to_action = {
        0: 0,  # NOOP
        1: 2,  # RIGHT (UP)
        2: 3  # LEFT (DOWN)
    }

    episode_fitnesses = []
    for i in range(config.num_episodes):  # Average over multiple episodes
        img = env.reset()
        # Avoid bug in Pong where agent can win if it doesn't move
        img, reward, done, state = env.step(2)  # Move agent up once
        network.reset()

        while not done:
            obs = [state['labels']['player_x'],
                   state['labels']['player_y'],
                   state['labels']['enemy_x'],
                   state['labels']['enemy_y'],
                   state['labels']['ball_x'],
                   state['labels']['ball_y']]

            # Normalise inputs in the range [0, 1]
            obs[0] = (obs[0] - 0) / (205 - 0)    # player x
            obs[1] = (obs[1] - 38) / (203 - 38)  # player y
            obs[2] = (obs[2] - 0) / (205 - 0)    # enemy x
            obs[3] = (obs[3] - 38) / (203 - 38)  # enemy y
            obs[4] = (obs[4] - 0) / (205 - 0)    # ball x
            obs[5] = (obs[5] - 44) / (207 - 44)  # ball y

            output = network.forward(obs)
            action = output_to_action[np.argmax(output)]

            img, reward, done, state = env.step(action)

        episode_fitnesses.append(state['labels']['player_score'])

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
        job_id = evaluate_network.remote(network, config)
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
    population = neat.Population(config)

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

        if best.fitness >= config.fitness_threshold:
            # Solved
            break

        # Save current best
        save_object(best, base_dir + f'solution_{generation}.pickle')

        generation += batch_size

    # Save best genome and stats reporter
    if best:
        print('Saving best genome...')
        save_object(best, base_dir + 'solution.pickle')
        save_object(stats, base_dir + 'stats_reporter.pickle')


def main():
    """The main entry point to the program. Performs multiple evolutionary runs.
    """
    args = parse_args(sys.argv[1:])
    signal.signal(signal.SIGINT, signal_handler)

    if not args.config:
        print('No experiment configuration file provided.')
    else:
        # Initialise Ray
        ray.init(address='auto')
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
