"""Test NEAT implementation using the XOR problem.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python experiment.py path/to/config.ini
"""
import os
import sys
import pickle
import argparse
import time

import neat

from custom_neat.nn.feed_forward import NN
from custom_neat.genome import Genome
from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet
from custom_neat.config import CustomConfig
from custom_neat.population import Population


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


def evaluate_genomes(genomes, config):
    """Evaluate the population of genomes.

    Modifies only the fitness member of each genome.

    Args:
        genomes (list):  A list of (genome_id, genome) pairs of the genomes to
            be evaluated.
    """
    # 2-input XOR inputs and expected outputs.
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    fitness = 4.0
    for _, genome in genomes:
        # Build network
        network = NN.create(genome)

        # Evaluate network
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = network.forward(xi)
            fitness -= (output[0] - xo[0]) ** 2

        # Update genome's fitness
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
    solution = population.run(fitness_function=evaluate_genomes, n=100)
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


if __name__ == '__main__':
    main()
