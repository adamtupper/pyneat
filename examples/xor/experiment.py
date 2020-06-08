"""Test NEAT implementation using the XOR problem.

Example:
    Ensure that the following is executed in the `peal` conda environment.

        $ python experiment.py path/to/config.ini results
"""
import os
import sys
import pickle
import argparse
import time
import shutil

import neat
import numpy as np

from pyneat.nn.feed_forward import NN
from pyneat.graph_utils import required_for_output
from pyneat.genome import Genome, NodeType
from pyneat.reproduction import Reproduction
from pyneat.species import SpeciesSet
from pyneat.config import CustomConfig
from pyneat.population import Population
from pyneat.stagnation import Stagnation


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
        genomes (list): A list of (genome_id, genome) pairs of the genomes to
            be evaluated.
        config (CustomConfig): The experiment configuration.
    """
    # 2-input XOR inputs and expected outputs.
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    for _, genome in genomes:
        fitness = 4.0

        # Build network
        network = NN.create(genome)

        # Evaluate network
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = network.forward(xi)
            fitness -= (output[0] - xo[0]) ** 2

        # Update genome's fitness
        genome.fitness = fitness


def run(config, base_dir):
    """Performs a number of evolutionary runs.

    Args:
        config (CustomConfig): The experiment configuration.
        base_dir (str): The base directory to store the results for the
            experiment.
    """
    # Store statistics for each run
    generations = []
    evaluations = []
    n_hidden = []
    n_hidden_used = []
    n_connections = []
    n_connections_used = []
    succeeded = []
    durations = []
    fitnesses = []

    # Perform evolutionary runs
    for i in range(config.num_runs):
        run_dir = os.path.join(base_dir, f'run_{i}')
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(run_dir)
        os.makedirs(checkpoint_dir)

        population = Population(config)
        population.add_reporter(neat.StatisticsReporter())
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(generation_interval=config.checkpoint_interval,
                                                  time_interval_seconds=None,
                                                  filename_prefix=os.path.join(checkpoint_dir, 'checkpoint_')))

        start = time.time()
        solution = population.run(fitness_function=evaluate_genomes, n=100)
        end = time.time()
        durations.append(end - start)

        if solution.fitness > config.fitness_threshold:
            succeeded.append(True)
            generations.append(population.generation)
            evaluations.append(population.generation * config.pop_size)
            n_hidden.append(len([g for g in solution.nodes.values() if g.type == NodeType.HIDDEN]))
            n_connections.append(len([g for g in solution.connections.values() if g.expressed]))
            fitnesses.append(solution.fitness)

            enabled_connections = [(g.node_in, g.node_out) for g in solution.connections.values() if g.expressed]
            required_nodes = required_for_output(solution.inputs,
                                                 solution.biases,
                                                 solution.outputs,
                                                 enabled_connections,
                                                 list(solution.nodes.keys()))
            n_hidden_used.append(len([n for n in required_nodes if solution.nodes[n].type == NodeType.HIDDEN]))

            n_connections_used.append(len([g for g in solution.connections.values() if g.node_out in required_nodes and g.expressed]))

            # Save solution
            with open(os.path.join(run_dir, 'solution.pickle'), 'wb') as file:
                pickle.dump(solution, file)
        else:
            succeeded.append(False)

    print()
    print('Results:')
    print(f'\tAvg. # Generations:\t{np.mean(generations):.3f}')
    print(f'\tAvg. # Evaluations:\t{np.mean(evaluations):.3f}')
    print(f'\tStd. Dev. # Evaluations:\t{np.std(evaluations):.3f}')

    print(f'\tAvg. # Hidden Nodes:\t{np.mean(n_hidden):.3f}')
    print(f'\tStd. Dev. # Hidden Nodes:\t{np.std(n_hidden):.3f}')
    print(f'\tAvg. # Used Hidden Nodes:\t{np.mean(n_hidden_used):.3f}')
    print(f'\tStd. Dev. # Used Hidden Nodes:\t{np.std(n_hidden_used):.3f}')

    print(f'\tAvg. # Enabled Connections:\t{np.mean(n_connections):.3f}')
    print(f'\tStd. Dev. # Enabled Connections:\t{np.std(n_connections):.3f}')
    print(f'\tAvg. # Enabled & Used Connections:\t{np.mean(n_connections_used):.3f}')
    print(f'\tStd. Dev. # Enabled & Used Connections:\t{np.std(n_connections_used):.3f}')

    print(f'\tSuccess Rate:\t{len([x for x in succeeded if x]) / config.num_runs * 100:.3f}%')
    print(f'\tAvg. Time Elapsed:\t{time.strftime("%H:%M:%S", time.gmtime(np.mean(durations)))}')
    print(f'\tAvg. Solution Fitness:\t{np.mean(fitnesses):.3f}')
    print(f'\tWorst # Generations:\t{max(generations)}')
    print(f'\tWorst # Evaluations:\t{max(evaluations)}')
    print()


def main():
    """The main entry point to the program. Performs setup for the experiments.
    """
    args = parse_args(sys.argv[1:])

    if not args.config:
        print('No experiment configuration file provided.')
    else:
        # Load the experiment configuration file
        config = CustomConfig(Genome,
                              Reproduction,
                              SpeciesSet,
                              Stagnation,
                              args.config)

        if os.path.exists(args.results_dir):
            shutil.rmtree(args.results_dir)
        else:
            os.makedirs(args.results_dir)

        # Run the experiment
        run(config, args.results_dir)


if __name__ == '__main__':
    main()
