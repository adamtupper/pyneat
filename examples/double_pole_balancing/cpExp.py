# ******************************** #
# Double pole balancing experiment #
# ******************************** #
import math
import random
import pickle
import os

from cart_pole import CartPole

import neat

from custom_neat.population import Population
from custom_neat.genome import Genome
from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet
from custom_neat.config import CustomConfig
import visualize


def save_object(genome, filename):
    """Save pickled genome object.

    Args:
        genome (Genome): The genome object to save.
        filename (str): The name of the file to save the pickled genome to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(genome, file)


def evaluate_population(population, config):
    _, genomes = zip(*population)
    simulation = CartPole(genomes, markov=False)
    simulation.print_status = False   # Comment this line to print the status
    simulation.run()


if __name__ == "__main__":

    if not os.path.exists('results/checkpoints/'):
        os.makedirs('results/checkpoints/')
    if not os.path.exists('results/genomes/'):
        os.makedirs('results/genomes/')

    config = CustomConfig(Genome, Reproduction, SpeciesSet, neat.DefaultStagnation, 'dpnv_config.ini')

    # change the number of inputs accordingly to the type
    # of experiment: markov (6) or non-markov (3)
    # you can also set the configs in dpole_config as long
    # as you have two config files for each type of experiment
    config.genome_config.num_inputs = 3

    # neuron model type
    # chromosome.node_gene_type = genome.NodeGene
    #chromosome.node_gene_type = genome.CTNodeGene

    # population.Population.evaluate = evaluate_population
    # pop = population.Population()
    # pop.epoch(500, report=1, save_best=0)

    population = Population(config)

    # Add reporters
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(generation_interval=config.checkpoint_interval,
                                              time_interval_seconds=None,
                                              filename_prefix='results/' + 'checkpoints/' + 'neat-checkpoint-'))

    # Set generation limit
    max_generations = config.max_generations
    generation = 0
    while generation < max_generations:
        # Draw genomes
        for key, genome in population.population.items():
            visualize.draw_net(genome, filename=f'results/genomes/genome_{key}')

        batch_size = 1
        best = population.run(fitness_function=evaluate_population, n=batch_size)

        visualize.plot_stats(stats, ylog=False, view=False, filename='results/' + "fitness.svg")

        mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
        print("Average mean fitness over last 5 generations: {0}".format(mfs))

        mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
        print("Average min fitness over last 5 generations: {0}".format(mfs))

        # Save current best
        # save_object(best, base_dir + f'solution_{generation}.pickle')

        generation += batch_size

    # Save best genome and stats reporter
    if best:
        print('Saving best genome...')
        save_object(best, 'results/' + 'solution.pickle')
        save_object(stats, 'results/' + 'stats_reporter.pickle')

    # visualize the best topology
    #visualize.draw_net(winner) # best chromosome
    # Plots the evolution of the best/average fitness
    #visualize.plot_stats(pop.stats)
    # Visualizes speciation
    #visualize.plot_species(pop.species_log)

    print(f'Number of evaluations: {best.key}')
    print(f'Winner score: {best.score}')
    # from time import strftime
    # date = strftime("%Y_%m_%d_%Hh%Mm%Ss")
    # saves the winner
    # file = open('winner_'+date, 'w')
    # pickle.dump(winner, file)
    # file.close()
    pickle.dump(population, open('results/population.pickle', 'wb'))
    #print winner