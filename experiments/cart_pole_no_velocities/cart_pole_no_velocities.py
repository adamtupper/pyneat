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
"""
import pickle
import argparse

import gym
import ray
import neat

from custom_neat.nn.recurrent import RNN
from custom_neat.genome import Genome
from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet

import visualize

# Initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Experiment configuration file")
args = parser.parse_args()

# Initialise Ray
ray.init()
print('Ray Configuration:')
print(f'Available resources: {ray.available_resources()}')
print()


# TODO: Put fitness evaluation functions in a class to store progress over time


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
            # remove velocities and normalise inputs to [-1, 1]
            observation = [observation[0] / 2.4, observation[2] / 41.8]

            output = network.forward(observation)
            action = round(output[0])

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


def run():
    """The entry point for the program, runs the experiment.
    """
    if not args.config:
        print('No experiment configuration file provided.')
    else:
        # Load the experiment configuration file
        config = neat.Config(Genome,
                             Reproduction,
                             SpeciesSet,
                             neat.DefaultStagnation,
                             args.config)

        # Configure algorithm
        population = neat.Population(config)
        best = None

        # Add reporters
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None))

        while True:
            try:
                best = population.run(fitness_function=evaluate_genomes, n=5)

                visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

                mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
                print("Average mean fitness over last 5 generations: {0}".format(mfs))

                mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
                print("Average min fitness over last 5 generations: {0}".format(mfs))

                if best.fitness >= 475:
                    # Solved
                    break
            except KeyboardInterrupt:
                print('User interrupt.')
                break

        # Save best genome
        if best:
            with open('solution.pickle', 'wb') as file:
                pickle.dump(best, file)

    ray.shutdown()


if __name__ == '__main__':
    run()
