"""Implement the NEAT reproduction scheme.

The reproduction scheme is specifies the behaviour for creating, mutating and in
any way altering the population of genomes during evolution.
"""
from itertools import count
import math
import statistics
import random

from neat.config import ConfigParameter, write_pretty_params, Config

from custom_neat.genome import Genome


class ReproductionConfig:
    """Sets up and hold configuration information for the Reproduction class.
    """

    def __init__(self, params):
        """Creates a new ReproductionConfig object.

        Args:
            params (dict): A dictionary of config parameters and values.
        """
        self._params = [ConfigParameter('crossover_prob', float),
                        ConfigParameter('inter_species_crossover_prob', float),
                        ConfigParameter('elitism', int),
                        ConfigParameter('survival_threshold', float),
                        ConfigParameter('min_species_size', int)]

        # Use the configuration data to interpret the supplied parameters
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

    def save(self, f):
        """Save the reproduction configuration.

        Args:
            f (str): The filename to write the configuration to.
        """
        write_pretty_params(f, self, self._params)


class Reproduction:
    """Implements the NEAT reproduction scheme.

    TODO: Decide which attributes should be private.

    Attributes:
        reproduction_config (ReproductionConfig): The configuration for
            reproduction hyperparameters.
        reporters (ReporterSet): The set of reporters to log events via.
        genome_indexer (generator): Keeps track of the next genome ID when
            generating offspring.
        stagnation (DefaultStagnation): Keeps track of which species have
            stagnated.
        ancestors (dict): A dictionary that stores the parents of each
            offspring produced.
    """

    @classmethod
    def parse_config(cls, param_dict):
        """Takes a dictionary of configuration items, returns an object that
        will later be passed to the write_config method.

        Note: This is a required interface method.

        Args:
            param_dict (dict): A dictionary of configuration parameter values.

        Returns:
            ReproductionConfig: The reproduction configuration.
        """
        return ReproductionConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        """Takes a file-like object and the configuration object created by
        parse_config. This method should write the configuration item
        definitions to the given file.

        Note: This is a required interface method.

        Args:
            f (str): The filename of the file to write the configuration to.
            config (ReproductionConfig): The reproduction config to save.
        """
        config.save(f)

    def __init__(self, config, reporters, stagnation):
        """Create a new Reproduction object.

        Note: This is a required interface method.

        Args:
            config (ReproductionConfig): The configuration for
                reproduction hyperparameters.
            reporters (ReporterSet): The set of reporters to log events via.
            stagnation (DefaultStagnation): Keeps track of which species have
                stagnated.
        """
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(0)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        """Create a brand new population.

        Note: This is a required interface method.

        Args:
            genome_type (Genome): The type of the genome to create individuals
                using.
            genome_config (GenomeConfig): The genome configuration.
            num_genomes (int): The number of genomes to create (population
                size).

        Returns:
            dict: A dictionary of genome ID, genome pairs that make up the new
                population.
        """
        genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            genome = genome_type(key)
            genome.configure_new(genome_config)
            genomes[key] = genome
            self.ancestors[key] = tuple()

        return genomes

    def reproduce(self, config, species, pop_size, generation):
        """Produces the next generation of genomes.

        Note: This is a required interface method.

        Args:
            config (Config): The experiment configuration.
            species (SpeciesSet): The current allocation of genomes to species.
            pop_size (int): The desired size of the population.
            generation (int): The number of the next generation.

        Returns:
            dict: A dictionary of genome ID, genome pairs that make up the new
                population.
        """
        species_set = species
        num_elites = self.reproduction_config.elitism
        elitism_threshold = self.reproduction_config.min_species_size
        survival_threshold = self.reproduction_config.survival_threshold

        # Ensure that the number of elites cannot exceed the minimum species
        # size for elitism.
        assert num_elites <= elitism_threshold

        # Filter stagnant species
        all_fitnesses = []
        remaining_species = {}
        for species_key, species, stagnant in self.stagnation.update(species_set, generation):
            if stagnant:
                self.reporters.species_stagnant(species_key, species)
            else:
                all_fitnesses.extend(m.fitness for m in species.members.values())
                remaining_species[species_key] = species

        # Check for extinction
        if not remaining_species:
            species_set.species = {}
            return {}

        # Compute number of offspring per remaining species
        offspring_numbers = self.compute_num_offspring(remaining_species, pop_size)

        # Report average fitness metrics
        mean_fitness = statistics.mean(all_fitnesses)
        self.reporters.info("Mean fitness: {:.3f}".format(mean_fitness))

        # TODO: Remove redundancy in parent pool and population generation
        # Generate parent pool for each species
        parent_pool = {}
        for species_id, species in remaining_species.items():
            old_members = list(species.members.items())

            # Sort members in order of descending fitness
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Eliminate the lowest performing members of the species
            cutoff = int(math.ceil(survival_threshold) * len(old_members))

            # Ensure that there are always at least two parents remaining
            cutoff = max(cutoff, 2)
            old_members = old_members[:cutoff]

            parent_pool[species_id] = [m for m in old_members]

        # Generate new population
        new_population = {}
        species_set.species = {}
        for species_id, species in remaining_species.items():
            num_offspring = offspring_numbers[species_id]
            old_members = list(species.members.items())
            species.members = {}
            species_set.species[species.key] = species

            # Sort members in order of descending fitness
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation if species is large enough
            num_members = len(old_members)  # len(species.members.keys())
            if num_members > elitism_threshold and num_offspring > 0:
                for genome_id, genome in old_members[:num_elites]:
                    new_population[genome_id] = genome
                    num_offspring -= 1

                    if num_offspring == 0:
                        break

            if num_offspring <= 0:
                # No more offspring required for this species
                continue

            # Eliminate the lowest performing members of the species
            cutoff = int(math.ceil(survival_threshold) * len(old_members))

            # Ensure that there are always at least two parents remaining
            cutoff = max(cutoff, 2)
            old_members = old_members[:cutoff]

            while num_offspring > 0:
                num_offspring -= 1

                parent1_id, parent1 = random.choice(old_members)
                child_id = next(self.genome_indexer)

                if random.random() < self.reproduction_config.crossover_prob:
                    # Offspring is generated through mutation alone
                    child = parent1.copy()
                    child.key = child_id
                    child.mutate(config.genome_config)
                    self.ancestors[child_id] = (parent1,)
                else:
                    # Offspring is generated through mutation and crossover
                    # TODO: Test check for a sufficient number of species
                    if random.random() < self.reproduction_config.inter_species_crossover_prob:
                        # Inter-species crossover
                        candidates = [i for i in parent_pool.keys() if i != species_id]
                        if len(candidates) > 1:
                            other_species_id = random.choice(candidates)
                            parent2_id, parent2 = random.choice(parent_pool[other_species_id])
                        else:
                            # Fallback to intra-species crossover
                            parent2_id, parent2 = random.choice(old_members)
                    else:
                        # Intra-species crossover
                        parent2_id, parent2 = random.choice(old_members)

                    child = Genome(child_id)
                    child.configure_crossover(parent1, parent2, config.genome_config)
                    child.mutate(config.genome_config)
                    self.ancestors[child_id] = (parent1, parent2)

                new_population[child_id] = child

        return new_population

    @staticmethod
    def compute_num_offspring(remaining_species, popn_size):
        """Compute the number of offspring per species (proportional to fitness).

        Note: The largest remainder method is used to ensure the population size
        is maintained (https://en.wikipedia.org/wiki/Largest_remainder_method).

        TODO: Investigate a more efficient implementation of offspring allocation

        Args:
            remaining_species (dict): A dictionary ({species ID: species}) of
                the remaining species after filtering for stagnation.
            popn_size (int): The specified size of the population.

        Returns:
            dict: A dictionary of the number of offspring allowed for each
                species of the form {species ID: number of offspring}.
        """
        # Find genome of lowest fitness
        lowest_fitness = math.inf
        for species_id, species in remaining_species.items():
            for genome_id, genome in species.members.items():
                if genome.fitness < lowest_fitness:
                    lowest_fitness = genome.fitness

        # Calculate the sum of adjusted fitnesses for each species
        species_size = len(remaining_species.keys())
        for species_id, species in remaining_species.items():
            species.adj_fitness = 0.0  # reset sum of the adjusted fitnesses
            for genome_id, genome in species.members.items():
                species.adj_fitness += (genome.fitness - lowest_fitness) / species_size

        # Calculate the number of offspring for each species
        offspring = {}
        adj_fitness_sum = sum([s.adj_fitness for s in remaining_species.values()])
        for species_id, species in remaining_species.items():
            if adj_fitness_sum != 0:
                offspring[species_id] = popn_size * (species.adj_fitness / adj_fitness_sum)
            else:
                # All members of all species have zero fitness
                # Allocate each species an equal number of offspring
                offspring[species_id] = popn_size / species_size

        # Ensure that the species sizes sum to population size
        # Sort offspring numbers by fractional remainder
        sorted_ids = sorted(offspring.keys(), key=lambda k: offspring[k] - math.floor(offspring[k]))
        offspring = {id: math.floor(n) for id, n in offspring.items()}

        # Assign extra offspring to species based on fractional remainder
        idx = 0
        while sum(offspring.values()) < popn_size:
            offspring[sorted_ids[idx]] = offspring[sorted_ids[idx]] + 1
            idx += 1

        assert sum(offspring.values()) == popn_size

        return offspring
