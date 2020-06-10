"""Implement the NEAT reproduction scheme.

The reproduction scheme is specifies the behaviour for creating, mutating and in
any way altering the population of genomes during evolution.
"""
from itertools import count
import math
import statistics
import random

from neat.config import ConfigParameter, write_pretty_params, Config

from pyneat.genome import Genome


class ReproductionConfig:
    """Sets up and hold configuration information for the Reproduction class.

    Config Parameters:
        mutate_only_prob (float): The probability that a child is generated
            through mutation alone. Crossover is only an option if there is more
            than one remaining parent in the parent pool for the species in
            question.
        crossover_avg_prob (float): The probability that the weights of mutual
            connections are averaged from both parents instead of chosen at
            random from one or the other.
        crossover_only_prob (float): The probability that a child
            generated via crossover is not also mutated.
        inter_species_crossover_prob (float): The probability (given crossover)
            that the child is instead generating using parents from different
            species. Relies on their being more than one species.
        num_elites (int): The number of elites from each species to be copied to
            the next generation. The size of a species must surpass the
            elitism_threshold for elitism to occur.
        elitism_threshold (int): Elitism will only be applied for a species if
            the number of remaining parents exceeds this threshold.
        survival_threshold (float): The proportion of members of each species
            that are added to the parent pool and are allowed to reproduce. The
            fittest members are kept.
    """

    def __init__(self, params):
        """Creates a new ReproductionConfig object.

        Args:
            params (dict): A dictionary of config parameters and values.
        """
        self._params = [ConfigParameter('mutate_only_prob', float),
                        ConfigParameter('crossover_avg_prob', float),
                        ConfigParameter('crossover_only_prob', float),
                        ConfigParameter('inter_species_crossover_prob', float),
                        ConfigParameter('num_elites', int),
                        ConfigParameter('survival_threshold', float),
                        ConfigParameter('elitism_threshold', int)]

        # Use the configuration data to interpret the supplied parameters
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

    def save(self, filename):
        """Save the reproduction configuration.

        Args:
            filename (str): The filename to write the configuration to.
        """
        write_pretty_params(filename, self, self._params)


class Reproduction:
    """Implements the NEAT reproduction scheme.

    TODO: Decide which attributes should be private.

    Attributes:
        reproduction_config (ReproductionConfig): The configuration for
            reproduction hyperparameters.
        reporters (ReporterSet): The set of reporters to log events via.
        genome_key_generator (generator): Keeps track of the next genome key when
            generating offspring.
        stagnation (Stagnation): Keeps track of which species have
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
    def write_config(cls, filename, config):
        """Takes a file-like object and the configuration object created by
        parse_config. This method should write the configuration item
        definitions to the given file.

        Note: This is a required interface method.

        Args:
            filename (str): The filename of the file to write the configuration to.
            config (ReproductionConfig): The reproduction config to save.
        """
        config.save(filename)

    def __init__(self, config, reporters, stagnation):
        """Create a new Reproduction object.

        Note: This is a required interface method.

        Args:
            config (ReproductionConfig): The configuration for
                reproduction hyperparameters.
            reporters (ReporterSet): The set of reporters to log events via.
            stagnation (Stagnation): Keeps track of which species have
                stagnated.
        """
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_key_generator = count(0)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes, innovation_store):
        """Create a brand new population.

        Note: This is a required interface method.

        Args:
            genome_type (Genome): The type of the genome to create individuals
                using.
            genome_config (GenomeConfig): The genome configuration.
            num_genomes (int): The number of genomes to create (population
                size).
            innovation_store (InnovationStore): The population-wide innovation
                store used for tracking new structural mutations.

        Returns:
            dict: A dictionary of genome key, genome pairs that make up the new
                population.
        """
        genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_key_generator)
            genome = genome_type(key, genome_config, innovation_store)
            genome.configure_new()
            genomes[key] = genome
            self.ancestors[key] = tuple()

        return genomes

    def generate_parent_pools(self, remaining_species):
        """Culls the lowest performing members of each remaining species

        Args:
            remaining_species (dict): Species key/species pairs for the
                remaining species after stagnated species have been removed.

        Returns:
            dict: The parent genomes for each species. A dictionary of the form
                species key, genomes.
        """
        survival_threshold = self.reproduction_config.survival_threshold
        parent_pool = {}

        for species_key, species in remaining_species.items():
            old_members = list(species.members.items())

            # Sort members in order of descending fitness
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Eliminate the lowest performing members of the species
            cutoff = int(math.ceil(survival_threshold * len(old_members)))
            parents = old_members[:cutoff]
            parent_pool[species_key] = parents

        return parent_pool

    def reproduce(self, config, species, pop_size, generation, innovation_store, refocus):
        """Produces the next generation of genomes.

        Note: This is a required interface method.

        The steps are broadly as follows:
            1. Filter stagnant species.
            2. Compute the number of offspring for each remaining species.
            3. Generate the parent pool for each remaining species (eliminate
                the lowest performing members).
            4. Generate the new population.

        Args:
            config (Config): The experiment configuration.
            species (SpeciesSet): The current allocation of genomes to species.
            pop_size (int): The desired size of the population.
            generation (int): The number of the next generation.
            innovation_store (InnovationStore): The population-wide innovation
                store used for tracking new structural mutations.

        Returns:
            dict: A dictionary of genome key, genome pairs that make up the new
                population.
        """
        species_set = species
        num_elites = self.reproduction_config.num_elites
        elitism_threshold = self.reproduction_config.elitism_threshold

        # Ensure that the number of elites cannot exceed the minimum species
        # size for elitism.
        assert num_elites <= elitism_threshold

        all_fitnesses = []
        remaining_species = {}
        if refocus:
            # Keep only the top two species
            sorted_species = [s for s in species_set.species.values() if s.fitness is not None]
            sorted_species.sort(key=lambda x: x.fitness, reverse=True)
            for species in sorted_species[:2]:
                all_fitnesses.extend(m.fitness for m in species.members.values())
                remaining_species[species.key] = species
        else:
            # Filter stagnant species
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

        # Report average fitness metrics (of remaining species)
        mean_fitness = statistics.mean(all_fitnesses)
        self.reporters.info("Mean fitness: {:.3f}".format(mean_fitness))

        # Generate parent pool for each species
        parent_pool = self.generate_parent_pools(remaining_species)

        # Generate new population
        new_population = {}
        species_set.species = {}
        for species_key, species in remaining_species.items():
            num_offspring = offspring_numbers[species_key]
            if num_offspring == 0:
                continue

            species.members = {}
            species_set.species[species_key] = species
            parents = parent_pool[species_key]

            # Add elite(s)
            if len(parents) > elitism_threshold:
                for i in range(num_elites):
                    """
                    TODO: Should elites be assigned new genome keys?
                    
                    This is the current implementation, but it might be better
                    for history tracking (when performing analysis) to keep the
                    same key. With regards to algorithm performance/correctness
                    this doesn't matter.
                    """
                    elite_key, elite = parents[i]
                    child_key = next(self.genome_key_generator)
                    child = elite.copy()
                    child.key = child_key

                    self.ancestors[child_key] = (elite,)
                    new_population[child_key] = child

                    num_offspring -= 1
                    if num_offspring == 0:
                        break

            # Produce offspring through mutation/crossover
            while num_offspring > 0:
                num_offspring -= 1
                child_key = next(self.genome_key_generator)

                if (len(parents) > 1) and (random.random() > self.reproduction_config.mutate_only_prob):
                    # Child is generated through mutation and crossover
                    (parent1_key, parent1), (parent2_key, parent2) = random.sample(parents, 2)

                    if random.random() < self.reproduction_config.inter_species_crossover_prob:
                        # Inter-species crossover (replace the 2nd parent with one from another species)
                        candidates = [i for i in parent_pool.keys() if i != species_key and len(parent_pool[i]) > 0]
                        if len(candidates) > 1:
                            other_species_key = random.choice(candidates)
                            parent2_key, parent2 = random.choice(parent_pool[other_species_key])

                    child = Genome(child_key, config.genome_config, innovation_store)
                    average = True if random.random() < self.reproduction_config.crossover_avg_prob else False
                    child.configure_crossover(parent1, parent2, average)
                    if random.random() > self.reproduction_config.crossover_only_prob:
                        child.mutate()
                    self.ancestors[child_key] = (parent1, parent2)
                else:
                    # Child is generated through mutation alone
                    parent_key, parent = random.choice(parents)
                    child = parent.copy()
                    child.key = child_key
                    child.mutate()
                    self.ancestors[child_key] = (parent,)

                new_population[child_key] = child

        return new_population

    @staticmethod
    def compute_num_offspring(remaining_species, pop_size):
        """Compute the number of offspring per species (proportional to fitness).

        Note: The largest remainder method is used to ensure the population size
        is maintained (https://en.wikipedia.org/wiki/Largest_remainder_method).

        TODO: Investigate a more efficient implementation of offspring allocation

        Args:
            remaining_species (dict): A dictionary ({species key: species}) of
                the remaining species after filtering for stagnation.
            pop_size (int): The specified size of the population.

        Returns:
            dict: A dictionary of the number of offspring allowed for each
                species of the form {species key: number of offspring}.
        """
        # Find genome of lowest fitness
        lowest_fitness = math.inf
        for species_key, species in remaining_species.items():
            for genome_key, genome in species.members.items():
                if genome.fitness < lowest_fitness:
                    lowest_fitness = genome.fitness

        # Calculate the sum of adjusted fitnesses for each species
        for species_key, species in remaining_species.items():
            species_size = len(species.members)
            species.adjusted_fitness = 0.0  # reset sum of the adjusted fitnesses
            for genome_key, genome in species.members.items():
                species.adjusted_fitness += (genome.fitness) / species_size

        # Calculate the number of offspring for each species
        offspring = {}
        adjusted_fitness_sum = sum([s.adjusted_fitness for s in remaining_species.values()])
        for species_key, species in remaining_species.items():
            if adjusted_fitness_sum != 0:
                offspring[species_key] = pop_size * (species.adjusted_fitness / adjusted_fitness_sum)
            else:
                # All members of all species have zero fitness
                # Allocate each species an equal number of offspring
                offspring[species_key] = pop_size / len(remaining_species)

        # Ensure that the species sizes sum to population size
        # Sort offspring numbers (in descending order) by fractional remainder
        sorted_keys = sorted(offspring.keys(), key=lambda k: offspring[k] - math.floor(offspring[k]), reverse=True)
        offspring = {key: math.floor(n) for key, n in offspring.items()}

        # Assign extra offspring to species based on fractional remainder
        idx = 0
        while sum(offspring.values()) < pop_size:
            offspring[sorted_keys[idx]] = offspring[sorted_keys[idx]] + 1
            idx += 1

        assert sum(offspring.values()) == pop_size

        return offspring
