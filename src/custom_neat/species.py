"""Divides a population into species based on genetic distance.
"""
import copy
from itertools import count
import math
import statistics

from neat.config import ConfigParameter, DefaultClassConfig, Config
from neat.species import GenomeDistanceCache


class Species:
    """Encapsulates all information about a particular species.

    Attributes:
        members (dict): A dictionary of {genome ID: genome} pairs for each
            genome in the species.
        adj_fitness (float): The sum of the adjusted fitnesses for each genome
            in the species.
    """
    def __init__(self, key, generation):
        """Create a new species.

        Args:next
            key:
            generation:
        """
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.members = {}
        self.representative = None
        self.fitness = None
        self.adj_fitness = None
        self.fitness_history = []

    def __eq__(self, other):
        self_attr = (self.key, self.created, self.last_improved, self.members,
                     self.representative, self.fitness, self.adj_fitness,
                     self.fitness_history)
        other_attr = (other.key, other.created, other.last_improved, other.members,
                      other.representative, other.fitness, other.adj_fitness,
                      other.fitness_history)
        return self_attr == other_attr

    def update(self, representative, members):
        """Replace the current individuals with a new set of individuals.

        Args:
            representative (Genome): The genome that is the new representative
            for this species.
            members (dict): A dictionary of genome ID and genome pairs of the
                new members of the species.

        Returns:

        """
        self.representative = copy.deepcopy(representative)
        self.members = members

    def get_fitnesses(self):
        """

        Returns:

        """
        return [m.fitness for m in self.members.values()]


class SpeciesSet:
    """Encapsulates the speciation scheme.
    """
    # TODO: The DefaultSpeciesSet class inherits from DefaultClassConfig, this may not be necessary
    def __init__(self, config, reporters):
        """

        Args:
            config (SpeciesSetConfig):
            reporters:
        """
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(0)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        """

        Args:
            param_dict:

        Returns:

        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config, population, generation):
        """

        Args:
            config (Config):
            population:
            generation:

        Returns:

        """
        unspeciated = set(population)
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}  # species ID: genome ID
        new_members = {}  # species ID: [genome IDs]

        # Find the best new representatives for each species (closest to the
        # current representatives)
        for species_id, species in self.species.items():
            best_representative = (None, None, math.inf)
            for genome_id in unspeciated:
                genome = population[genome_id]
                distance = distances(species.representative, genome)
                if distance < best_representative[2]:
                    best_representative = (genome_id, genome, distance)

            if best_representative[2] < config.species_set_config.compatibility_threshold:
                # Species has not become extinct
                new_representatives[species_id] = best_representative[1]
                new_members[species_id] = [best_representative[0]]
                unspeciated.remove(best_representative[0])

        # Partition the remaining population into species
        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            best_species = (None, math.inf)
            for species_id, representative in new_representatives.items():
                distance = distances(representative, genome)
                if distance < best_species[1]:
                    best_species = (species_id, distance)

            if best_species[0] is not None and best_species[1] < config.species_set_config.compatibility_threshold:
                # Genome fits an existing species
                new_members[best_species[0]].append(genome_id)
            else:
                # Genome belongs to a new species
                species_id = next(self.indexer)
                new_representatives[species_id] = genome
                new_members[species_id] = [genome_id]

        # Update set of species with new representatives and members
        for species_id, representative_id in new_representatives.items():
            species = self.species.get(species_id)
            if species is None:
                # Species is new
                species = Species(species_id, generation)
                self.species[species_id] = species

            species.update(representative_id, {id: population[id] for id in new_members[species_id]})

        if self.species:
            # If there are species remaining
            gdmean = statistics.mean(distances.distances.values())
            gdstdev = statistics.stdev(distances.distances.values())
            self.reporters.info(
                'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev)
            )

    def get_species_id(self, individual_id):
        """

        Args:
            individual_id:

        Returns:

        """
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        """

        Args:
            individual_id:

        Returns:

        """
        species_id = self.genome_to_species[individual_id]
        return self.species[species_id]
