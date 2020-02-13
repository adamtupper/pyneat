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
        key (int): The species ID.
        created (int): The generation in which the species was created.
        last_improved (int): The last generation where the fitness of the
            species improved.
        members (dict): A dictionary of {genome ID: genome} pairs for each
            genome in the species.
        representative (Genome): The genome that is the representative of the
            species, against which new genomes will be compared to see if they
            belong in this species.
        fitness (float): The species fitness.
        adjusted_fitness (float): The sum of the adjusted fitnesses for each genome
            in the species.
        fitness_history (:list:`float`): All previous fitness values. One for
            each generation this species has survived for.
    """
    def __init__(self, key, generation):
        """Create a new Species object.

        Args:
            key (int): The species ID.
            generation (int): The current generation.
        """
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.members = {}
        self.representative = None
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def __eq__(self, other):
        """Check to see if another species is equal to this one.

        Args:
            other (Species): The other species to compare.

        Returns:
            bool: True if the two species are equal, False otherwise.
        """
        self_attr = (self.key, self.created, self.last_improved, self.members,
                     self.representative, self.fitness, self.adjusted_fitness,
                     self.fitness_history)
        other_attr = (other.key, other.created, other.last_improved, other.members,
                      other.representative, other.fitness, other.adjusted_fitness,
                      other.fitness_history)
        return self_attr == other_attr

    def update(self, representative, members):
        """Replace the current individuals with a new set of individuals.

        Args:
            representative (Genome): The genome that is the new representative
            for this species.
            members (dict): A dictionary of genome ID and genome pairs of the
                new members of the species.
        """
        self.representative = copy.deepcopy(representative)
        self.members = members

    def get_fitnesses(self):
        """Get the fitnesses of each genome that belongs to this species.

        Returns:
            :list:`float`: The fitness of each genome that belongs to this
                species.
        """
        return [m.fitness for m in self.members.values()]


class SpeciesSet(DefaultClassConfig):
    """Encapsulates the speciation scheme.

    Attributes:
        species_set_config (DefaultClassConfig): The speciation configuration.
        reporters (ReporterSet): The set of reporters that log events.
        indexer (generator): Keeps track of the next species ID.
        species (dict): A dictionary of species ID, species pairs.
        genome_to_species (dict): A dictionary of genome ID, species ID pairs.
    """

    def __init__(self, config, reporters):
        """Creates a new SpeciesSet object.

        Args:
            config (DefaultClassConfig): The speciation configuration.
            reporters (ReporterSet): The set of reporters that log events.
        """
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(0)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        """Parse the speciation parameter values

        Args:
            param_dict (dict): A dictionary of parameter values.

        Returns:
            DefaultClassConfig: The speciation configuration.
        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config, population, generation):
        """Speciate the population.

        Args:
            config (Config): The global NEAT configuration.
            population (dict): A dictionary of genome ID, genome pairs.
            generation (int): The current generation.
        """
        unspeciated = set(population)
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}  # species ID: genome
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

            if best_representative[2] <= config.species_set_config.compatibility_threshold:
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

            if best_species[0] is not None and best_species[1] <= config.species_set_config.compatibility_threshold:
                # Genome fits an existing species
                new_members[best_species[0]].append(genome_id)
            else:
                # Genome belongs to a new species
                species_id = next(self.indexer)
                new_representatives[species_id] = genome
                new_members[species_id] = [genome_id]

        # Update set of species with new representatives and members
        self.genome_to_species = {}
        for species_id, representative in new_representatives.items():
            species = self.species.get(species_id)
            if species is None:
                # Species is new
                species = Species(species_id, generation)
                self.species[species_id] = species

            members = new_members[species_id]
            for genome_id in members:
                self.genome_to_species[genome_id] = species_id

            member_dict = {id: population[id] for id in new_members[species_id]}
            species.update(representative, member_dict)

        # Remove species without any members
        self.species = {k: s for k, s in self.species.items() if s.members}

        if self.species:
            # If there are species remaining
            gdmean = statistics.mean(distances.distances.values())
            gdstdev = statistics.stdev(distances.distances.values())
            self.reporters.info(
                'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev)
            )

    def get_species_id(self, individual_id):
        """Get the species ID of the species the given individual belongs to.

        Args:
            individual_id (int): The genome ID of the individual to check.

        Returns:
            int: The species ID the individual belongs to.
        """
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        """Get the species to given individual belongs to.

        Args:
            individual_id (int): The genome ID of the individual to check.

        Returns:
            Species: The species the individual belongs to.
        """
        species_id = self.genome_to_species[individual_id]
        return self.species[species_id]
