"""Tests for the reproduction module.
"""
import os

import pytest
import neat
from neat.reporting import ReporterSet

from pyneat.stagnation import Stagnation
from pyneat.reproduction import *
from pyneat.genome import *
from pyneat.species import *
from pyneat.innovation import InnovationStore


class TestReproduction:
    def setup_method(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'configurations', 'test_configuration.ini')
        self.config = neat.Config(Genome,
                                  Reproduction,
                                  SpeciesSet,
                                  Stagnation,
                                  config_path)
        self.reporters = ReporterSet()

    def test_create_new_population(self):
        """Test creating a new population of genomes.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 2
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 1.0

        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config, self.reporters, stagnation_scheme)
        pop_size = 10
        genomes = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())

        assert 10 == len(genomes)
        assert 10 == len({id for id in genomes.keys()})
        assert all([len(g.nodes) == 4 for id, g in genomes.items()])
        assert all([len(g.connections) == 3 for id, g in genomes.items()])

    def test_compute_num_offspring(self):
        """Test method for computing the number of offspring each species is
        allocated for the next generation.
        """
        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config, self.reporters, stagnation_scheme)
        popn_size = 10

        # Build Species 1
        genome1 = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome1.fitness = 10
        genome2 = Genome(key=1, config=self.config.genome_config, innovation_store=InnovationStore())
        genome2.fitness = 20
        species1 = Species(key=1, generation=1)
        species1.update(representative=genome1, members={1: genome1, 2: genome2})

        # Build Species 2
        genome3 = Genome(key=2, config=self.config.genome_config, innovation_store=InnovationStore())
        genome3.fitness = 40
        genome4 = Genome(key=3, config=self.config.genome_config, innovation_store=InnovationStore())
        genome4.fitness = 30
        species2 = Species(key=2, generation=1)
        species2.update(representative=genome3, members={3: genome3, 4: genome4})

        # Test Offspring numbers
        remaining_species = {1: species1, 2: species2}
        offspring_numbers = reproduction_scheme.compute_num_offspring(remaining_species, popn_size)

        assert 10 == sum(offspring_numbers.values())
        assert offspring_numbers[1] == 2
        assert offspring_numbers[2] == 8

    def test_compute_num_offspring_negative(self):
        """Test method for computing the number of offspring each species is
        allocated for the next generation when genomes have negative fitnesses.
        """
        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config, self.reporters, stagnation_scheme)
        popn_size = 10

        # Build Species 1
        genome1 = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome1.fitness = -30
        genome2 = Genome(key=1, config=self.config.genome_config, innovation_store=InnovationStore())
        genome2.fitness = -40
        species1 = Species(key=1, generation=1)
        species1.update(representative=genome1, members={1: genome1, 2: genome2})

        # Build Species 2
        genome3 = Genome(key=2, config=self.config.genome_config, innovation_store=InnovationStore())
        genome3.fitness = -10
        genome4 = Genome(key=3, config=self.config.genome_config, innovation_store=InnovationStore())
        genome4.fitness = -20
        species2 = Species(key=2, generation=1)
        species2.update(representative=genome3, members={3: genome3, 4: genome4})

        # Test Offspring numbers
        remaining_species = {1: species1, 2: species2}
        offspring_numbers = reproduction_scheme.compute_num_offspring(remaining_species, popn_size)

        assert 10 == sum(offspring_numbers.values())
        assert offspring_numbers[1] == 2
        assert offspring_numbers[2] == 8

    def test_reproduce_too_many_elites(self):
        """Test that reproduce works as expected when too many elites are specified.
        """
        # Setup configuration
        self.config.reproduction_config.num_elites = 5
        self.config.reproduction_config.min_species_size = 3
        self.config.reproduction_config.survival_threshold = 0.75

        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())
        for genome in population.values():
            genome.fitness = 1

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        with pytest.raises(AssertionError):
            reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=InnovationStore(), refocus=False)

    def test_reproduce_elitism(self):
        """Test that reproduce produces a population with the correct number of
        elites carried over.
        """
        self.config.reproduction_config.num_elites = 1
        self.config.reproduction_config.elitism_threshold = 1
        self.config.reproduction_config.survival_threshold = 0.8
        self.config.reproduction_config.mutate_only_prob = 1.0

        self.config.genome_config.weight_mutate_prob = 1.0
        self.config.genome_config.node_add_prob = 0.2
        self.config.genome_config.conn_add_prob = 0.5

        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        innovation_store = InnovationStore()
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, innovation_store)
        for genome in population.values():
            genome.fitness = 1

        for genome_id, genome in list(population.items())[:5]:
            # Increase weights for half the genomes to ensure two species
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        assert 2 == len(species_set.species)

        new_population = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=innovation_store, refocus=False)

        assert len(population) == len(new_population)

        num_duplicates = 0
        for old in population.values():
            for new in new_population.values():
                old_attrs = (old.nodes, old.connections, old.inputs, old.outputs, old.biases)
                new_attrs = (new.nodes, new.connections, new.inputs, new.outputs, new.biases)
                if old_attrs == new_attrs:
                    num_duplicates += 1

        assert num_duplicates == 2

    def test_reproduce_no_elitism(self):
        """Test that reproduce produces a population with all new genomes if the
        number of elites specified is zero.
        """
        self.config.reproduction_config.num_elites = 0
        self.config.reproduction_config.elitism_threshold = 4
        self.config.reproduction_config.survival_threshold = 0.8

        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        innovation_store = InnovationStore()
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, innovation_store)
        for genome in population.values():
            genome.fitness = 1

        for genome_id, genome in list(population.items())[:5]:
            # Increase weights for half the genomes to ensure two species
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        assert 2 == len(species_set.species)

        new_population = reproduction_scheme.reproduce(self.config, species_set, pop_size,  generation=1, innovation_store=innovation_store, refocus=False)

        assert len(population) == len(new_population)
        old_genomes = population.values()
        new_genomes = new_population.values()

        for old_genome in old_genomes:
            for new_genome in new_genomes:
                assert old_genome != new_genome

    def test_reproduce_no_mutation_or_crossover(self):
        """Test that the two populations are equal if no mutations or crossover are
        allowed.
        """
        self.config.reproduction_config.num_elites = 0
        self.config.reproduction_config.elitism_threshold = 4
        self.config.reproduction_config.survival_threshold = 1.0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0
        self.config.reproduction_config.crossover_prob = 0.0

        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())
        for genome in population.values():
            genome.fitness = 1

        for genome_id, genome in list(population.items())[:5]:
            # Increase weights for half the genomes to ensure two species
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        assert 2 == len(species_set.species)

        new_population = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=InnovationStore(), refocus=False)

        old_genomes = population.values()
        new_genomes = new_population.values()

        for old_genome in old_genomes:
            match = False
            for new_genome in new_genomes:
                if old_genome == new_genome:
                    match = True

            if not match:
                return False

    def test_reproduce_filter_stagnant(self):
        """Test that a stagnant population is filtered out.
        """
        self.config.reproduction_config.num_elites = 0
        self.config.reproduction_config.elitism_threshold = 4
        self.config.reproduction_config.survival_threshold = 1.0
        self.config.reproduction_config.crossover_prob = 0.0
        self.config.stagnation_config.species_fitness_func = 'max'
        self.config.stagnation_config.max_stagnation = 10
        self.config.stagnation_config.species_elitism = 0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0
        self.config.reproduction_config.crossover_prob = 0.0

        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        population0 = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())
        for genome in population0.values():
            genome.fitness = 1

        for genome_id, genome in list(population0.items())[:5]:
            # Increase weights for half the genomes to ensure two species
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population0, generation=0)

        assert 2 == len(species_set.species)

        population1 = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=InnovationStore(), refocus=False)
        species_set.speciate(self.config, population1, generation=10)
        for genome in population1.values():
            genome.fitness = 1

        assert 2 == len(species_set.species)

        # Skip to generation 11 to stagnate all species
        population2 = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=11, innovation_store=InnovationStore(), refocus=False)
        species_set.speciate(self.config, population2, generation=11)
        for genome in population2.values():
            genome.fitness = 1

        assert 0 == len(species_set.species)

    def test_generate_parent_pools(self):
        """Test that the worst performing members of each species are being
        culled from the parent pool.
        """
        self.config.reproduction_config.survival_threshold = 0.5
        self.config.genome_config.compatibility_disjoint_coefficient = 0.0
        self.config.genome_config.compatibility_weight_coefficient = 0.0
        self.config.species_set_config.compatibility_threshold = 1.0

        stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10

        # Create new population
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())

        # Assign different fitnesses to each member of the population
        for i, genome in enumerate(population.values()):
            genome.fitness = i

        # Speciate the population
        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        assert len(species_set.species) == 1

        parent_pools = reproduction_scheme.generate_parent_pools(species_set.species)

        # Check that only half of the genomes in the species survived
        assert len(parent_pools) == 1
        assert len(parent_pools[0]) == len(population) / 2

    # def test_reproduce_bug(self):
    #     """
    #     """
    #     self.config.reproduction_config.num_elites = 1
    #     self.config.reproduction_config.elitism_threshold = 4
    #     self.config.reproduction_config.survival_threshold = 1.0
    #     self.config.genome_config.weight_mutate_prob = 0.0
    #     self.config.genome_config.conn_add_prob = 0.0
    #     self.config.genome_config.node_add_prob = 0.0
    #     self.config.reproduction_config.crossover_prob = 0.0
    #     self.config.pop_size = 128
    #
    #     stagnation_scheme = Stagnation(self.config.stagnation_config, self.reporters)
    #     reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
    #
    #     population = pickle.load(open('failed_population_population.pickle', 'rb'))
    #     species_set = pickle.load(open('failed_population_species.pickle', 'rb'))
    #
    #     for g in population.values():
    #         if g.fitness is None:
    #             g.fitness = 0
    #
    #     for s in species_set.species.values():
    #         for m in s.members.values():
    #             if m.fitness is None:
    #                 m.fitness = 0
    #
    #     innov_store = InnovationStore()
    #     new_population = reproduction_scheme.reproduce(self.config, species_set,
    #                                                    self.config.pop_size, generation=1,
    #                                                    innovation_store=innov_store,
    #                                                    refocus=False)
    #
    #     assert len(new_population) == self.config.pop_size
