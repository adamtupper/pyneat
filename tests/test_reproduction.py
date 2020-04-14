"""Tests for the reproduction module.
"""
import configparser
import itertools
import os

import pytest
import neat
from neat.reporting import ReporterSet
from neat.stagnation import DefaultStagnation

from custom_neat.reproduction import *
from custom_neat.genome import *
from custom_neat.species import *
from custom_neat.innovation import InnovationStore


class TestReproduction:
    def setup_method(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'configurations', 'test_configuration.ini')
        self.config = neat.Config(Genome,
                                  Reproduction,
                                  SpeciesSet,
                                  neat.DefaultStagnation,
                                  config_path)
        self.reporters = ReporterSet()

    def test_create_new_population(self):
        """Test creating a new population of genomes.
        """
        stagnation_scheme = DefaultStagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config, self.reporters, stagnation_scheme)
        pop_size = 10
        genomes = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())

        assert 10 == len(genomes)
        assert 10 == len({id for id in genomes.keys()})
        assert all([len(g.nodes) == 3 for id, g in genomes.items()])

    def test_compute_num_offspring(self):
        """Test method for computing the number of offspring each species is
        allocated for the next generation.
        """
        stagnation_scheme = DefaultStagnation(self.config.stagnation_config, self.reporters)
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

    def test_reproduce_too_many_elites(self):
        """Test that reproduce works as expected when too many elites are specified.
        """
        # Setup configuration
        self.config.reproduction_config.elitism = 5
        self.config.reproduction_config.min_species_size = 3
        self.config.reproduction_config.survival_threshold = 0.75

        stagnation_scheme = DefaultStagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())
        for genome in population.values():
            genome.fitness = 1

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        with pytest.raises(AssertionError):
            reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=InnovationStore())

    def test_reproduce_elitism(self):
        """Test that reproduce produces a population with the correct number of
        elites carried over.
        """
        self.config.reproduction_config.elitism = 1
        self.config.reproduction_config.min_species_size = 4
        self.config.reproduction_config.survival_threshold = 0.8

        stagnation_scheme = DefaultStagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        innovation_store = InnovationStore()
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, innovation_store)
        for genome in population.values():
            genome.fitness = 1

        for genome_id, genome in list(population.items())[:5]:
            # Increase weight and biases for half the genomes to ensure two species
            for g in genome.nodes.values():
                g.bias += 50
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        assert 2 == len(species_set.species)

        new_population = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=innovation_store)

        assert len(population) == len(new_population)
        old_genomes = population.values()
        new_genomes = new_population.values()
        assert 2 == len([1 for (old, new) in itertools.product(old_genomes, new_genomes) if old.nodes == new.nodes and old.connections == new.connections])

    def test_reproduce_no_elitism(self):
        """Test that reproduce produces a population with all new genomes if the
        number of elites specified is zero.
        """
        self.config.reproduction_config.elitism = 0
        self.config.reproduction_config.min_species_size = 4
        self.config.reproduction_config.survival_threshold = 0.8

        stagnation_scheme = DefaultStagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        innovation_store = InnovationStore()
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, innovation_store)
        for genome in population.values():
            genome.fitness = 1

        for genome_id, genome in list(population.items())[:5]:
            # Increase weight and biases for half the genomes to ensure two species
            for g in genome.nodes.values():
                g.bias += 50
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        assert 2 == len(species_set.species)

        new_population = reproduction_scheme.reproduce(self.config, species_set, pop_size,  generation=1, innovation_store=innovation_store)

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
        self.config.reproduction_config.elitism = 0
        self.config.reproduction_config.min_species_size = 4
        self.config.reproduction_config.survival_threshold = 1.0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.bias_mutate_prob = 0.0
        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0
        self.config.reproduction_config.crossover_prob = 0.0

        stagnation_scheme = DefaultStagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        population = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())
        for genome in population.values():
            genome.fitness = 1

        for genome_id, genome in list(population.items())[:5]:
            # Increase weight and biases for half the genomes to ensure two species
            for g in genome.nodes.values():
                g.bias += 50
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population, generation=1)

        assert 2 == len(species_set.species)

        new_population = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=InnovationStore())

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
        self.config.reproduction_config.elitism = 0
        self.config.reproduction_config.min_species_size = 4
        self.config.reproduction_config.survival_threshold = 1.0
        self.config.reproduction_config.crossover_prob = 0.0
        self.config.stagnation_config.species_fitness_func = 'max'
        self.config.stagnation_config.max_stagnation = 10
        self.config.stagnation_config.species_elitism = 0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.bias_mutate_prob = 0.0
        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0
        self.config.reproduction_config.crossover_prob = 0.0

        # config = configparser.ConfigParser()
        # config[NETWORK] = {
        #     NUM_INPUTS: 1,
        #     NUM_OUTPUTS: 1,
        # }
        # config[NEAT] = {
        #     POPULATION_SIZE: 10,
        #     ELITES: 0,
        #     ELITISM_THRESHOLD: 4,
        #     SURVIVAL_THRESHOLD: 1.0,
        #     CROSSOVER_PROB: 0.0,
        #     CONNECTION_GENE_DISABLE_PROB: 0.0,
        #     INTER_SPECIES_CROSSOVER_PROB: 0.0,
        #     WEIGHT_MUTATION_PROB: 0.0,
        #     WEIGHT_REPLACE_PROB: 0.0,
        #     WEIGHT_PERTURB_STD_DEV: 0.2,
        #     ADD_CONNECTION_PROB: 0.0,
        #     ADD_NODE_PROB: 0.0,
        #     DIST_COEFF_1: 1.0,
        #     DIST_COEFF_2: 1.0,
        #     SPECIATION_THRESHOLD: 10.0,
        #     STAGNATION_THRESHOLD: 10,
        # }
        stagnation_scheme = DefaultStagnation(self.config.stagnation_config, self.reporters)
        reproduction_scheme = Reproduction(self.config.reproduction_config, self.reporters, stagnation_scheme)
        pop_size = 10
        population0 = reproduction_scheme.create_new(Genome, self.config.genome_config, pop_size, InnovationStore())
        for genome in population0.values():
            genome.fitness = 1

        for genome_id, genome in list(population0.items())[:5]:
            # Increase weight and biases for half the genomes to ensure two species
            for g in genome.nodes.values():
                g.bias += 50
            for g in genome.connections.values():
                g.weight += 50

        species_set = SpeciesSet(self.config.species_set_config, self.reporters)
        species_set.speciate(self.config, population0, generation=0)

        assert 2 == len(species_set.species)

        population1 = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=1, innovation_store=InnovationStore())
        species_set.speciate(self.config, population1, generation=10)
        for genome in population1.values():
            genome.fitness = 1

        assert 2 == len(species_set.species)

        # Skip to generation 11 to stagnate all species
        population2 = reproduction_scheme.reproduce(self.config, species_set, pop_size, generation=11, innovation_store=InnovationStore())
        species_set.speciate(self.config, population2, generation=11)
        for genome in population2.values():
            genome.fitness = 1

        assert 0 == len(species_set.species)
