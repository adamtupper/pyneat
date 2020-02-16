"""Tests for the species module.
"""

import os
import configparser

import pytest
import neat
from neat.reporting import ReporterSet
from neat.activations import  identity_activation

from custom_neat.species import *
from custom_neat.genome import *
from custom_neat.reproduction import Reproduction


class TestSpecies:
    def test_species_constructor(self):
        """Test the Species constructor.
        """
        species = Species(key=0, generation=1)

        assert 0 == species.key
        assert 1 == species.created
        assert 1 == species.last_improved
        assert {} == species.members
        assert species.fitness is None
        assert species.adjusted_fitness is None
        assert [] == species.fitness_history

    def test_species_update(self):
        """Test the Species update method.
        """
        species = Species(key=0, generation=1)
        new_representative = 0
        new_members = {
            0: Genome(key=0, config=None),
            1: Genome(key=1, config=None),
        }

        assert species.representative is None
        assert {} == species.members

        species.update(new_representative, new_members)

        assert new_representative == species.representative
        assert new_members == species.members


class TestSpeciesSet:
    def setup_method(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'configurations', 'test_configuration.ini')
        self.config = neat.Config(Genome,
                                  Reproduction,
                                  SpeciesSet,
                                  neat.DefaultStagnation,
                                  config_path)

    def test_species_set_constructor(self):
        """Test the SpeciesSet constructor.
        """
        start_index = 0
        species = {}

        species_set = SpeciesSet(self.config, ReporterSet())

        assert start_index == next(species_set.species_key_generator)
        assert species == species_set.species

    def test_species_set_speciate_initial(self):
        """Test the speciate method behaves as expected when creating species for
        the first time.
        """
        # Configure speciation parameters
        self.config.species_set_config.compatibility_threshold = 1.0
        self.config.genome_config.compatibility_disjoint_coefficient = 0.5
        self.config.genome_config.compatibility_weight_coefficient = 0.5

        # Create population
        generation = 1
        weights = [1.0, 1.5,  # First two genomes belong to the same species
                   10.0, 10.5]  # Second two genomes belong to the same species
        population = {}
        for i, weight in enumerate(weights):
            # Build genome
            genome = Genome(key=i, config=self.config.genome_config)
            genome.nodes = {
                0: NodeGene(type=NodeType.INPUT, bias=weight, activation=identity_activation),
                1: NodeGene(type=NodeType.OUTPUT, bias=weight, activation=identity_activation)
            }
            genome.connections = {
                (0, 1): ConnectionGene(
                    in_node=0,
                    out_node=1,
                    weight=weight,
                    expressed=True
                )
            }

            # Add genome to population
            population.update({i: genome})

        species_set = SpeciesSet(self.config, ReporterSet())
        species_set.speciate(self.config, population, generation)

        expected_species0 = Species(key=0, generation=generation)
        expected_species0.update(representative=population[0], members={0: population[0], 1: population[1]})

        expected_species1 = Species(key=1, generation=generation)
        expected_species1.update(representative=population[2], members={2: population[2], 3: population[3]})

        assert 2 == len(species_set.species)
        assert 2 == next(species_set.species_key_generator)
        assert expected_species0 == species_set.species[0]
        assert expected_species1 == species_set.species[1]

    def test_species_set_speciate_new_species(self):
        """Test the speciate method behaves as expected when a new species needs to
        be created.
        """
        # Configure speciation parameters
        self.config.species_set_config.compatibility_threshold = 1.0
        self.config.genome_config.compatibility_disjoint_coefficient = 0.5
        self.config.genome_config.compatibility_weight_coefficient = 0.5

        # Create population for generation 1
        generation = 1
        weights = [1.0, 1.5,  # First two genomes belong to the same species
                   10.0, 10.5]  # Second two genomes belong to the same species
        population = {}
        for i, weight in enumerate(weights):
            # Build genome
            genome = Genome(key=i, config=self.config.genome_config)
            genome.nodes = {
                0: NodeGene(type=NodeType.INPUT, bias=weight, activation=identity_activation),
                1: NodeGene(type=NodeType.OUTPUT, bias=weight, activation=identity_activation)
            }
            genome.connections = {
                (0, 1): ConnectionGene(
                    in_node=0,
                    out_node=1,
                    weight=weight,
                    expressed=True
                )
            }

            # Add genome to population
            population.update({i: genome})

        species_set = SpeciesSet(self.config, ReporterSet())
        species_set.speciate(self.config, population, generation)

        # Mutate population for generation 2
        generation = 2
        weights = [1.5,  # First genome belongs to the same species as before
                   5.0,  # Second genome splits off into it's own species
                   10.0, 10.5]  # Last two genomes belong to the same species
        for i, weight in enumerate(weights):
            # Update genomes
            genome = population[i]
            genome.nodes = {
                0: NodeGene(type=NodeType.INPUT, bias=weight, activation=identity_activation),
                1: NodeGene(type=NodeType.OUTPUT, bias=weight, activation=identity_activation)
            }
            genome.connections = {
                (0, 1): ConnectionGene(
                    in_node=0,
                    out_node=1,
                    weight=weight,
                    expressed=True
                )
            }

        species_set.speciate(self.config, population, generation)

        expected_species0 = Species(key=0, generation=1)
        expected_species0.update(representative=population[0], members={0: population[0]})

        expected_species1 = Species(key=1, generation=1)
        expected_species1.update(representative=population[2], members={2: population[2], 3: population[3]})

        expected_species2 = Species(key=2, generation=2)
        expected_species2.update(representative=population[1], members={1: population[1]})

        assert 3 == len(species_set.species)
        assert 3 == next(species_set.species_key_generator)
        assert expected_species0 == species_set.species[0]
        assert expected_species1 == species_set.species[1]
        assert expected_species2 == species_set.species[2]

    def test_species_set_speciate_no_new_species(self):
        """Test the speciate method behaves as expected when no new species should
        be created.
        """
        # Configure speciation parameters
        self.config.species_set_config.compatibility_threshold = 1.0
        self.config.genome_config.compatibility_disjoint_coefficient = 0.5
        self.config.genome_config.compatibility_weight_coefficient = 0.5

        # Create population for generation 1
        generation = 1
        weights = [1.0, 1.5,  # First two genomes belong to the same species
                   10.0, 10.5]  # Second two genomes belong to the same species
        population = {}
        for i, weight in enumerate(weights):
            # Build genome
            genome = Genome(key=i, config=self.config.genome_config)
            genome.nodes = {
                0: NodeGene(type=NodeType.INPUT, bias=weight, activation=identity_activation),
                1: NodeGene(type=NodeType.OUTPUT, bias=weight, activation=identity_activation)
            }
            genome.connections = {
                (0, 1): ConnectionGene(
                    in_node=0,
                    out_node=1,
                    weight=weight,
                    expressed=True
                )
            }

            # Add genome to population
            population.update({i: genome})

        species_set = SpeciesSet(self.config, ReporterSet())
        species_set.speciate(self.config, population, generation)

        # Mutate population for generation 2
        generation = 2
        weights = [1.2, 1.4,  # First two genomes belong to the same species
                   10.3, 10.1]  # Second two genomes belong to the same species
        for i, weight in enumerate(weights):
            # Update genomes
            genome = population[i]
            genome.nodes = {
                0: NodeGene(type=NodeType.INPUT, bias=weight, activation=identity_activation),
                1: NodeGene(type=NodeType.OUTPUT, bias=weight, activation=identity_activation)
            }
            genome.connections = {
                (0, 1): ConnectionGene(
                    in_node=0,
                    out_node=1,
                    weight=weight,
                    expressed=True
                )
            }

        species_set.speciate(self.config, population, generation)

        expected_species0 = Species(key=0, generation=1)
        expected_species0.update(representative=population[0], members={0: population[0], 1: population[1]})

        expected_species1 = Species(key=1, generation=1)
        expected_species1.update(representative=population[3], members={2: population[2], 3: population[3]})

        assert 2 == len(species_set.species)
        assert 2 == next(species_set.species_key_generator)
        assert expected_species0 == species_set.species[0]
        assert expected_species1 == species_set.species[1]
