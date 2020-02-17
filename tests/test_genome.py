"""
Tests for the genome encoding module.
"""

import itertools
import configparser
import os

import pytest
import neat
from neat.activations import identity_activation

from custom_neat.genome import *
from custom_neat.species import SpeciesSet
from custom_neat.reproduction import Reproduction
from custom_neat.innovation import InnovationStore

__author__ = "Adam Tupper"
__copyright__ = "Adam Tupper"
__license__ = "mit"


class TestGenome:
    def setup_method(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'configurations', 'test_configuration.ini')
        self.config = neat.Config(Genome,
                                  Reproduction,
                                  SpeciesSet,
                                  neat.DefaultStagnation,
                                  config_path)

    def test_create_node_gene(self):
        """Test the creation of node genes.
        """
        node_gene = NodeGene(key=0,
                             type=NodeType.INPUT,
                             bias=0.0,
                             activation=identity_activation)
        assert node_gene.key == 0
        assert node_gene.type == NodeType.INPUT
        assert pytest.approx(0.0, node_gene.bias)
        assert identity_activation == node_gene.activation

    def test_create_connection_gene(self):
        """Test the creation of connection genes.
        """
        in_node = 0
        out_node = 1
        key = 2
        connection_gene = ConnectionGene(
            key=key,
            node_in=in_node,
            node_out=out_node,
            weight=1.0,
            expressed=True
        )

        assert connection_gene.node_in == in_node
        assert connection_gene.node_out == out_node
        assert connection_gene.key == key
        assert connection_gene.weight == pytest.approx(1.0)
        assert connection_gene.expressed

    def test_create_genome(self):
        """Test the Genome constructor.
        """
        innovation_store = InnovationStore()
        genome = Genome(key=0, config=None, innovation_store=innovation_store)

        assert innovation_store == genome.innovation_store
        assert 0 == genome.key
        assert genome.fitness is None
        assert {} == genome.nodes
        assert {} == genome.connections
        assert [] == genome.inputs
        assert [] == genome.outputs

    def test_configure_new(self):
        """Test randomly configuring a new genome .
        """
        # Alter relevant configuration parameters for this test
        self.config.genome_config.num_inputs = 2
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.bias_init_std_dev = 1.0
        self.config.genome_config.weight_init_std_dev = 1.0
        self.config.genome_config.init_conn_prob = 1.0  # fully-connected

        num_input_nodes = self.config.genome_config.num_inputs
        num_output_nodes = self.config.genome_config.num_outputs
        innovation_store = InnovationStore()
        genome = Genome(key=0, config=self.config.genome_config, innovation_store=innovation_store)
        genome.configure_new()

        assert 0 == genome.key
        assert genome.fitness is None

        assert num_input_nodes + num_output_nodes == len(genome.nodes)
        # Initial connection probability is 1.0 in config, therefore network
        # should be fully connected.
        assert num_input_nodes * num_output_nodes == len(genome.connections)

        actual_node_types = [gene.type for k, gene in genome.nodes.items()]
        expected_node_types = [NodeType.INPUT] * num_input_nodes + [NodeType.OUTPUT] * num_output_nodes
        assert actual_node_types == expected_node_types

        three_std_dev = 3 * self.config.genome_config.bias_init_std_dev
        biases = [gene.bias for k, gene in genome.nodes.items()]
        assert all(-three_std_dev <= b <= three_std_dev for b in biases)

        three_std_dev = 3 * self.config.genome_config.weight_init_std_dev
        weights = [gene.weight for k, gene in genome.connections.items()]
        assert all(-three_std_dev <= w <= three_std_dev for w in weights)

        assert all([gene.expressed for k, gene in genome.connections.items()])

        actual_conns = set(genome.connections.keys())
        input_node_indices = list(range(0, num_input_nodes))
        output_node_indices = list(range(num_input_nodes, num_input_nodes + num_output_nodes))
        expected_conns = set(itertools.product(input_node_indices, output_node_indices))
        assert actual_conns == expected_conns

    def test_copy(self):
        """Test copying a genome.
        """
        genome = Genome(key=0, config=None, innovation_store=InnovationStore())
        duplicate = genome.copy()

        assert genome == duplicate
        assert genome.__str__() != duplicate.__str__()

    def test_add_connection(self):
        """Test adding connections to the genome.
        """
        genome = Genome(key=0, config=None, innovation_store=InnovationStore())

        # Add a dummy connection between non-existent nodes
        genome.add_connection(node_in=0, node_out=1, weight=1.0)
        assert 1 == len(genome.connections)

        new_gene = genome.connections[(0, 1)]
        assert 0 == new_gene.node_in
        assert 1 == new_gene.node_out
        assert -3.0 <= new_gene.weight <= 3.0  # assert it is within 3 std dev of mean
        assert new_gene.expressed
        assert 0 == new_gene.key

    def test_mutate_only_weights(self):
        """Test the mutation function behaves as expected when weight mutations
        are guaranteed but bias and structural mutations are prohibited.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 2
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.init_conn_prob = 1.0

        self.config.genome_config.weight_mutate_prob = 1.0
        self.config.genome_config.weight_replace_prob = 0.5
        self.config.genome_config.weight_perturb_std_dev = 0.2

        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0
        self.config.genome_config.bias_mutate_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_biases = [g.bias for g in genome.nodes.values()]
        old_weights = [g.weight for g in genome.connections.values()]
        genome.mutate()

        assert 3 == len(genome.nodes)
        assert 2 == len(genome.connections)

        new_biases = [g.bias for g in genome.nodes.values()]
        assert all(old == new for (old, new) in zip(old_biases, new_biases))

        new_weights = [g.weight for g in genome.connections.values()]
        assert all(old != new for (old, new) in zip(old_weights, new_weights))

    def test_mutate_only_biases(self):
        """Test the mutation function behaves as expected when bias mutations
        are guaranteed but weight and structural mutations are prohibited.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 2
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.init_conn_prob = 1.0

        self.config.genome_config.bias_mutate_prob = 1.0
        self.config.genome_config.bias_replace_prob = 0.5
        self.config.genome_config.bias_perturb_std_dev = 0.2

        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0
        self.config.genome_config.weight_mutate_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_biases = [g.bias for g in genome.nodes.values()]
        old_weights = [g.weight for g in genome.connections.values()]
        genome.mutate()

        assert 3 == len(genome.nodes)
        assert 2 == len(genome.connections)

        new_biases = [g.bias for g in genome.nodes.values()]
        assert all(old != new for (old, new) in zip(old_biases, new_biases))

        new_weights = [g.weight for g in genome.connections.values()]
        assert all(old == new for (old, new) in zip(old_weights, new_weights))

    def test_mutate_only_connections(self):
        """Test the mutation function behaves as expected when the
        'add connection' mutation is guaranteed, but weight, bias and 'add node'
        mutations are prohibited.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.init_conn_prob = 0.0  # no initial connections

        self.config.genome_config.conn_add_prob = 1.0

        self.config.genome_config.bias_mutate_prob = 0.0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()
        genome.connections = {}

        assert not genome.connections

        old_biases = [g.bias for g in genome.nodes.values()]
        genome.mutate()

        assert 2 == len(genome.nodes)
        assert 1 == len(genome.connections)

        new_biases = [g.bias for g in genome.nodes.values()]
        assert all(old == new for (old, new) in zip(old_biases, new_biases))

    def test_mutate_only_nodes(self):
        """Test the mutation function behaves as expected when the 'add node'
        mutation is guaranteed, but weight, bias and 'add connection' mutations
        are prohibited.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.init_conn_prob = 1.0  # fully-connected

        self.config.genome_config.node_add_prob = 1.0

        self.config.genome_config.bias_mutate_prob = 0.0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.conn_add_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        num_node_genes = len(genome.nodes)
        num_connection_genes = len(genome.connections)

        old_biases = [g.bias for g in genome.nodes.values()]
        old_weights = [g.weight for g in genome.connections.values()]
        genome.mutate()

        assert num_node_genes + 1 == len(genome.nodes)
        assert num_connection_genes + 2 == len(genome.connections)

        new_biases = [g.bias for g in genome.nodes.values()]
        assert all(old == new for (old, new) in zip(old_biases, new_biases))

        new_weights = [g.weight for g in genome.connections.values()]
        assert all(old == new for (old, new) in zip(old_weights, new_weights))

    def test_mutate_none(self):
        """Test the mutation function behaves as expected when all mutations are
        prohibited.
        """
        # Alter configuration for test
        self.config.genome_config.node_add_prob = 0.0
        self.config.genome_config.bias_mutate_prob = 0.0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.conn_add_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()
        expected = genome.copy()

        genome.mutate()

        assert expected == genome

    def test_mutate_add_node(self):
        """Test the function for the 'add node' mutation.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.activation_func = 'identity'
        self.config.genome_config.init_conn_prob = 1.0
        self.config.genome_config.weight_init_std_dev = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        genome.mutate_add_node(activation=identity_activation)
        assert len(genome.connections) == 3
        assert len(genome.nodes) == 3

        new_node_gene = genome.nodes[max(genome.nodes.keys())]
        assert new_node_gene.type == NodeType.HIDDEN

        old_connection_gene = genome.connections[(0, 1)]
        assert not old_connection_gene.expressed

        new_connection_gene_a = genome.connections[(0, 3)]
        assert new_connection_gene_a.node_in == 0
        assert new_connection_gene_a.node_out == 3
        assert -3.0 <= new_connection_gene_a.weight <= 3.0  # assert weight is within 3 std dev of mean
        assert new_connection_gene_a.expressed

        new_connection_gene_b = genome.connections[(3, 1)]
        assert new_connection_gene_b.node_in == 3
        assert new_connection_gene_b.node_out == 1
        assert -3.0 <= new_connection_gene_b.weight <= 3.0  # assert weight is within 3 std dev of mean
        assert new_connection_gene_b.expressed

    def test_mutate_add_connection_fail(self):
        """Test the function for the 'add connection' mutation when all possible
        connections are already present (i.e. no connection should be added).
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.initial_conn_prob = 1.0  # fully-connected

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        # Manually add recurrent connections to saturate network
        genome.connections[(1, 1)] = ConnectionGene(3, 1, 1, 0.0, True)
        genome.connections[(0, 0)] = ConnectionGene(4, 0, 0, 0.0, True)
        genome.connections[(1, 0)] = ConnectionGene(5, 0, 0, 0.0, True)

        genome.mutate_add_connection(std_dev=1.0)
        assert 4 == len(genome.connections)

    def test_mutate_add_connection_succeed(self):
        """Test the function for the 'add connection' mutation when there are
        valid connections that can be added.
        """
        # Set random seed
        random.seed(0)

        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.initial_conn_prob = 0.0  # no connections

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        genome.mutate_add_connection(std_dev=1.0)
        assert 1 == len(genome.connections)

        assert 0 == genome.connections[(0, 1)].node_in
        assert 1 == genome.connections[(0, 1)].node_out

    def test_mutate_weights(self):
        """Test the mutation of genome connection weights.
        """
        # Test configuration
        self.config.genome_config.num_inputs = 3
        self.config.genome_config.num_outputs = 2
        self.config.genome_config.init_conn_prob = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_weights = [g.weight for g in genome.connections.values()]
        old_biases = [g.bias for g in genome.nodes.values()]
        genome.mutate_weights(replace_prob=0.5,
                              init_std_dev=0.5,
                              perturb_std_dev=0.5,
                              max_val=10,
                              min_val=-10)
        new_weights = [g.weight for g in genome.connections.values()]
        new_biases = [g.bias for g in genome.nodes.values()]

        assert 5 == len(genome.nodes)
        assert 6 == len(genome.connections)
        assert all(old != new for (old, new) in zip(old_weights, new_weights))
        assert all(old == new for (old, new) in zip(old_biases, new_biases))

    def test_mutate_biases(self):
        """Test the mutation of genome node biases.
        """
        # Test configuration
        self.config.genome_config.num_inputs = 3
        self.config.genome_config.num_outputs = 2
        self.config.genome_config.init_conn_prob = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_weights = [g.weight for g in genome.connections.values()]
        old_biases = [g.bias for g in genome.nodes.values()]
        genome.mutate_biases(replace_prob=0.5,
                             init_std_dev=0.5,
                             perturb_std_dev=0.5,
                             max_val=10,
                             min_val=-10)
        new_weights = [g.weight for g in genome.connections.values()]
        new_biases = [g.bias for g in genome.nodes.values()]

        assert 5 == len(genome.nodes)
        assert 6 == len(genome.connections)
        assert all(old == new for (old, new) in zip(old_weights, new_weights))
        assert all(old != new for (old, new) in zip(old_biases, new_biases))

    def test_crossover_ordered(self):
        """Test that the crossover operator function works as expected when the
        first parent passed has the higher fitness.
            """
        # Test configuration
        self.config.genome_config.gene_disable_prob = 0.0
        innovation_store = InnovationStore()

        parent1 = Genome(key=0, config=self.config.genome_config, innovation_store=innovation_store)
        for node_config in ((-1, -1, NodeType.INPUT), (-2, -2, NodeType.OUTPUT), (0, 1, NodeType.HIDDEN)):
            node_in, node_out, node_type = node_config
            parent1.add_node(node_in, node_out, 1.0, node_type)
        for connection_config in ((0, 2, 1.0), (2, 1, 1.0)):
            node_out, node_in, weight = connection_config
            parent1.add_connection(node_in, node_out, weight)
        parent1.fitness = 1.

        parent2 = Genome(key=1, config=self.config.genome_config)
        parent2.nodes = {
            0: NodeGene(key=0, type=NodeType.INPUT, bias=2., activation=identity_activation),
            1: NodeGene(key=1, type=NodeType.OUTPUT, bias=2., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
            3: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
        }
        parent2.add_connection(0, 2, 2.)
        parent2.add_connection(0, 3, 2.)
        parent2.add_connection(2, 1, 2.)
        parent2.add_connection(3, 1, 2.)
        parent2.fitness = 2.

        child = Genome(key=2, config=self.config.genome_config)
        child.configure_crossover(parent1, parent2)
        assert 4 == len(child.nodes)
        assert 4 == len(child.connections)
        assert 4 == next(child.node_key_generator)

        child = Genome(key=2, config=self.config.genome_config)
        child.configure_crossover(parent2, parent1)
        assert 4 == len(child.nodes)
        assert 4 == len(child.connections)
        assert 4 == next(child.node_key_generator)

    def test_crossover_reversed(self):
        """Test that the crossover operator function works as expected when the
        second parent has the higher fitness.
        """
        # Test configuration
        self.config.genome_config.gene_disable_prob = 0.0

        parent1 = Genome(key=0, config=self.config.genome_config)
        parent1.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=1., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=1., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=1., activation=identity_activation),
        }
        parent1.add_connection(0, 2, 1.)
        parent1.add_connection(2, 1, 1.)
        parent1.fitness = 1.

        parent2 = Genome(key=1, config=self.config.genome_config)
        parent2.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=2., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=2., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
            3: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
        }
        parent2.add_connection(0, 2, 2.)
        parent2.add_connection(0, 3, 2.)
        parent2.add_connection(2, 1, 2.)
        parent2.add_connection(3, 1, 2.)
        parent2.fitness = 2.

        child = Genome(key=2, config=self.config.genome_config)
        child.configure_crossover(parent2, parent1)
        assert 4 == len(child.nodes)
        assert 4 == len(child.nodes)
        assert 4 == next(child.node_key_generator)

    def test_crossover_no_excess_or_disjoint(self):
        """Test that the crossover operator function works when the higher
        performing parent has no excess or disjoint genes to be inherited.
        """
        # Test configuration
        self.config.genome_config.gene_disable_prob = 0.0

        parent1 = Genome(key=0, config=self.config.genome_config)
        parent1.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=1., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=1., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=1., activation=identity_activation),
        }
        parent1.add_connection(0, 2, 1.)
        parent1.add_connection(2, 1, 1.)
        parent1.fitness = 2.

        parent2 = Genome(key=1, config=self.config.genome_config)
        parent2.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=2., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=2., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
            3: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
        }
        parent2.add_connection(0, 2, 2.)
        parent2.add_connection(0, 3, 2.)
        parent2.add_connection(2, 1, 2.)
        parent2.add_connection(3, 1, 2.)
        parent2.fitness = 1.

        child = Genome(key=2, config=self.config.genome_config)
        child.configure_crossover(parent1, parent2)
        assert 3 == len(child.nodes)
        assert 2 == len(child.connections)
        assert 3 == next(child.node_key_generator)

    def test_crossover_disable_mutual_genes(self):
        """Test that the crossover operator function works when the connection gene
        disable probability is 1.0. i.e. all mutual genes that are disabled in at
        least one parent should be disabled in the offspring.
        """
        # Test configuration
        self.config.genome_config.gene_disable_prob = 1.0

        parent1 = Genome(key=0,config=self.config.genome_config)
        parent1.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=1., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=1., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=1., activation=identity_activation),
        }
        parent1.add_connection(0, 2, 1., expressed=False)
        parent1.add_connection(2, 1, 1.)
        parent1.fitness = 2.

        parent2 = Genome(key=1, config=self.config.genome_config)
        parent2.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=2., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=2., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
        }
        parent2.add_connection(0, 2, 2.)
        parent2.add_connection(2, 1, 2., expressed=False)
        parent2.fitness = 1.

        child = Genome(key=2, config=self.config.genome_config)
        child.configure_crossover(parent1, parent2)
        assert 3 == len(child.nodes)
        assert 2 == len(child.connections)
        assert all([not g.expressed for g in child.connections.values()])
        assert 3 == next(child.node_key_generator)

    def test_distance(self):
        """Test the genetic distance method.
        """
        # Test configuration
        self.config.genome_config.compatibility_disjoint_coefficient = 1.0
        self.config.genome_config.compatibility_weight_coefficient = 1.0

        genome1 = Genome(key=0, config=self.config.genome_config)
        genome1.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=1., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=1., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=1., activation=identity_activation),
        }
        genome1.add_connection(0, 2, 1.)
        genome1.add_connection(2, 1, 1.)
        genome1.fitness = 2.

        genome2 = Genome(key=1, config=self.config.genome_config)
        genome2.nodes = {
            0: NodeGene(type=NodeType.INPUT, bias=2., activation=identity_activation),
            1: NodeGene(type=NodeType.OUTPUT, bias=2., activation=identity_activation),
            2: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
            3: NodeGene(type=NodeType.HIDDEN, bias=2., activation=identity_activation),
        }
        genome2.add_connection(0, 2, 2.)
        genome2.add_connection(0, 3, 2.)
        genome2.add_connection(2, 1, 2.)
        genome2.add_connection(3, 1, 2.)
        genome2.fitness = 1.

        assert pytest.approx(11 / 8, genome1.distance(genome2))
