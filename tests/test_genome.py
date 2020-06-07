"""
Tests for the genome encoding module.
"""
import os

import pytest
import neat
from pyneat.activations import identity_activation

from pyneat.genome import *
from pyneat.species import SpeciesSet
from pyneat.reproduction import Reproduction
from pyneat.innovation import InnovationStore, InnovationType, InnovationRecord
from pyneat.stagnation import Stagnation


class TestGenome:
    def setup_method(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'configurations', 'test_configuration.ini')
        self.config = neat.Config(Genome,
                                  Reproduction,
                                  SpeciesSet,
                                  Stagnation,
                                  config_path)

    def test_create_node_gene(self):
        """Test the creation of node genes.
        """
        node_gene = NodeGene(key=0,
                             type=NodeType.INPUT,
                             activation=None)
        assert node_gene.key == 0
        assert node_gene.type == NodeType.INPUT
        assert node_gene.activation is None

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
        assert [] == genome.biases

    def test_configure_new(self):
        """Test randomly configuring a new genome .
        """
        # Alter relevant configuration parameters for this test
        self.config.genome_config.num_inputs = 2
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.weight_init_power = 1.0
        self.config.genome_config.initial_conn_prob = 1.0  # fully-connected

        num_input_nodes = self.config.genome_config.num_inputs
        num_output_nodes = self.config.genome_config.num_outputs
        num_bias_nodes = self.config.genome_config.num_biases
        innovation_store = InnovationStore()
        genome = Genome(key=0, config=self.config.genome_config, innovation_store=innovation_store)
        genome.configure_new()

        assert 0 == genome.key
        assert genome.fitness is None

        assert len(genome.nodes) == num_input_nodes + num_output_nodes + num_bias_nodes
        assert len(genome.connections) == (num_input_nodes + num_bias_nodes) * num_output_nodes

        actual_node_types = [gene.type for k, gene in genome.nodes.items()]
        expected_node_types = [NodeType.INPUT] * num_input_nodes + \
                              [NodeType.OUTPUT] * num_output_nodes + \
                              [NodeType.BIAS] * num_bias_nodes
        assert actual_node_types == expected_node_types

        limit = self.config.genome_config.weight_init_power
        weights = [gene.weight for k, gene in genome.connections.items()]
        assert all(-limit <= w <= limit for w in weights)

        assert all([gene.expressed for k, gene in genome.connections.items()])

        actual_conn_keys = list(genome.connections.keys())
        actual_input_node_keys = list(range(0, num_input_nodes))
        actual_output_node_keys = list(range(num_input_nodes, num_input_nodes + num_output_nodes))
        actual_bias_node_keys = list(range(num_input_nodes + num_output_nodes, num_input_nodes + num_output_nodes + num_bias_nodes))
        expected_conn_keys = [4, 5, 6]
        expected_input_node_keys = [0, 1]
        expected_output_node_keys = [2]
        expected_bias_node_keys = [3]

        assert actual_conn_keys == expected_conn_keys
        assert actual_input_node_keys == expected_input_node_keys
        assert actual_output_node_keys == expected_output_node_keys
        assert actual_bias_node_keys == expected_bias_node_keys

    def test_copy(self):
        """Test copying a genome.
        """
        genome = Genome(key=0, config=None, innovation_store=InnovationStore())
        duplicate = genome.copy()

        assert genome == duplicate
        assert genome.__str__() != duplicate.__str__()

    def test_copy_keep_innov_store(self):
        """Test that the copy function works correctly, maintaining the same
        innovation store for all genomes.
        """
        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome_copy = genome.copy()

        assert genome.innovation_store == genome_copy.innovation_store

    def test_add_connection(self):
        """Test adding connections to the genome.
        """
        genome = Genome(key=0, config=None, innovation_store=InnovationStore())

        # Add a dummy connection between non-existent nodes
        genome.add_connection(node_in=0, node_out=1, weight=1.0)
        assert 1 == len(genome.connections)

        new_gene = genome.connections[0]
        assert 0 == new_gene.node_in
        assert 1 == new_gene.node_out
        assert -3.0 <= new_gene.weight <= 3.0  # assert it is within 3 std dev of mean
        assert new_gene.expressed
        assert 0 == new_gene.key

    def test_add_node(self):
        """Test add a node to the genome.
        """
        # Alter configuration for this test
        self.config.genome_config.activation_func = 'identity'

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())

        # Position the node between two dummy nodes (for innovation store purposes)
        genome.add_node(-1, -1, NodeType.INPUT)
        assert len(genome.nodes) == 1

        new_gene = genome.nodes[0]
        assert new_gene.key == 0
        assert new_gene.type == NodeType.INPUT
        assert new_gene.activation is None

    def test_add_bias_node(self):
        """Test adding a bias node to the genome.
        """
        # Alter configuration for this test
        self.config.genome_config.activation_func = 'sigmoid'

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())

        genome.add_bias_node(0)
        assert len(genome.nodes) == 1

        new_gene = genome.nodes[0]
        assert new_gene.key == 0
        assert new_gene.type == NodeType.BIAS
        assert new_gene.activation is None

    def test_mutate_only_weights(self):
        """Test the mutation function behaves as expected when weight mutations
        are guaranteed but bias and structural mutations are prohibited.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 2
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 1.0

        self.config.genome_config.weight_mutate_prob = 1.0
        self.config.genome_config.weight_replace_prob = 0.5
        self.config.genome_config.weight_perturb_power = 0.2

        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_weights = [g.weight for g in genome.connections.values()]
        genome.mutate()

        assert 4 == len(genome.nodes)
        assert 3 == len(genome.connections)

        new_weights = [g.weight for g in genome.connections.values()]
        assert all(old != new for (old, new) in zip(old_weights, new_weights))

    def test_mutate_only_connections(self):
        """Test the mutation function behaves as expected when the
        'add connection' mutation is guaranteed, but weight, bias and 'add node'
        mutations are prohibited.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 0.0  # no initial connections

        self.config.genome_config.conn_add_prob = 1.0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.node_add_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()
        assert len(genome.connections) == 1

        genome.mutate()

        assert 3 == len(genome.nodes)
        assert 2 == len(genome.connections)

    def test_mutate_only_nodes(self):
        """Test the mutation function behaves as expected when the 'add node'
        mutation is guaranteed, but weight, bias and 'add connection' mutations
        are prohibited.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 1.0  # fully-connected

        self.config.genome_config.node_add_prob = 1.0

        self.config.genome_config.bias_mutate_prob = 0.0
        self.config.genome_config.weight_mutate_prob = 0.0
        self.config.genome_config.conn_add_prob = 0.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        num_node_genes = len(genome.nodes)
        num_connection_genes = len(genome.connections)

        old_weights = [g.weight for g in genome.connections.values()]
        genome.mutate()

        assert len(genome.nodes) == num_node_genes + 1
        assert len(genome.connections) == num_connection_genes + 2

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

    def test_mutate_add_node_fail(self):
        """Test the function for the 'add node' mutation fails when the
        mutation is already present in the genome.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.activation_func = 'identity'
        self.config.genome_config.initial_conn_prob = 1.0
        self.config.genome_config.weight_init_power = 1.0

        # Populate the innovation store with the mutation that will attempt to
        # be duplicated
        innovation_store = InnovationStore()
        innovation_store.key_to_record[3] = InnovationRecord(3, InnovationType.NEW_NODE, 0, 1)
        innovation_store.mutation_to_key[(0, 1, InnovationType.NEW_NODE)] = 3

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=innovation_store)
        genome.configure_new()

        # Add the node that will attempt to be added via mutation
        genome.add_node(0, 1, NodeType.HIDDEN)

        node_added = genome.mutate_add_node()
        assert not node_added
        assert len(genome.connections) == 2
        assert len(genome.nodes) == 4

        connection_gene = genome.connections[3]
        assert connection_gene.expressed

    def test_mutate_add_node_succeed(self):
        """Test the function for the 'add node' mutation succeeds when the
        mutation is not already present in the genome.
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.activation_func = 'identity'
        self.config.genome_config.initial_conn_prob = 1.0
        self.config.genome_config.weight_init_power = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        node_added = genome.mutate_add_node()
        assert node_added
        assert len(genome.connections) == 4
        assert len(genome.nodes) == 4

        new_node_gene = genome.nodes[max(genome.nodes.keys())]
        assert new_node_gene.type == NodeType.HIDDEN

        old_connection_gene = genome.connections[3]
        assert not old_connection_gene.expressed

        limit = self.config.genome_config.weight_init_power

        new_connection_gene_a = genome.connections[6]
        assert new_connection_gene_a.node_in == 0
        assert new_connection_gene_a.node_out == 5
        assert -limit <= new_connection_gene_a.weight <= limit  # assert weight is within the limits
        assert new_connection_gene_a.expressed

        new_connection_gene_b = genome.connections[7]
        assert new_connection_gene_b.node_in == 5
        assert new_connection_gene_b.node_out == 1
        assert -limit <= new_connection_gene_b.weight <= limit  # assert weight is within the limits
        assert new_connection_gene_b.expressed

    def test_mutate_add_connection_fail(self):
        """Test the function for the 'add connection' mutation when all possible
        connections are already present (i.e. no connection should be added).
        """
        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 1.0  # fully-connected
        self.config.genome_config.weight_init_power = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        # Manually add recurrent connections to saturate network
        genome.add_connection(1, 1, 0.0, True)

        assert len(genome.connections) == 3
        connection_added = genome.mutate_add_connection()
        assert not connection_added
        assert len(genome.connections) == 3

    def test_mutate_add_connection_succeed(self):
        """Test the function for the 'add connection' mutation when there are
        valid connections that can be added.
        """
        # Set random seed
        random.seed(0)

        # Alter configuration for this test
        self.config.genome_config.num_inputs = 1
        self.config.genome_config.num_outputs = 1
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 0.0  # no connections
        self.config.genome_config.weight_init_power = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        # Make sure there are no connections to start with
        assert len(genome.connections) == 1

        connection_added = genome.mutate_add_connection()
        assert connection_added
        # Make sure only one connection was added
        assert len(genome.connections) == 2

        # Make sure the a recurrent connection was not added to an input or bias node
        assert genome.connections[4].node_out != 0
        assert genome.connections[4].node_out != 2

    def test_mutate_weights(self):
        """Test the mutation of genome connection weights.
        """
        # Test configuration
        self.config.genome_config.num_inputs = 3
        self.config.genome_config.num_outputs = 2
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 1.0
        self.config.genome_config.weight_replace_prob = 0.5
        self.config.genome_config.weight_init_power = 1.0
        self.config.genome_config.weight_perturb_power = 1.0
        self.config.genome_config.weight_max_value = 10
        self.config.genome_config.weight_min_value = -10

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_weights = [g.weight for g in genome.connections.values()]
        genome.mutate_weights()
        new_weights = [g.weight for g in genome.connections.values()]
        upper_limit = self.config.genome_config.weight_max_value
        lower_limit = self.config.genome_config.weight_min_value

        assert 6 == len(genome.nodes)
        assert 8 == len(genome.connections)
        assert all(old != new for (old, new) in zip(old_weights, new_weights))
        assert all(lower_limit <= w <= upper_limit for w in new_weights)

    def test_mutate_structural_only_connection(self):
        """Test that if an add connection structural mutation is performed no
        weight or bias mutations are performed.
        """
        # Test configuration
        self.config.genome_config.num_inputs = 3
        self.config.genome_config.num_outputs = 2
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 1.0
        self.config.genome_config.conn_add_prob = 1.0  # Guarantee a connection will be added
        self.config.genome_config.node_add_prob = 0.0
        self.config.genome_config.weight_mutate_prob = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_weights = [g.weight for g in genome.connections.values()]
        assert len(genome.connections) == 8

        genome.mutate()
        new_weights = [g.weight for g in genome.connections.values()]

        assert len(genome.connections) == 9
        assert all(old == new for (old, new) in zip(old_weights, new_weights))

    def test_mutate_structural_only_node(self):
        """Test that if an add node structural mutation is performed no
        weight or bias mutations are performed.
        """
        # Test configuration
        self.config.genome_config.num_inputs = 3
        self.config.genome_config.num_outputs = 2
        self.config.genome_config.num_biases = 1
        self.config.genome_config.initial_conn_prob = 1.0
        self.config.genome_config.node_add_prob = 1.0  # Guarantee a connection will be added
        self.config.genome_config.conn_add_prob = 0.0
        self.config.genome_config.weight_mutate_prob = 1.0

        genome = Genome(key=0, config=self.config.genome_config, innovation_store=InnovationStore())
        genome.configure_new()

        old_weights = [g.weight for g in genome.connections.values()]
        assert len(genome.nodes) == 6
        assert len(genome.connections) == 8

        genome.mutate()
        new_weights = [g.weight for g in genome.connections.values()]

        assert len(genome.nodes) == 7
        print([(g.node_in, g.node_out) for g in genome.connections.values()])
        assert len(genome.connections) == 10
        assert all(old == new for (old, new) in zip(old_weights, new_weights))

    def test_crossover_a(self):
        """Test that the crossover operator.
        """
        # Set seed for consistency (of mutual gene inheritance)
        random.seed(3)

        # Test configuration
        self.config.genome_config.gene_disable_prob = 0.0

        # Construct parent 1
        parent1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        parent1.fitness = 2.
        parent1.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.OUTPUT, identity_activation),
            5: NodeGene(5, NodeType.HIDDEN, identity_activation)
        }
        parent1.connections = {
            3: ConnectionGene(key=3, node_in=0, node_out=2, weight=1., expressed=False),
            4: ConnectionGene(key=4, node_in=1, node_out=2, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=0, node_out=5, weight=1., expressed=True),
            7: ConnectionGene(key=7, node_in=5, node_out=2, weight=1., expressed=True)
        }

        # Construct parent 2
        parent2 = Genome(key=1, config=self.config.genome_config, innovation_store=None)
        parent2.fitness = 1.
        parent2.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.OUTPUT, identity_activation),
            8: NodeGene(8, NodeType.HIDDEN, identity_activation),
        }
        parent2.connections = {
            3: ConnectionGene(key=3, node_in=0, node_out=2, weight=2., expressed=True),
            4: ConnectionGene(key=4, node_in=1, node_out=2, weight=2., expressed=False),
            9: ConnectionGene(key=9, node_in=1, node_out=8, weight=2., expressed=True),
            10: ConnectionGene(key=10, node_in=8, node_out=2, weight=2., expressed=True)
        }

        child = Genome(key=2, config=self.config.genome_config, innovation_store=None)
        child.configure_crossover(parent1, parent2, average=False)
        assert 4 == len(child.nodes)
        assert 4 == len(child.connections)

        # Check that the structure of the child matches the fitter parent, and
        # that mutual genes were inherited from both parents.
        assert [0, 1, 2, 5] == list(child.nodes.keys())
        assert [3, 4, 6, 7] == list(child.connections.keys())
        assert child.inputs == [0, 1]
        assert child.outputs == [2]
        assert child.biases == []
        assert 2.0 in [g.weight for g in child.connections.values()]

        child = Genome(key=2, config=self.config.genome_config, innovation_store=None)
        child.configure_crossover(parent2, parent1, average=False)
        assert 4 == len(child.nodes)
        assert 4 == len(child.connections)

        assert [0, 1, 2, 5] == list(child.nodes.keys())
        assert [3, 4, 6, 7] == list(child.connections.keys())
        assert child.inputs == [0, 1]
        assert child.outputs == [2]
        assert child.biases == []
        assert 2.0 in [g.weight for g in child.connections.values()]

    def test_crossover_b(self):
        """Test that the crossover operator.
        """
        # Set seed for consistency (of mutual gene inheritance)
        random.seed(3)

        # Test configuration
        self.config.genome_config.gene_disable_prob = 0.0

        # Construct parent 1
        parent1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        parent1.fitness = 1.
        parent1.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.OUTPUT, identity_activation),
            3: NodeGene(3, NodeType.BIAS, None),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation)
        }
        parent1.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=2, weight=1., expressed=False),
            5: ConnectionGene(key=5, node_in=1, node_out=2, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=3, node_out=2, weight=1., expressed=True),
            8: ConnectionGene(key=8, node_in=0, node_out=7, weight=1., expressed=True),
            9: ConnectionGene(key=9, node_in=7, node_out=2, weight=1., expressed=True)
        }

        # Construct parent 2
        parent2 = Genome(key=1, config=self.config.genome_config, innovation_store=None)
        parent2.fitness = 2.
        parent2.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.OUTPUT, identity_activation),
            3: NodeGene(3, NodeType.BIAS, None),
            10: NodeGene(10, NodeType.HIDDEN, identity_activation),
        }
        parent2.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=2, weight=2., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=2, weight=2., expressed=False),
            6: ConnectionGene(key=6, node_in=3, node_out=0, weight=2., expressed=True),
            11: ConnectionGene(key=11, node_in=1, node_out=10, weight=2., expressed=True),
            12: ConnectionGene(key=12, node_in=10, node_out=2, weight=2., expressed=True),
        }

        child = Genome(key=2, config=self.config.genome_config, innovation_store=None)
        child.configure_crossover(parent1, parent2, average=False)
        assert 5 == len(child.nodes)
        assert 5 == len(child.connections)

        # Check that the structure of the child matches the fitter parent, and
        # that mutual genes were inherited from both parents.
        assert list(child.nodes.keys()) == [0, 1, 2, 3, 10]
        assert [4, 5, 6, 11, 12] == list(child.connections.keys())
        assert child.inputs == [0, 1]
        assert child.outputs == [2]
        assert child.biases == [3]
        assert 1.0 in [g.weight for g in child.connections.values()]

        child = Genome(key=2, config=self.config.genome_config, innovation_store=None)
        child.configure_crossover(parent2, parent1, average=False)
        assert 5 == len(child.nodes)
        assert 5 == len(child.connections)

        # Check that the structure of the child matches the fitter parent, and
        # that mutual genes were inherited from both parents.
        assert list(child.nodes.keys()) == [0, 1, 2, 3, 10]
        assert [4, 5, 6, 11, 12] == list(child.connections.keys())
        assert child.inputs == [0, 1]
        assert child.outputs == [2]
        assert child.biases == [3]
        assert 1.0 in [g.weight for g in child.connections.values()]

    def test_crossover_no_excess_or_disjoint(self):
        """Test that the crossover operator function works when the higher
        performing parent has no excess or disjoint genes to be inherited.
        """
        # Test configuration
        self.config.genome_config.gene_disable_prob = 0.0

        parent1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        parent1.fitness = 2.
        parent1.nodes = {
            0: NodeGene(key=0, type=NodeType.INPUT, activation=None),
            1: NodeGene(key=1, type=NodeType.OUTPUT, activation=identity_activation),
        }
        parent1.connections = {
            2: ConnectionGene(key=2, node_in=0, node_out=1, weight=1., expressed=True)
        }

        parent2 = Genome(key=1, config=self.config.genome_config, innovation_store=None)
        parent2.fitness = 1.
        parent2.nodes = {
            0: NodeGene(key=0, type=NodeType.INPUT, activation=None),
            1: NodeGene(key=1, type=NodeType.OUTPUT, activation=identity_activation),
            3: NodeGene(key=3, type=NodeType.HIDDEN, activation=identity_activation)
        }
        parent2.connections = {
            2: ConnectionGene(key=2, node_in=0, node_out=1, weight=2., expressed=False),
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=2., expressed=True),
            5: ConnectionGene(key=5, node_in=3, node_out=1, weight=2., expressed=True)
        }

        child = Genome(key=2, config=self.config.genome_config, innovation_store=None)
        child.configure_crossover(parent1, parent2, average=False)
        assert 2 == len(child.nodes)
        assert 1 == len(child.connections)
        assert [0, 1] == list(child.nodes)
        assert [2] == list(child.connections)

    def test_crossover_disable_mutual_genes(self):
        """Test that the crossover operator function works when the connection gene
        disable probability is 1.0. i.e. all mutual genes that are disabled in at
        least one parent should be disabled in the offspring.
        """
        # Test configuration
        self.config.genome_config.gene_disable_prob = 1.0

        parent1 = Genome(key=0,config=self.config.genome_config, innovation_store=None)
        parent1.fitness = 2.
        parent1.nodes = {
            0: NodeGene(key=0, type=NodeType.INPUT, activation=None),
            1: NodeGene(key=1, type=NodeType.OUTPUT, activation=identity_activation),
            3: NodeGene(key=3, type=NodeType.HIDDEN, activation=identity_activation)
        }
        parent1.connections = {
            2: ConnectionGene(key=2, node_in=0, node_out=1, weight=1., expressed=False),
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=False),
            5: ConnectionGene(key=5, node_in=3, node_out=1, weight=1., expressed=True)
        }

        parent2 = Genome(key=1, config=self.config.genome_config, innovation_store=None)
        parent2.fitness = 1.
        parent2.nodes = {
            0: NodeGene(key=0, type=NodeType.INPUT, activation=None),
            1: NodeGene(key=1, type=NodeType.OUTPUT, activation=identity_activation),
            3: NodeGene(key=3, type=NodeType.HIDDEN, activation=identity_activation)
        }
        parent2.connections = {
            2: ConnectionGene(key=2, node_in=0, node_out=1, weight=2., expressed=False),
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=2., expressed=True),
            5: ConnectionGene(key=5, node_in=3, node_out=1, weight=2., expressed=False)
        }

        child = Genome(key=2, config=self.config.genome_config, innovation_store=None)
        child.configure_crossover(parent1, parent2, average=False)
        assert 3 == len(child.nodes)
        assert 3 == len(child.connections)
        assert all([not g.expressed for g in child.connections.values()])

    def test_average_crossover(self):
        """Test that average crossover is performed correctly.
        """
        # Test configuration
        self.config.genome_config.gene_disable_prob = 0.0

        parent1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        parent1.fitness = 2.
        parent1.nodes = {
            0: NodeGene(key=0, type=NodeType.INPUT, activation=None),
            1: NodeGene(key=1, type=NodeType.INPUT, activation=None),
            2: NodeGene(key=2, type=NodeType.OUTPUT, activation=identity_activation),
            3: NodeGene(key=3, type=NodeType.BIAS, activation=identity_activation)
        }
        parent1.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=2, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=2, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=3, node_out=2, weight=1., expressed=True),
        }

        parent2 = Genome(key=1, config=self.config.genome_config, innovation_store=None)
        parent2.fitness = 1.
        parent2.nodes = {
            0: NodeGene(key=0, type=NodeType.INPUT, activation=None),
            1: NodeGene(key=1, type=NodeType.INPUT, activation=None),
            2: NodeGene(key=2, type=NodeType.OUTPUT, activation=identity_activation),
            3: NodeGene(key=3, type=NodeType.BIAS, activation=identity_activation)
        }
        parent2.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=2, weight=2., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=2, weight=2., expressed=True),
            6: ConnectionGene(key=6, node_in=3, node_out=2, weight=2., expressed=True),
        }

        child = Genome(key=2, config=self.config.genome_config, innovation_store=None)
        child.configure_crossover(parent1, parent2, average=True)
        assert 4 == len(child.nodes)
        assert 3 == len(child.connections)
        assert [0, 1, 2, 3] == list(child.nodes)
        assert [4, 5, 6] == list(child.connections)
        assert all([cg.weight == 1.5 for cg in child.connections.values()])

    def test_distance(self):
        """Test the genetic distance method.
        """
        # Test configuration
        self.config.genome_config.compatibility_disjoint_coefficient = 1.0
        self.config.genome_config.compatibility_weight_coefficient = 3.0

        genome1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome1.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation)
        }
        genome1.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1.94, expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=-0.36, expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=-0.61, expressed=True)
        }

        genome2 = Genome(key=1, config=self.config.genome_config, innovation_store=None)
        genome2.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.OUTPUT, identity_activation),
        }
        genome2.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=2.69, expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=-0.18, expressed=False),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1.18, expressed=True),
            8: ConnectionGene(key=8, node_in=1, node_out=7, weight=1.0, expressed=True),
            9: ConnectionGene(key=9, node_in=7, node_out=3, weight=-0.18, expressed=True)
        }

        assert genome1.distance(genome2) == pytest.approx(5.72, abs=1e-3)
        assert genome2.distance(genome1) == pytest.approx(5.72, abs=1e-3)

    def test_distance_normalisation_enabled_large_genome(self):
        """Test that the gene distance is normalised when it is enabled AND
        the genomes are larger than the threshold.
        """
        self.config.genome_config.compatibility_disjoint_coefficient = 1.0
        self.config.genome_config.compatibility_weight_coefficient = 1.0
        self.config.genome_config.normalise_gene_dist = True

        genome1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome1.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation),
            11: NodeGene(7, NodeType.HIDDEN, identity_activation),
            12: NodeGene(7, NodeType.HIDDEN, identity_activation),
            13: NodeGene(7, NodeType.HIDDEN, identity_activation),
            14: NodeGene(7, NodeType.HIDDEN, identity_activation),
            15: NodeGene(7, NodeType.HIDDEN, identity_activation),
            16: NodeGene(7, NodeType.HIDDEN, identity_activation),
            17: NodeGene(7, NodeType.HIDDEN, identity_activation),
            18: NodeGene(7, NodeType.HIDDEN, identity_activation),
            19: NodeGene(7, NodeType.HIDDEN, identity_activation),
            20: NodeGene(20, NodeType.HIDDEN, identity_activation)
        }
        genome1.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        genome2 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome2.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation),
            11: NodeGene(7, NodeType.HIDDEN, identity_activation),
            12: NodeGene(7, NodeType.HIDDEN, identity_activation),
            13: NodeGene(7, NodeType.HIDDEN, identity_activation),
            14: NodeGene(7, NodeType.HIDDEN, identity_activation),
        }

        genome2.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        assert genome1.distance(genome2) == pytest.approx(0.2857, abs=1e-4)
        assert genome2.distance(genome1) == pytest.approx(0.2857, abs=1e-4)

    def test_distance_normalisation_disabled_large_genome(self):
        """Test that the gene distance isn't normalised when the genomes are
        large enough (at least one of them) but normalisation is disabled.
        """
        self.config.genome_config.compatibility_disjoint_coefficient = 1.0
        self.config.genome_config.compatibility_weight_coefficient = 1.0
        self.config.genome_config.normalise_gene_dist = False

        genome1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome1.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation),
            11: NodeGene(7, NodeType.HIDDEN, identity_activation),
            12: NodeGene(7, NodeType.HIDDEN, identity_activation),
            13: NodeGene(7, NodeType.HIDDEN, identity_activation),
            14: NodeGene(7, NodeType.HIDDEN, identity_activation),
            15: NodeGene(7, NodeType.HIDDEN, identity_activation),
            16: NodeGene(7, NodeType.HIDDEN, identity_activation),
            17: NodeGene(7, NodeType.HIDDEN, identity_activation),
            18: NodeGene(7, NodeType.HIDDEN, identity_activation),
            19: NodeGene(7, NodeType.HIDDEN, identity_activation),
            20: NodeGene(20, NodeType.HIDDEN, identity_activation)
        }
        genome1.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        genome2 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome2.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation),
            11: NodeGene(7, NodeType.HIDDEN, identity_activation),
            12: NodeGene(7, NodeType.HIDDEN, identity_activation),
            13: NodeGene(7, NodeType.HIDDEN, identity_activation),
            14: NodeGene(7, NodeType.HIDDEN, identity_activation),
        }

        genome2.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        assert genome1.distance(genome2) == pytest.approx(6)
        assert genome2.distance(genome1) == pytest.approx(6)

    def test_distance_normalisation_enabled_small_genome(self):
        """Test that the gene distance isn't normalised when it is enabled, but
        the genomes are smaller than the threshold.
        """
        self.config.genome_config.compatibility_disjoint_coefficient = 1.0
        self.config.genome_config.compatibility_weight_coefficient = 1.0
        self.config.genome_config.normalise_gene_dist = True

        genome1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome1.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation)
        }
        genome1.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        genome2 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome2.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation),
            11: NodeGene(7, NodeType.HIDDEN, identity_activation),
            12: NodeGene(7, NodeType.HIDDEN, identity_activation),
        }

        genome2.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        assert genome1.distance(genome2) == pytest.approx(2)
        assert genome2.distance(genome1) == pytest.approx(2)

    def test_distance_normalisation_disabled_small_genome(self):
        """Test that the gene distance isn't normalised when it is disabled AND
        the genomes are smaller than the threshold.
        """
        self.config.genome_config.compatibility_disjoint_coefficient = 1.0
        self.config.genome_config.compatibility_weight_coefficient = 1.0
        self.config.genome_config.normalise_gene_dist = False

        genome1 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome1.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation)
        }
        genome1.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        genome2 = Genome(key=0, config=self.config.genome_config, innovation_store=None)
        genome2.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.INPUT, None),
            3: NodeGene(3, NodeType.OUTPUT, identity_activation),
            7: NodeGene(7, NodeType.HIDDEN, identity_activation),
            8: NodeGene(7, NodeType.HIDDEN, identity_activation),
            9: NodeGene(7, NodeType.HIDDEN, identity_activation),
            10: NodeGene(7, NodeType.HIDDEN, identity_activation),
            11: NodeGene(7, NodeType.HIDDEN, identity_activation),
            12: NodeGene(7, NodeType.HIDDEN, identity_activation),
        }

        genome2.connections = {
            4: ConnectionGene(key=4, node_in=0, node_out=3, weight=1., expressed=True),
            5: ConnectionGene(key=5, node_in=1, node_out=3, weight=1., expressed=True),
            6: ConnectionGene(key=6, node_in=2, node_out=3, weight=1., expressed=True)
        }

        assert genome1.distance(genome2) == pytest.approx(2)
        assert genome2.distance(genome1) == pytest.approx(2)
