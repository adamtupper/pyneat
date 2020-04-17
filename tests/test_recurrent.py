"""Tests for the recurrent neural network module.
"""

import pytest
from neat.activations import identity_activation, sigmoid_activation

from custom_neat.genome import *
from custom_neat.nn.recurrent import RNN


def test_create_all_required():
    """Test the create method of the RNN class.
    """
    # Manually create genome for deterministic testing
    genome = Genome(key=0, config=None, innovation_store=None)
    genome.nodes = {
        0: NodeGene(0, NodeType.INPUT, activation=sigmoid_activation),
        1: NodeGene(1, NodeType.OUTPUT, activation=sigmoid_activation),
        2: NodeGene(2, NodeType.BIAS, activation=identity_activation)
    }
    genome.connections = {3: ConnectionGene(3, 0, 1, 1.0, True),
                          4: ConnectionGene(3, 2, 1, 1.0, True)}
    genome.inputs = [0]
    genome.outputs = [1]
    genome.biases = [2]

    network = RNN.create(genome)

    assert network.input_nodes == [0]
    assert network.output_nodes == [1]
    assert network.node_evals == [(1, sigmoid_activation, [(0, 1.0), (2, 1.0)])]
    assert network.prev_states == {0: 0.0, 1: 0.0, 2: 1.0}
    assert network.curr_states == {0: 0.0, 1: 0.0, 2: 1.0}
    assert network.curr_states == {0: 0.0, 1: 0.0, 2: 1.0}


def test_create_some_required():
    """Test the create method of the RNN class correctly weeds out redundant
    nodes.
    """
    # Manually create genome for deterministic testing
    genome = Genome(key=0, config=None, innovation_store=None)
    genome.nodes = {
        0: NodeGene(0, NodeType.INPUT, activation=identity_activation),
        1: NodeGene(1, NodeType.INPUT, activation=identity_activation),
        2: NodeGene(2, NodeType.HIDDEN, activation=identity_activation),
        3: NodeGene(3, NodeType.HIDDEN, activation=identity_activation),
        4: NodeGene(4, NodeType.OUTPUT, activation=identity_activation),
        5: NodeGene(5, NodeType.OUTPUT, activation=identity_activation),
    }
    genome.connections = {
        6: ConnectionGene(6, 0, 2, 1.0, True),
        7: ConnectionGene(7, 0, 3, 2.0, True),
        8: ConnectionGene(8, 1, 3, 3.0, True),
        9: ConnectionGene(9, 2, 4, 4.0, True),
        10: ConnectionGene(10, 2, 5, 5.0, True),
    }
    genome.inputs = [0, 1]
    genome.outputs = [4, 5]

    network = RNN.create(genome)

    assert network.input_nodes == [0, 1]
    assert network.output_nodes == [4, 5]
    assert network.prev_states == {0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0}
    assert network.curr_states == {0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0}

    expected_node_evals = [
        (2, identity_activation, [(0, 1.0)]),
        (4, identity_activation, [(2, 4.0)]),
        (5, identity_activation, [(2, 5.0)]),
    ]
    assert network.node_evals == expected_node_evals


def test_reset():
    """Test the reset method.
    """
    # Manually create genome for deterministic testing
    genome = Genome(key=2, config=None, innovation_store=None)
    genome.nodes = {
        0: NodeGene(0, NodeType.INPUT, activation=identity_activation),
        1: NodeGene(1, NodeType.INPUT, activation=identity_activation),
        2: NodeGene(2, NodeType.HIDDEN, activation=identity_activation),
        3: NodeGene(3, NodeType.HIDDEN, activation=identity_activation),
        4: NodeGene(4, NodeType.OUTPUT, activation=identity_activation),
        5: NodeGene(5, NodeType.OUTPUT, activation=identity_activation),
        11: NodeGene(11, NodeType.BIAS, activation=identity_activation)
    }
    genome.connections = {
        6: ConnectionGene(6, 0, 2, 1.0, True),
        7: ConnectionGene(7, 0, 3, 2.0, True),
        8: ConnectionGene(8, 1, 3, 3.0, True),
        9: ConnectionGene(9, 2, 4, 4.0, True),
        10: ConnectionGene(10, 2, 5, 5.0, True),
        12: ConnectionGene(12, 11, 4, 6.0, True),
        13: ConnectionGene(13, 11, 5, 7.0, True),
    }

    network = RNN.create(genome)
    network.prev_states = {k: 1.0 for k, _ in network.prev_states.items()}
    network.curr_states = {k: 1.0 for k, _ in network.curr_states.items()}

    network.reset()
    assert network.prev_states == {k: 0.0 for k, _ in network.prev_states.items() if k != 11}
    assert network.curr_states == {k: 0.0 for k, _ in network.curr_states.items() if k != 11}

    assert network.prev_states == {k: 1.0 for k, _ in network.prev_states.items() if k == 11}
    assert network.curr_states == {k: 1.0 for k, _ in network.curr_states.items() if k == 11}


def test_forward_arch_1_in_order():
    """Test that the forward pass correctly calculates the right outputs over
    three time steps for an architecture that consists of a single input,
    hidden and output node, with a recurrent connection on the hidden node.

    Nodes are defined in topological order.
    """
    # Manually create genome for deterministic testing
    genome = Genome(key=0, config=None, innovation_store=None)
    genome.nodes = {
        0: NodeGene(0, type=NodeType.INPUT, activation=identity_activation),
        3: NodeGene(3, type=NodeType.BIAS, activation=identity_activation),
        1: NodeGene(1, type=NodeType.HIDDEN, activation=identity_activation),
        2: NodeGene(2, type=NodeType.OUTPUT, activation=identity_activation)
    }
    genome.connections = {
        4: ConnectionGene(4, node_in=0, node_out=1, weight=1., expressed=True),
        5: ConnectionGene(5, node_in=1, node_out=1, weight=2., expressed=True),
        6: ConnectionGene(6, node_in=1, node_out=2, weight=3., expressed=True),
        7: ConnectionGene(7, node_in=3, node_out=1, weight=4., expressed=True),
        6: ConnectionGene(8, node_in=3, node_out=2, weight=5., expressed=True),
    }

    network = RNN.create(genome)

    assert pytest.approx([0.0], network.forward([1.]))
    assert pytest.approx([5.0], network.forward([2.]))
    assert pytest.approx([17.0], network.forward([3.]))


def test_forward_arch_1_out_of_order():
    """Test that the forward pass correctly calculates the right outputs over
    three time steps for an architecture that consists of a single input,
    hidden and output node, with a recurrent connection on the hidden node.

    Nodes are not defined in topological order.
    """
    # Manually create genome for deterministic testing
    genome = Genome(key=0, config=None, innovation_store=None)
    genome.nodes = {
        0: NodeGene(0, type=NodeType.OUTPUT, activation=identity_activation),
        1: NodeGene(1, type=NodeType.INPUT, activation=identity_activation),
        2: NodeGene(2, type=NodeType.HIDDEN, activation=identity_activation),
        3: NodeGene(3, type=NodeType.BIAS, activation=identity_activation),
    }
    genome.connections = {
        4: ConnectionGene(4, node_in=1, node_out=2, weight=1., expressed=True),
        5: ConnectionGene(5, node_in=2, node_out=2, weight=2., expressed=True),
        6: ConnectionGene(6, node_in=2, node_out=0, weight=3., expressed=True),
        7: ConnectionGene(7, node_in=3, node_out=2, weight=4., expressed=True),
        6: ConnectionGene(8, node_in=3, node_out=0, weight=5., expressed=True),
    }

    network = RNN.create(genome)

    assert pytest.approx([0.0], network.forward([1.]))
    assert pytest.approx([5.0], network.forward([2.]))
    assert pytest.approx([17.0], network.forward([3.]))


def test_forward_arch_2_in_order():
    """Test that the forward pass correctly calculates the right outputs over
    three time steps for an architecture that consists of a single input,
    hidden and output node, with a recurrent connection between the output node
    and the hidden node.

    Nodes are defined in topological order.
    """
    # Manually create genome for deterministic testing
    genome = Genome(key=0, config=None, innovation_store=None)
    genome.nodes = {
        0: NodeGene(0, type=NodeType.INPUT, activation=identity_activation),
        1: NodeGene(1, type=NodeType.HIDDEN, activation=identity_activation),
        2: NodeGene(2, type=NodeType.OUTPUT, activation=identity_activation),
        3: NodeGene(3, type=NodeType.BIAS, activation=identity_activation),
    }
    genome.connections = {
        4: ConnectionGene(4, node_in=0, node_out=1, weight=1., expressed=True),
        5: ConnectionGene(5, node_in=1, node_out=2, weight=3., expressed=True),
        6: ConnectionGene(6, node_in=2, node_out=1, weight=2., expressed=True),
        7: ConnectionGene(7, node_in=3, node_out=2, weight=0.5, expressed=True),
    }

    network = RNN.create(genome)

    assert pytest.approx([0.0], network.forward([1.]))
    assert pytest.approx([0.5], network.forward([2.]))
    assert pytest.approx([0.5], network.forward([3.]))


def test_forward_arch_2_out_of_order():
    """Test that the forward pass correctly calculates the right outputs over
    three time steps for an architecture that consists of a single input,
    hidden and output node, with a recurrent connection between the output node
    and the hidden node.

    Nodes are not defined in topological order.
    """
    # Manually create genome for deterministic testing
    genome = Genome(key=0, config=None, innovation_store=None)
    genome.nodes = {
        0: NodeGene(0, type=NodeType.HIDDEN, activation=identity_activation),
        1: NodeGene(1, type=NodeType.BIAS, activation=identity_activation),
        2: NodeGene(2, type=NodeType.OUTPUT, activation=identity_activation),
        3: NodeGene(3, type=NodeType.INPUT, activation=identity_activation),
    }
    genome.connections = {
        4: ConnectionGene(4, node_in=1, node_out=2, weight=0.5, expressed=True),
        5: ConnectionGene(5, node_in=0, node_out=1, weight=3.0, expressed=True),
        6: ConnectionGene(6, node_in=2, node_out=0, weight=2.0, expressed=True),
        7: ConnectionGene(7, node_in=3, node_out=0, weight=1.0, expressed=True),
    }

    network = RNN.create(genome)

    assert pytest.approx([0.0], network.forward([1.]))
    assert pytest.approx([0.5], network.forward([2.]))
    assert pytest.approx([0.5], network.forward([3.]))
