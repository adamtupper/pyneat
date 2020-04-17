"""Tests for the graph utils module.
"""

import pytest

from custom_neat.graph_utils import *
from custom_neat.genome import *


def test_required_for_output_no_connections():
    """Test that input, output and bias nodes are always identified as required.
    """
    input_nodes = [0]
    output_nodes = [1]
    bias_nodes = [2]
    connection_genes = []

    required = required_for_output(input_nodes, output_nodes, bias_nodes, connection_genes)
    assert len(required) == 3
    assert required == {0, 1, 2}


def test_required_for_output_single_redundant_hidden():
    """Test that a redundant hidden node is excluded from required.
    """
    input_nodes = [0, 1]
    output_nodes = [4, 5]
    bias_nodes = [11]
    connection_genes = [
        ConnectionGene(6, 0, 2, 0.0, True),
        ConnectionGene(7, 0, 3, 0.0, True),
        ConnectionGene(8, 1, 3, 0.0, True),
        ConnectionGene(9, 2, 4, 0.0, True),
        ConnectionGene(10, 2, 5, 0.0, True),
        ConnectionGene(12, 11, 4, 0.0, True),
        ConnectionGene(12, 11, 5, 0.0, True),
        ConnectionGene(12, 11, 3, 0.0, True),
    ]

    required = required_for_output(input_nodes, output_nodes, bias_nodes,
                                   connection_genes)
    assert len(required) == 6
    assert required == {0, 1, 2, 4, 5, 11}


def test_required_for_output_multilayer_recurrent():
    """Test that the reuired_for_output function works for multi-layer,
    recurrent networks.
    """
    input_nodes = [0, 1]
    output_nodes = [5, 6, 7]
    bias_nodes = [14]
    connection_genes = [
        ConnectionGene(8, 0, 2, 0.0, True),
        ConnectionGene(9, 0, 3, 0.0, True),
        ConnectionGene(10, 0, 4, 0.0, True),
        ConnectionGene(11, 2, 2, 0.0, True),
        ConnectionGene(12, 2, 6, 0.0, True),
        ConnectionGene(13, 7, 4, 0.0, True),
        ConnectionGene(15, 14, 5, 0.0, True),
        ConnectionGene(16, 14, 6, 0.0, True),
        ConnectionGene(17, 14, 7, 0.0, True),
        ConnectionGene(18, 14, 3, 0.0, True),
    ]

    required = required_for_output(input_nodes, output_nodes, bias_nodes, connection_genes)
    assert len(required) == 7
    assert required == {0, 1, 2, 5, 6, 7, 14}


def test_required_for_output_simple_recurrent():
    """Test that for a simple three neuron RNN that all nodes are marked as
    required."""
    input_nodes = [0]
    bias_nodes = [3]
    output_nodes = [2]
    connection_genes = [
        ConnectionGene(4, node_in=0, node_out=1, weight=1., expressed=True),
        ConnectionGene(5, node_in=1, node_out=1, weight=2., expressed=True),
        ConnectionGene(6, node_in=1, node_out=2, weight=3., expressed=True),
        ConnectionGene(7, node_in=3, node_out=1, weight=4., expressed=True),
        ConnectionGene(8, node_in=3, node_out=2, weight=5., expressed=True),
    ]

    required = required_for_output(input_nodes, output_nodes, bias_nodes, connection_genes)
    assert len(required) == 4
    assert required == {0, 1, 2, 3}
