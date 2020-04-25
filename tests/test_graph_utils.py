"""Tests for the graph utils module.
"""

import pytest

from pyneat.graph_utils import *
from pyneat.genome import *


def test_is_required_feed_forward_simple():
    nodes = [0, 1, 2, 3]
    connections = [(0, 1), (1, 2), (3, 2)]
    inputs = [0]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_recurrent_simple():
    nodes = [0, 1, 2, 3]
    connections = [(0, 1), (1, 2), (3, 2), (1, 1)]
    inputs = [0]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_feed_forward_complex_no_hidden_1():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(0, 21), (3, 21), (3, 2), (10, 2)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_feed_forward_complex_no_hidden_2():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(10, 2), (3, 2), (21, 10)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_feed_forward_complex_hidden_1():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(0, 21), (3, 21), (3, 2), (10, 2), (21, 10)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3, 10, 21}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_feed_forward_complex_hidden_2():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(0, 21), (3, 21), (3, 2), (10, 2), (0, 10)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3, 10}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_recurrent_complex_no_hidden_1():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(0, 21), (3, 21), (3, 2), (10, 2), (2, 2), (2, 21)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_recurrent_complex_no_hidden_2():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(10, 2), (3, 2), (21, 10), (10, 10)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_recurrent_complex_hidden_1():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(0, 21), (3, 21), (3, 2), (10, 2),
                   (21, 10), (2, 2), (10, 21)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3, 10, 21}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_recurrent_complex_hidden_2():
    nodes = [0, 1, 2, 3, 10, 21]
    connections = [(0, 21), (3, 21), (3, 2), (10, 2),
                   (0, 10), (21, 21), (10, 10), (2, 2)]
    inputs = [0, 1]
    biases = [3]
    outputs = [2]

    expected = {0, 1, 2, 3, 10}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_feed_forward_bias_to_hidden():
    nodes = [0, 1, 2, 3]
    connections = [(0, 2), (3, 2), (1, 3)]
    inputs = [0]
    biases = [1]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_recurrent_bias_to_hidden_1():
    nodes = [0, 1, 2, 3]
    connections = [(0, 2), (3, 2), (1, 3), (2, 3)]
    inputs = [0]
    biases = [1]
    outputs = [2]

    expected = {0, 1, 2, 3}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_is_required_recurrent_bias_to_hidden_2():
    nodes = [0, 1, 2, 3]
    connections = [(0, 2), (1, 3), (2, 3)]
    inputs = [0]
    biases = [1]
    outputs = [2]

    expected = {0, 1, 2}
    actual = required_for_output(inputs, biases, outputs, connections, nodes)

    assert actual == expected


def test_required_for_output_no_connections():
    """Test that input, output and bias nodes are always identified as required.
    """
    input_nodes = [0]
    output_nodes = [1]
    bias_nodes = [2]

    connections = []
    nodes = [0, 1, 2]

    required = required_for_output(input_nodes, bias_nodes, output_nodes, connections, nodes)
    assert len(required) == 3
    assert required == {0, 1, 2}


def test_required_for_output_single_redundant_hidden():
    """Test that a redundant hidden node is excluded from required.
    """
    input_nodes = [0, 1]
    output_nodes = [4, 5]
    bias_nodes = [11]
    connections = [(0, 2), (0, 3), (1, 3), (2, 4),
                   (2, 5), (11, 4), (11, 5), (11, 3)]
    nodes = [0, 1, 2, 3, 4, 5, 11]

    required = required_for_output(input_nodes, bias_nodes, output_nodes,
                                   connections, nodes)
    assert len(required) == 6
    assert required == {0, 1, 2, 4, 5, 11}


def test_required_for_output_multilayer_recurrent():
    """Test that the required_for_output function works for multi-layer,
    recurrent networks.
    """
    input_nodes = [0, 1]
    output_nodes = [5, 6, 7]
    bias_nodes = [14]
    connections = [(0, 2), (0, 3), (0, 4), (2, 2), (2, 6), (7, 4), (14, 5),
                   (14, 6), (14, 7), (14, 3)]
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 14]

    required = required_for_output(input_nodes, bias_nodes, output_nodes, connections, nodes)
    assert len(required) == 7
    assert required == {0, 1, 2, 5, 6, 7, 14}


def test_required_for_output_simple_recurrent():
    """Test that for a simple three neuron RNN that all nodes are marked as
    required."""
    input_nodes = [0]
    bias_nodes = [3]
    output_nodes = [2]
    connections = [(0, 1), (1, 1), (1, 2), (3, 1), (3, 2)]
    nodes = [0, 1, 2, 3]

    required = required_for_output(input_nodes, bias_nodes, output_nodes, connections, nodes)
    assert len(required) == 4
    assert required == {0, 1, 2, 3}


def test_required_for_output_complex_feed_forward():
    """Test that the function works for a relatively complex feed-forward
    network.
    """
    input_nodes = [0, 1]
    bias_nodes = [3]
    output_nodes = [2]
    connections = [(3, 2), (10, 2), (0, 21), (3, 21)]
    nodes = [0, 1, 2, 3, 10, 21]

    required = required_for_output(input_nodes, bias_nodes, output_nodes, connections, nodes)
    assert len(required) == 4
    assert required == {0, 1, 2, 3}
