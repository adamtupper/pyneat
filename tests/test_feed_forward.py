"""Tests for the feed-forward neural network module.
"""

import os
import pytest

import neat

from pyneat.genome import *
from pyneat.reproduction import Reproduction
from pyneat.species import SpeciesSet
from pyneat.innovation import InnovationStore
from pyneat.stagnation import Stagnation

from pyneat.nn.feed_forward import NN
from pyneat.activations import steep_sigmoid_activation


class TestFeedForward:
    def setup_method(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'configurations',
                                   'test_configuration.ini')
        self.config = neat.Config(Genome,
                                  Reproduction,
                                  SpeciesSet,
                                  Stagnation,
                                  config_path)

    @staticmethod
    def xor_output(x1, x2):
        h1 = steep_sigmoid_activation(-0.5 + (x1 + -x2))
        h2 = steep_sigmoid_activation(-0.5 + (-x1 + x2))
        o1 = steep_sigmoid_activation(-0.5 + (h1 + h2))

        return o1

    def test_xor(self):
        """Check to see that a correct XOR network is evaluated correctly.
        """
        genome = Genome(0, self.config.genome_config, InnovationStore())

        genome.add_node(-1, -1, NodeType.INPUT)  # Node 0
        genome.add_node(-3, -3, NodeType.INPUT)  # Node 1
        genome.add_node(-2, -2, NodeType.OUTPUT)  # Node 2
        genome.add_bias_node(0)  # Node 3

        genome.add_node(0, 2, NodeType.HIDDEN)  # Node 4
        genome.add_node(1, 2, NodeType.HIDDEN)  # Node 5

        # Connect inputs to hiddens
        genome.add_connection(node_in=0, node_out=4, weight=1.0, expressed=True)
        genome.add_connection(node_in=0, node_out=5, weight=-1.0, expressed=True)
        genome.add_connection(node_in=1, node_out=4, weight=-1.0, expressed=True)
        genome.add_connection(node_in=1, node_out=5, weight=1.0, expressed=True)

        # Connect hiddens to outputs
        genome.add_connection(node_in=4, node_out=2, weight=1.0, expressed=True)
        genome.add_connection(node_in=5, node_out=2, weight=1.0, expressed=True)

        # Add bias connections
        genome.add_connection(node_in=3, node_out=4, weight=-0.5, expressed=True)
        genome.add_connection(node_in=3, node_out=5, weight=-0.5, expressed=True)
        genome.add_connection(node_in=3, node_out=2, weight=-0.5, expressed=True)

        network = NN.create(genome)

        for inputs in ([0, 0], [0, 1], [1, 0], [1, 1]):
            x1, x2 = inputs
            assert network.forward(inputs)[0] == pytest.approx(TestFeedForward.xor_output(x1, x2))
