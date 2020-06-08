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

    def test_xor_highly_redundant_genome(self):
        """Check to see that the correct evaluation is performed for an XOR
         genome that has very high redundancy.
        """
        genome = Genome(0, self.config.genome_config, InnovationStore())

        genome.nodes = {
            0: NodeGene(0, NodeType.INPUT, None),
            1: NodeGene(1, NodeType.INPUT, None),
            2: NodeGene(2, NodeType.OUTPUT, steep_sigmoid_activation),
            3: NodeGene(3, NodeType.BIAS, None),
            10: NodeGene(10, NodeType.HIDDEN, steep_sigmoid_activation),
            33: NodeGene(33, NodeType.HIDDEN, steep_sigmoid_activation),
            44: NodeGene(44, NodeType.HIDDEN, steep_sigmoid_activation),
            77: NodeGene(77, NodeType.HIDDEN, steep_sigmoid_activation),
        }
        genome.connections = {
            4: ConnectionGene(4, node_in=0, node_out=2, weight=-2.25, expressed=False),
            5: ConnectionGene(5, node_in=1, node_out=2, weight=-2.20, expressed=False),
            6: ConnectionGene(6, node_in=3, node_out=2, weight=-3.45, expressed=True),
            11: ConnectionGene(11, node_in=1, node_out=10, weight=1.38, expressed=False),
            12: ConnectionGene(12, node_in=10, node_out=2, weight=7.61, expressed=True),
            18: ConnectionGene(18, node_in=3, node_out=10, weight=-2.19, expressed=True),
            34: ConnectionGene(34, node_in=1, node_out=33, weight=1.69, expressed=True),
            35: ConnectionGene(35, node_in=33, node_out=10, weight=3.9, expressed=False),
            36: ConnectionGene(36, node_in=0, node_out=33, weight=0.5, expressed=True),
            45: ConnectionGene(45, node_in=33, node_out=44, weight=-2.01, expressed=False),
            46: ConnectionGene(46, node_in=44, node_out=10, weight=0.38, expressed=False),
            78: ConnectionGene(78, node_in=33, node_out=77, weight=-0.33, expressed=False),
            79: ConnectionGene(79, node_in=77, node_out=44, weight=0.87, expressed=True)
        }

        genome.inputs = [0, 1]
        genome.outputs = [2]
        genome.biases = [3]

        network = NN.create(genome)

        # 2-input XOR inputs
        xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

        # Inputs are disconnected from the outputs so the output should be constant
        expected_output = steep_sigmoid_activation(7.61 * steep_sigmoid_activation(1 * -2.19) + 1 * -3.45)
        for xi in xor_inputs:
            output = network.forward(xi)
            assert output[0] == pytest.approx(expected_output)
