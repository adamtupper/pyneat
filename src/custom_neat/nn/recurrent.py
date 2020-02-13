"""Builds a recurrent neural network from a genome.
"""

from custom_neat.genome import *
from custom_neat.graph_utils import required_for_output


class RNN:
    """Time-delayed recurrent neural network (RNN) implementation.

    TODO: Decide which attributes should be private (probably all).

    Attributes:
        input_nodes (:list:`int`): The node IDs of all input nodes.
        output_nodes (:list:`int`): The node IDs of all output nodes.
        node_evals (:list:`tuple`): The information required to evaluate each
            node (e.g. activation fn, bias, node inputs).
        prev_states (dict): A dictionary of node ID, value pairs that stores the
            output value for each node from the previous time step.
        curr_states (dict): A dictionary of node ID, value pairs that stores the
            output value for each node at the current time step.
    """

    def __init__(self, input_nodes, output_nodes, node_evals):
        """Create a new RNN object.

        Args:
            input_nodes (:list:`int`): The node IDs of all input nodes.
            output_nodes (:list:`int`): The node IDs of all output nodes.
            node_evals (:list:`tuple`): The information required to evaluate
            each node (e.g. activation fn, bias, node inputs).
        """
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.node_evals = node_evals

        # Store the previous and current states for each node
        self.prev_states = {}
        self.curr_states = {}

        # Initialise the previous and current states for each node
        for node_idx in self.input_nodes + self.output_nodes:
            self.prev_states[node_idx] = 0.0
            self.curr_states[node_idx] = 0.0

        for node_idx, node_bias, node_activation, node_inputs in self.node_evals:
            self.prev_states[node_idx] = 0.0
            self.curr_states[node_idx] = 0.0
            for in_node_idx, _ in node_inputs:
                self.prev_states[in_node_idx] = 0.0
                self.curr_states[in_node_idx] = 0.0

    def reset(self):
        """Reset the network.

        Set all previous and current states to zero.
        """
        self.prev_states = {k: 0.0 for k, _ in self.prev_states.items()}
        self.curr_states = {k: 0.0 for k, _ in self.curr_states.items()}

    def forward(self, inputs):
        """Calculate the new output values for the current time step, given the
        new inputs to the network.

        Args:
            inputs (:list:`float`): The input values for each input node of the
                network.

        Returns:
            :list:`float`: The output values for each output node of the
                network.
        """
        # Update input nodes with latest inputs
        for node_idx, value in zip(self.input_nodes, inputs):
            self.curr_states[node_idx] = value
            # self.prev_states[node_idx] = value

        # Propagate input values through the network
        for node_idx, bias, activation, node_inputs in self.node_evals:
            weighted_inputs = [self.prev_states[in_node_idx] * weight for in_node_idx, weight in node_inputs]
            self.curr_states[node_idx] = activation(bias + sum(weighted_inputs))

        self.prev_states = {node: val for (node, val) in self.curr_states.items()}

        return [self.curr_states[i] for i in self.output_nodes]

    @staticmethod
    def create(genome):
        """Create a new RNN object from the provided genome.

        Args:
            genome (Genome): The genome that encodes the RNN to be built.

        Returns:
            RNN: The RNN encoded by the genome.
        """
        # Fetch indices of network input and output nodes
        input_nodes = genome.inputs
        output_nodes = genome.outputs

        # Find all required nodes for computing the network outputs
        required = required_for_output(input_nodes, output_nodes, genome.connections.values())

        # Build a dict of all of the inputs for each node
        node_inputs = {}
        for gene in genome.connections.values():
            if not gene.expressed:
                # Skip inactive connections
                continue

            if gene.in_node not in required or gene.out_node not in required:
                # Skip connections to and from non-required nodes
                continue

            if gene.out_node not in node_inputs:
                node_inputs[gene.out_node] = [(gene.in_node, gene.weight)]
            else:
                node_inputs[gene.out_node].append((gene.in_node, gene.weight))

        # Gather information required to evaluate each node
        node_evals = []
        for node_idx, inputs in node_inputs.items():
            node_gene = genome.nodes[node_idx]
            node_evals.append((node_idx, node_gene.bias, node_gene.activation, inputs))

        return RNN(input_nodes, output_nodes, node_evals)
