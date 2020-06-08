"""Builds a feed-forward neural network from a genome.
"""

from pyneat.graph_utils import required_for_output, group_layers


class NN:
    """Feed-forward neural network (NN) implementation.

    TODO: Decide which attributes should be private (probably all).

    Attributes:
        input_nodes (:list:`int`): The node IDs of all input nodes.
        output_nodes (:list:`int`): The node IDs of all output nodes.
        bias_nodes (:list:`int`): The node IDs of all bias nodes.
        node_evals (:list:`tuple`): The information required to evaluate each
            node (e.g. activation fn, bias, node inputs).
        values (dict): Node key, value pairs holding the current activation
            value of each node.
    """

    def __init__(self, input_nodes, output_nodes, bias_nodes, node_evals):
        """Create a new NN object.

        Args:
            input_nodes (:list:`int`): The node IDs of all input nodes.
            output_nodes (:list:`int`): The node IDs of all output nodes.
            bias_nodes (:list:`int`): The node IDs of all bias nodes.
            node_evals (:list:`tuple`): The information required to evaluate
                each node (e.g. node key, activation fn, node inputs).
        """
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.bias_nodes = bias_nodes
        self.node_evals = node_evals

        # Store the activations for each node (bias node activations are always 1)
        self.values = {k: 1.0 for k in self.bias_nodes}

    def forward(self, inputs):
        """Calculate the new output values given the inputs to the network.

        Args:
            inputs (:list:`float`): The input values for each input node of the
                network.

        Returns:
            :list:`float`: The output values for each output node of the
                network.
        """
        # Update input nodes with latest inputs
        for node_key, value in zip(self.input_nodes, inputs):
            self.values[node_key] = value

        # Calculate outputs
        for node_key, activation, node_inputs in self.node_evals:
            weighted_inputs = [self.values[k] * w for k, w in node_inputs]
            self.values[node_key] = activation(sum(weighted_inputs))
        outputs = [self.values[k] for k in self.output_nodes]

        return outputs

    @staticmethod
    def create(genome):
        """Create a new feed-forward NN object from the provided genome.

        Args:
            genome (Genome): The genome that encodes the NN to be built.

        Returns:
            NN: The NN encoded by the genome.
        """
        # Fetch indices of network input and output nodes
        input_nodes = genome.inputs
        output_nodes = genome.outputs
        bias_nodes = genome.biases

        required_connections = [(g.node_in, g.node_out) for g in genome.connections.values() if g.expressed]
        nodes = [k for k in genome.nodes.keys()]
        required_nodes = required_for_output(input_nodes, bias_nodes,
                                             output_nodes, required_connections,
                                             nodes)

        # Gather information required to evaluate each node
        layers = group_layers(input_nodes, output_nodes, bias_nodes,
                              required_connections, required_nodes)
        node_evals = []
        for layer in layers:
            for node_key in layer:
                inputs = []
                for conn_gene in genome.connections.values():
                    if conn_gene.expressed and conn_gene.node_out == node_key:
                        inputs.append((conn_gene.node_in, conn_gene.weight))

                node_gene = genome.nodes[node_key]
                node_evals.append((node_key, node_gene.activation, inputs))

        return NN(input_nodes, output_nodes, bias_nodes, node_evals)
