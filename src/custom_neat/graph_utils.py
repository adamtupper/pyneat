"""Utility functions for various graph operations.
"""


def required_for_output(input_nodes, output_nodes, bias_nodes, connections):
    """Returns the set of nodes that are required to calculate the output when
    the graph is interpreted as a neural network.

    The input, output and bias nodes are always included in the set of required
    nodes.

    TODO: Complete function docstring.

    Args:
        input_nodes (:list:`int`): The node IDs of all input nodes.
        output_nodes (:list:`int`): The node IDs of all output nodes.
        bias_nodes (:list:`int`): The node IDs of all bias nodes.
        connections (:list:`ConnectionGene`): The connection genes of the
            genome.

    Returns:
        set: The IDs of all required nodes for generating an RNN from the
            genome.
    """
    required = set(output_nodes)
    updated = True

    while updated:
        new_nodes = {c.node_in for c in connections
                     if c.node_out in required and c.node_in not in required and c.expressed}

        if new_nodes:
            required = required.union(new_nodes)
            updated = True
        else:
            updated = False

    required = required.union(set(input_nodes)).union(set(bias_nodes))

    return required
