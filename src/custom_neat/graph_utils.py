"""Utility functions for various graph operations.
"""


def required_for_output(input_nodes, output_nodes, connections):
    """Returns the set of nodes that are required to calculate the output when
    the graph is interpreted as a neural network.

    The input and output nodes are always included in the set of required nodes.

    TODO: Complete function docstring.

    Args:
        input_nodes (list):
        output_nodes (list):
        connections (list):

    Returns:

    """
    required = set(output_nodes)
    updated = True

    while updated:
        new_nodes = {c.in_node for c in connections
                     if c.out_node in required and c.in_node not in required and c.expressed}

        if new_nodes:
            required = required.union(new_nodes)
            updated = True
        else:
            updated = False

    required = required.union(set(input_nodes))

    return required
