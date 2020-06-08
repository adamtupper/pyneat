"""Utility functions for various graph operations.
"""
from collections import deque


def find_path(sources, goals, connections):
    """Try to find a path between the any of the start nodes and any of the goal
    nodes.

    Args:
        sources (list): A list of node keys that the path may start from.
        goals (list): A list of node keys that the path may finish at.
        connections (list): A list of tuples that specify the input and output
            node keys of each connection.

    Returns:
        list: A list of each node along the discovered path.
    """
    visited = set()
    expanded = set()
    queue = deque()

    for s in sources:
        queue.appendleft([s])

    while queue:
        path = queue.pop()
        head = path[-1]
        visited.add(head)

        neighbours = [o for (i, o) in connections if i == head]
        for neighbour in neighbours:
            if neighbour in goals:
                return path + [neighbour]
            elif neighbour not in visited:
                queue.appendleft(path + [neighbour])

    return []


def required_for_output(inputs, biases, outputs, connections, nodes):
    """Check to see if a node is required for computing the output of the
    network.

    A hidden node h in a NN is required if the following hold:
        a) there is a path from h to an output node
        b) there is a path from an input node to h

    Shortcuts can be taken if there is a path from h1 to h2 and h1 has been marked
    as required.

    Args:
        inputs (list): The keys of input nodes.
        biases (list): The keys of bias nodes.
        outputs (list): The keys of output nodes.
        connections (list): A list of tuples that specify the input and output
            node keys of each connection.
        nodes (list): The keys of all nodes in the network.

    Returns:
        set: The set of nodes required for computing the output of the network.
    """
    non_hidden_nodes = set(inputs + biases + outputs)
    hidden_nodes = set(nodes) - set(inputs + biases + outputs)
    required = set()

    for h in hidden_nodes:
        if h not in required:
            # if the node hasn't already marked as required
            path_to_output = find_path([h], outputs + list(required), connections)
            path_from_input = find_path(inputs + biases, [h] + list(required), connections)

            if path_to_output and path_from_input:
                # add hidden node and other hidden nodes along the path found
                for node in path_from_input + path_to_output:
                    if node not in non_hidden_nodes:
                        required.add(node)

    return required.union(non_hidden_nodes)


def creates_cycle(connections, test):
    """Checks to see if adding the test connection to the network would create
    a cycle.

    Args:
        connections (list): A list of (node in key, node out key) pairs for each
            connection.
        test (tuple): A tuple of the form (node in key, node out key) that
            represents the connection trying to be added that needs to be
            tested.

    Returns:
        bool: True if 'test' creates a cycle, False otherwise.
    """
    node_in, node_out = test

    if node_in == node_out:
        # Self-loop
        return True

    visited = {node_out}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == node_in:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def group_layers(inputs, outputs, biases, connections, nodes):
    """Group nodes together into layers that can be evaluated in parallel by a
     feed-forward neural network.

    i.e. nodes in the same layer are independent conditional on the nodes in
    previous layers.

    Args:
        inputs (list): The keys of the input nodes.
        outputs (list): The keys of the output nodes.
        biases (list): The keys of the bias nodes.
        connections (list): A list of (node in key, node out key) pairs for each
            expressed connection.
        nodes (set): The set of nodes required for calculating the output value.

    Returns:
        list: A list of sets that contain the node kets for the nodes in each
            layer.
    """
    layers = []
    s = set(inputs + biases)
    while True:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if (a in s) and (b not in s))
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers
