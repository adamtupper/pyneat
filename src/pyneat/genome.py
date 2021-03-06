"""This module defines the genome encoding used by NEAT.
"""
import random
import copy
from enum import Enum

from neat.config import ConfigParameter, write_pretty_params

from pyneat.activations import ActivationFunctionSet
from pyneat.innovation import InnovationType
from pyneat.graph_utils import creates_cycle


class NodeType(Enum):
    """Define the types for nodes in the network.
    """
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
    BIAS = 3


class ConnectionGene:
    """Defines a connection gene used in the genome encoding.

    Attributes:
        key (int): The innovation key for this gene.
        node_in (int): The key of the node this connection from.
        node_out (int): The key of the node this connection is to.
        weight (float): The connection weight.
        expressed (bool): True if the connection is expressed (enabled) in the
            phenotype, False otherwise.
    """

    def __init__(self, key, node_in, node_out, weight, expressed):
        """Creates a new ConnectionGene object.

        Args:
            key (int): The innovation key for this gene.
            node_in (int): The key of the node this connection from/the node
                that leads into this connection.
            node_out (int): The key of the node this connection is to/the node
                that this connection leads into.
            weight (float): The connection weight.
            expressed (bool): True if the connection is expressed (enabled) in the
                phenotype, False otherwise.
        """
        self.key = key
        self.node_in = node_in
        self.node_out = node_out
        self.weight = weight
        self.expressed = expressed

    def __eq__(self, other):
        """Test for equality against another connection gene.

        Args:
            other (ConnectionGene): The connection gene to compare.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        return (self.key, self.node_in, self.node_out, self.weight, self.expressed) == \
               (other.key, other.node_in, other.node_out, other.weight, other.expressed)

    def copy(self):
        """Create a copy of the connection gene.

        Returns:
            ConnectionGene: A copy of itself.
        """
        return copy.copy(self)


class NodeGene:
    """Defines a node gene used in the genome encoding.

    Attributes:
        key (int): The innovation key (also the node key) for this gene.
        type (NodeType): The type of the node (either input, output or hidden).
        activation (function): The node activation function. Note that input and
            bias nodes should not have an activation function (i.e. it is the
            identity function).
    """

    def __init__(self, key, type, activation):
        """Creates a new NodeGene object.

        Args:
            key (int): The innovation key (also the node key) for this gene.
            type (NodeType): The type of the node (either input, output or hidden).
            activation (function): The node activation function.
        """
        self.key = key
        self.type = type
        self.activation = activation

    def __eq__(self, other):
        """Test for equality against another node gene.

        Args:
            other (NodeGene): The node gene to compare.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        return (self.key, self.type, self.activation) == \
               (other.key, other.type, other.activation)

    def copy(self):
        """Create a copy of the node gene.

        Returns:
            NodeGene: A copy of itself.
        """
        return copy.copy(self)


class GenomeConfig:
    """Sets up and holds configuration information for the Genome class.

    Config Parameters:
        **num_inputs (int):** The number of inputs each network should have.

        **num_outputs (int):** The number of outputs each network should have.

        **num_biases (int):** The number of bias nodes the network should have.

        **initial_conn_prob (float):** The initial connection probability of each
        potential connection between inputs and outputs. 0.0 = no connections,
        i.e. all inputs are disconnected from the outputs. 1.0 = fully
        connected, i.e. all inputs are connected to all outputs.

        **activation_func (str):** The name of the activation function to be used by
        hidden and output nodes. Must be present in the set of possible
        activation functions.

        **compatibility_disjoint_coefficient (float):** The disjoint and excess
        coefficient to be used when calculating genome distance.

        **compatibility_weight_coefficient (float):** The weight and bias
        coefficient to be used when calculation genome distance.

        **normalise_gene_dist (bool):** Whether or not normalise the gene dist (for
        genetic distance calculations) for large genomes.

        **feed_forward (bool):** False if recurrent connections are allowed, True
        otherwise.

        **conn_add_prob (float):** The probability of adding a new connection when
        performing mutations.

        **node_add_prob (float):** The probability of adding a new node when
        performing mutations.

        **weight_mutate_prob (float):** The probability of mutating the connection
        weights of a genome when performing mutations.

        **weight_replace_prob (float):** The probability of replacing, instead of
        perturbing, a connection weight when performing weight mutations.

        **weight_init_power (float):** Sets the range of possible values for weight
        replacements and new weight initialisations.

        **weight_perturb_power (float):** Sets the range of possible values for
        weight perturbations.

        **weight_min_value (float):** Sets the minimum allowed value for connection
        weights.

        **weight_max_value (float):** Sets the maximum allowed value for connection
        weights.

        **gene_disable_prob (float):** The probability of disabling a gene in the
        child that is disabled in either of the parents when performing
        crossover.
    """

    def __init__(self, params):
        """Creates a new GenomeConfig object.

        Args:
            params (dict): A dictionary of config parameters and values.
        """
        # Create full set of available activation functions
        self.activation_defs = ActivationFunctionSet()

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_biases', int),
                        ConfigParameter('initial_conn_prob', float),
                        ConfigParameter('activation_func', str),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('normalise_gene_dist', bool),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('weight_mutate_prob', float),
                        ConfigParameter('weight_replace_prob', float),
                        ConfigParameter('weight_init_power', float),
                        ConfigParameter('weight_perturb_power', float),
                        ConfigParameter('weight_min_value', float),
                        ConfigParameter('weight_max_value', float),
                        ConfigParameter('gene_disable_prob', float)]

        # Use the configuration data to interpret the supplied parameters
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

    def save(self, filename):
        """Save the genome configuration.

        Args:
            filename (str): The filename to write the configuration to.
        """
        write_pretty_params(filename, self, self._params)


class Genome:
    """Defines a genome used to encode a neural network.

    Attributes:
        key (int): A unique identifier for the genome.
        config (GenomeConfig): The genome configuration settings.
        fitness (float): The fitness of the genome.
        nodes (dict): A dictionary of node key (int), node gene pairs.
        connections (dict): A dictionary of connection gene key, connection gene
            pairs.
        inputs (:list:`int`): The node keys of input nodes.
        outputs (:list:`int`): The node keys of output nodes.
        biases (:list:`int`): The node keys of bias nodes.
        innovation_store (InnovationStore): The global innovation store used for
            tracking new structural mutations.
    """
    @classmethod
    def parse_config(cls, param_dict):
        """Takes a dictionary of configuration items, returns an object that
        will later be passed to the write_config method.

        Args:
            param_dict (dict): A dictionary of configuration parameter values.

        Returns:
            GenomeConfig: The genome configuration.
        """
        return GenomeConfig(param_dict)

    @classmethod
    def write_config(cls, filename, config):
        """Takes a file-like object and the configuration object created by
        parse_config. This method should write the configuration item
        definitions to the given file.

        Args:
            filename (str): The name of the file to write the genome configuration to.
            config (GenomeConfig): The genome configuration to save.
        """
        config.save(filename)

    def __init__(self, key, config, innovation_store):
        """Creates a new Genome object.

        TODO: Write new test for when no input/output nodes are specified.

        Args:
            key (int): A unique identifier for the genome.
            config (GenomeConfig): The genome configuration settings.
            innovation_store (InnovationStore): The global innovation store used
                for tracking new structural mutations.
        """
        self.key = key
        self.config = config
        self.innovation_store = innovation_store
        self.fitness = None

        # (gene key, gene) pairs for genes
        self.nodes = {}
        self.connections = {}

        # Store the keys for input, bias and output node genes
        self.inputs = []
        self.outputs = []
        self.biases = []

    def configure_new(self):
        """Configure a new genome based on the given configuration.

        The initial inputs and outputs for input and output nodes are specified
        as negatives so that matching innovation keys are generated for
        corresponding input and output nodes between genomes. Inputs nodes use
        odd negative numbers, and output nodes use even negative numbers.
        """
        # Create the required number of input nodes
        for i in range(-1, -2 * self.config.num_inputs - 1, -2):
            self.add_node(i, i, NodeType.INPUT)

        # Create the required number of output nodes
        for i in range(-2, -2 * self.config.num_outputs - 1, -2):
            self.add_node(i, i, NodeType.OUTPUT)

        # Create the required number of bias nodes
        for i in range(self.config.num_biases):
            self.add_bias_node(i)

        # Add a hidden node (only for debugging, to match Stanley et al.'s evolved DPNV solution)
        # self.add_node(1, 3, NodeType.HIDDEN)

        # Add initial connections
        for node_in in self.inputs:
            for node_out in self.outputs:
                if random.random() < self.config.initial_conn_prob:
                    weight = random.uniform(-1.0, 1.0) * self.config.weight_init_power
                    self.add_connection(node_in, node_out, weight)

        for node_in in self.biases:
            for node_out in self.outputs:
                # TODO: Should bias nodes be probabilistically connected?
                weight = random.uniform(-1.0, 1.0) * self.config.weight_init_power
                self.add_connection(node_in, node_out, weight)

        # Add extra connections (only for debugging, to match Stanley et al.'s evolved DPNV solution)
        # self.add_connection(1, 5, random.uniform(-1.0, 1.0) * self.config.weight_init_power)
        # self.add_connection(2, 5, random.uniform(-1.0, 1.0) * self.config.weight_init_power)
        # self.add_connection(5, 3, random.uniform(-1.0, 1.0) * self.config.weight_init_power)
        # self.add_connection(5, 5, random.uniform(-1.0, 1.0) * self.config.weight_init_power)
        # self.add_connection(3, 3, random.uniform(-1.0, 1.0) * self.config.weight_init_power)

    def __eq__(self, other):
        """Check for genome equality.

        Args:
            other (Genome): The genome to compare itself to.

        Returns:
            bool: True if this genome is equal to the other, False otherwise.
        """
        self_attrs = (self.key, self.nodes, self.connections, self.inputs, self.outputs, self.biases)
        other_attrs = (other.key, other.nodes, other.connections, other.inputs, other.outputs, other.biases)

        return self_attrs == other_attrs

    def copy(self):
        """Create a copy of the genome.

        Note: Copies share the same config and innovation store.

        Returns:
            Genome: A copy of itself, but with the same config and innovation
                store.
        """
        new_copy = Genome(self.key, self.config, self.innovation_store)
        new_copy.fitness = self.fitness
        new_copy.inputs = self.inputs.copy()
        new_copy.outputs = self.outputs.copy()
        new_copy.biases = self.biases.copy()
        new_copy.nodes = {k: g.copy() for k, g in self.nodes.items()}
        new_copy.connections = {k: g.copy() for k, g in self.connections.items()}

        return new_copy

    def add_node(self, node_in, node_out, node_type):
        """Add a new node positioned between two other nodes. Input and output
        nodes are positioned between non-existent nodes.

        Args:
            node_in (int): The key of the node that precedes this new node.
            node_out (int): The key of the node that succeeds this new node.
            node_type (NodeType): The type of node to be added.

        Returns:
            int: The key of the new node
        """
        key = self.innovation_store.get_innovation_key(node_in, node_out, InnovationType.NEW_NODE)
        assert key not in self.nodes

        activation = None if node_type in [NodeType.INPUT, NodeType.BIAS] else self.config.activation_defs.get(self.config.activation_func)

        self.nodes[key] = NodeGene(
            key=key,
            type=node_type,
            activation=activation
        )

        if node_type == NodeType.INPUT:
            self.inputs.append(key)
        elif node_type == NodeType.OUTPUT:
            self.outputs.append(key)

        return key

    def add_bias_node(self, num):
        """Add a new bias node.

        Args:
            num (int): A number that can uniquely identify bias nodes in the
                innovation store.
        """
        key = self.innovation_store.get_innovation_key(num, num, InnovationType.NEW_BIAS)
        assert key not in self.nodes

        self.nodes[key] = NodeGene(
            key=key,
            type=NodeType.BIAS,
            activation=None
        )
        self.biases.append(key)

    def add_connection(self, node_in, node_out, weight, expressed=True):
        """Add a connection between two nodes.

        Args:
            node_in (int): The key of the node that leads into the new
                connection.
            node_out (int): The key of the node that the the new connection
                leads into.
            weight (float): The weight of the connection. Must be a value
                between [0, 1].
            expressed (bool): True if the connection should be expressed in the
                phenotype, False otherwise.
        """
        key = self.innovation_store.get_innovation_key(node_in, node_out, InnovationType.NEW_CONNECTION)
        assert key not in self.connections

        new_connection_gene = ConnectionGene(
            key=key,
            node_in=node_in,
            node_out=node_out,
            weight=weight,
            expressed=expressed
        )
        self.connections[key] = new_connection_gene

    def mutate(self):
        """Mutate the genome.

        Mutates the genome according to the mutation parameter values specified
        in the genome configuration.

        As per the original implementation of NEAT:

        - If any structural mutations are performed, weight and bias mutations
          will not be performed.
        - If an add node mutation is performed, an add connection mutation will
          not also be performed.
        """
        connection_added = False
        node_added = False

        if random.random() < self.config.node_add_prob:
            node_added = self.mutate_add_node()
        if random.random() < self.config.conn_add_prob:
            connection_added = self.mutate_add_connection()

        if not (connection_added or node_added):
            if random.random() < self.config.weight_mutate_prob:
                self.mutate_weights()

            # TODO: Are there any other non-structural mutations?

    def mutate_add_connection(self):
        """Performs an 'add connection' structural mutation.
        
        A single connection with a random weight is added between two previously
        unconnected nodes.

        Returns:
            bool: True if a connection was added, False otherwise.
        """
        possible_inputs = [k for k, g in self.nodes.items()]
        possible_outputs = [k for k, g in self.nodes.items() if g.type not in [NodeType.INPUT,NodeType.BIAS]]

        max_retries = 20
        attempts = 0
        while attempts < max_retries:
            node_in = random.choice(possible_inputs)
            node_out = random.choice(possible_outputs)

            connections = [(g.node_in, g.node_out) for g in self.connections.values()]
            if self.config.feed_forward and creates_cycle(connections, (node_in, node_out)):
                attempts += 1
                continue

            # Check for existing connection
            mutation = (node_in, node_out, InnovationType.NEW_CONNECTION)
            mutation_key = self.innovation_store.mutation_to_key.get(mutation)
            connection_gene = self.connections.get(mutation_key, None)

            if connection_gene and not connection_gene.expressed:
                # Enable if disabled
                self.connections[mutation_key].expressed = True
                return True

            elif not connection_gene:
                # Add a new connection
                connection_weight = random.uniform(-1.0, 1.0) * self.config.weight_perturb_power
                self.add_connection(node_in, node_out, connection_weight)
                return True

            attempts += 1  # Failed to find a spot to add/enable a connection, try again.

        return False

    def mutate_add_node(self):
        """Performs an 'add node' structural mutation.

        An existing connection is split and the new node is placed where the old
        connection used to be. The old connection is disabled and two new
        connection genes are added. The new connection leading into the new node
        receives a weight of 1.0 and the connection leading out of the new node
        receives the old connection weight.

        Connections from bias nodes and non-expressed nodes are not split.

        Returns:
            bool: True is a node was added, False otherwise.
        """

        max_retries = 20
        attempts = 0
        while attempts < max_retries:
            # NOTE: Gene dictionaries could be replaced with RandomDict() for faster
            # random access (currently O(n)): https://github.com/robtandy/randomdict
            old_gene_key = random.choice(list(self.connections.keys()))
            old_connection_gene = self.connections[old_gene_key]

            mutation = (old_connection_gene.node_in,
                        old_connection_gene.node_out,
                        InnovationType.NEW_NODE)
            node_mutation_key = self.innovation_store.mutation_to_key.get(mutation)

            if (node_mutation_key in self.nodes) or \
               (not old_connection_gene.expressed) or \
               (old_connection_gene.node_in in self.biases):
                # Try again if the selected connection is not splittable or the
                # mutation has already been applied
                attempts += 1
            else:
                # Split the selected connection and add the node
                old_connection_gene.expressed = False

                node_key = self.add_node(old_connection_gene.node_in,
                                         old_connection_gene.node_out,
                                         node_type=NodeType.HIDDEN)

                self.add_connection(node_in=old_connection_gene.node_in,
                                    node_out=node_key,
                                    weight=1.0)

                self.add_connection(node_in=node_key,
                                    node_out=old_connection_gene.node_out,
                                    weight=old_connection_gene.weight)

                return True

        return False

    def mutate_weights(self):
        """Mutates (perturbs) or replaces each connection weight in the genome.

        Each weight is either replaced (with some probability, specified in the
        genome config) or perturbed.

        Replaced weights and perturbations are drawn from a uniform distribution
        with range [-weight_perturb_power, weight_perturb_power).
        """
        for key, gene in self.connections.items():
            if random.random() < self.config.weight_replace_prob:
                # Replace weight
                gene.weight = random.uniform(-1.0, 1.0) * self.config.weight_perturb_power
            else:
                # Perturb weight
                gene.weight += random.uniform(-1.0, 1.0) * self.config.weight_perturb_power

                # Ensure weight remains within the desired range
                gene.weight = max(self.config.weight_min_value, gene.weight)
                gene.weight = min(self.config.weight_max_value, gene.weight)

            assert self.config.weight_min_value <= gene.weight <= self.config.weight_max_value

    def configure_crossover(self, parent1, parent2, average):
        """Performs crossover between two genomes.

        If the two genomes have equal fitness then the joint and excess genes
        are inherited from the smaller genome.

        Args:
            parent1 (Genome): The first parent.
            parent2 (Genome): The second parent.
            average (bool): Whether or not to average the weights of mutual
                connections or choose at random from one of the parents.
        """
        # Ensure parent1 is the fittest
        if (parent1.fitness == parent2.fitness) and \
           (len(parent2.connections) < len(parent1.connections)):
            # Favour smaller genome
            parent1, parent2 = parent2, parent1
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1

        # Inherit connection genes
        for key, gene1 in parent1.connections.items():
            gene2 = parent2.connections.get(key)

            if gene2 is None:
                # gene1 is excess or disjoint
                self.connections[key] = gene1.copy()
            else:
                # gene is mutual, either average or randomly choose from parents
                if average:
                    self.connections[key] = gene1.copy()
                    self.connections[key].weight = (gene1.weight + gene2.weight) / 2
                else:
                    if random.random() > 0.5:
                        self.connections[key] = gene1.copy()
                    else:
                        self.connections[key] = gene2.copy()

                if (not gene1.expressed) or (not gene2.expressed):
                    # Probabilistically disable gene if disabled in at least one parent
                    if random.random() < self.config.gene_disable_prob:
                        self.connections[key].expressed = False

        # Inherit node genes
        for key, gene1 in parent1.nodes.items():
            gene2 = parent2.nodes.get(key)

            if gene2 is None:
                # gene1 is excess or disjoint
                self.nodes[key] = gene1.copy()
            else:
                # gene is mutual, randomly choose from parents (makes no difference for node genes)
                if random.random() > 0.5:
                    self.nodes[key] = gene1.copy()
                else:
                    self.nodes[key] = gene2.copy()

            # Add to input/output nodes if applicable
            if gene1.type == NodeType.INPUT:
                self.inputs.append(key)
            elif gene1.type == NodeType.OUTPUT:
                self.outputs.append(key)
            elif gene1.type == NodeType.BIAS:
                self.biases.append(key)

    def distance(self, other):
        """Computes the compatibility (genetic) distance between two genomes.

        This is used for deciding how to speciate the population. Distance is a
        function of the number of disjoint and excess genes, as well as the
        weight/bias differences of matching genes.

        Update (13.04.20): Distance is measured using the original compatibility
        distance measure defined by Stanley & Miikkulainen (2002).

        Args:
            other (Genome): The other genome to compare itself to.

        Returns:
            float: The genetic distance between itself and the other genome.
        """
        c1 = self.config.compatibility_disjoint_coefficient
        c2 = self.config.compatibility_weight_coefficient

        # Find size of larger genome (set to 1 if both genomes contain <= 20 genes)
        self_n_genes = len(self.connections)
        other_n_genes = len(other.connections)
        if self.config.normalise_gene_dist and (self_n_genes > 20 or other_n_genes > 20):
            n_genes = max(self_n_genes, other_n_genes)
        else:
            n_genes = 1

        # Connection gene distance
        all_connections = set(self.connections.keys()).union(set(other.connections.keys()))
        non_matching_connections = set(self.connections.keys()) ^ set(other.connections.keys())
        matching_connections = all_connections - non_matching_connections

        sum_weight_diff = 0.0
        for key in matching_connections:
            sum_weight_diff += abs(self.connections[key].weight - other.connections[key].weight)
        avg_weight_diff = sum_weight_diff / len(matching_connections) if matching_connections else 0.

        weight_dist = c2 * avg_weight_diff
        gene_dist = c1 * len(non_matching_connections) / n_genes

        return gene_dist + weight_dist

    def size(self):
        """Returns a measure of genome complexity.

        Returns:
            tuple: A measure of the complexity of the genome given by
                (number of nodes, number of enabled connections)
        """
        num_nodes = len(self.nodes)
        num_enabled_connections = len([1 for key, gene in self.connections.items() if gene.expressed])

        return num_nodes, num_enabled_connections
