"""This module defines the genome encoding used by NEAT.
"""
from itertools import count
import random
import copy
from enum import Enum

from neat.activations import ActivationFunctionSet
from neat.config import ConfigParameter, write_pretty_params

from custom_neat.innovation import InnovationType


class NodeType(Enum):
    """Define the types for nodes in the network.
    """
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


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


class NodeGene:
    """Defines a node gene used in the genome encoding.

    Attributes:
        key (int): The innovation key (also the node key) for this gene.
        type (NodeType): The type of the node (either input, output or hidden).
        bias (float): The bias value of the node.
        activation (function): The node activation function.
    """

    def __init__(self, key, type, bias, activation):
        """Creates a new NodeGene object.

        Args:
            key (int): The innovation key (also the node key) for this gene.
            type (NodeType): The type of the node (either input, output or hidden).
            bias (float): The bias value of the node.
            activation (function): The node activation function.
        """
        self.key = key
        self.type = type
        self.bias = bias
        self.activation = activation

    def __eq__(self, other):
        """Test for equality against another node gene.

        Args:
            other (NodeGene): The node gene to compare.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        return (self.key, self.type, self.bias, self.activation) == \
               (other.key, other.type, other.bias, other.activation)


class GenomeConfig:
    """Sets up and holds configuration information for the Genome class.
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
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('initial_conn_prob', float),
                        ConfigParameter('weight_mutate_prob', float),
                        ConfigParameter('weight_replace_prob', float),
                        ConfigParameter('weight_init_std_dev', float),
                        ConfigParameter('weight_perturb_std_dev', float),
                        ConfigParameter('weight_min_value', float),
                        ConfigParameter('weight_max_value', float),
                        ConfigParameter('bias_mutate_prob', float),
                        ConfigParameter('bias_replace_prob', float),
                        ConfigParameter('bias_init_std_dev', float),
                        ConfigParameter('bias_perturb_std_dev', float),
                        ConfigParameter('bias_min_value', float),
                        ConfigParameter('bias_max_value', float),
                        ConfigParameter('gene_disable_prob', float),
                        ConfigParameter('activation_func', str)]

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
        innovation_store (InnovationStore): The global innovation store used for
            tracking new structural mutations.
    """
    @classmethod
    def parse_config(cls, param_dict):
        """Takes a dictionary of configuration items, returns an object that
        will later be passed to the write_config method.

        Note: This is a required interface method.

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

        Note: This is a required interface method.

        Args:
            filename (str): The name of the file to write the genome configuration to.
            config (GenomeConfig): The genome configuration to save.
        """
        config.save(filename)

    def __init__(self, key, config, innovation_store):
        """Creates a new Genome object.

        Note: This is a required interface method.

        TODO: Write new test for when no input/output nodes are specified.
        TODO: Make tests more robust to weight and bias initialisations.

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

        # Store the keys for input and output node genes
        self.inputs = []
        self.outputs = []

    def configure_new(self):
        """Configure a new genome based on the given configuration.

        The initial inputs and outputs for input and output nodes are specified
        as negatives so that matching innovation keys are generated for
        corresponding input and output nodes between genomes. Inputs nodes use
        odd negative numbers, and output nodes use even negative numbers.

        Note: This is a required interface method.
        """
        # Create the required number of input nodes
        for i in range(-1, -2 * self.config.num_inputs - 1, -2):
            self.add_node(i, i, random.uniform(-3.0, 3.0), NodeType.INPUT)

        # Create the required number of output nodes
        for i in range(-2, -2 * self.config.num_outputs - 1, -2):
            self.add_node(i, i, random.uniform(-3.0, 3.0), NodeType.OUTPUT)

        # Add initial connections
        for node_in in self.inputs:
            for node_out in self.outputs:
                if random.random() < self.config.initial_conn_prob:
                    # self.add_connection(node_in, node_out, random.normalvariate(mu=0.0, sigma=config.weight_init_std_dev))
                    self.add_connection(node_in, node_out, random.uniform(-3.0, 3.0))

    def __eq__(self, other):
        """Check for genome equality.

        Args:
            other (Genome): The genome to compare itself to.

        Returns:
            bool: True if this genome is equal to the other, False otherwise.
        """
        self_attrs = (self.key, self.nodes, self.connections, self.inputs, self.outputs)
        other_attrs = (other.key, other.nodes, other.connections, other.inputs, other.outputs)

        return self_attrs == other_attrs

    def copy(self):
        """Create a copy of the genome.

        Returns:
            Genome: A copy of itself.
        """
        return copy.deepcopy(self)

    def add_node(self, node_in, node_out, bias, node_type):
        """Add a new node positioned between two other nodes.

        Args:
            node_in (int): The key of the node that precedes this new node.
            node_out (int): The key of the node that succeeds this new node.
            bias (float): The bias value for the node.
            node_type (NodeType): The type of node to be added.

        Returns:
            int: The key of the new node
        """
        key = self.innovation_store.get_innovation_key(node_in, node_out, InnovationType.NEW_NODE)
        assert key not in self.nodes
        # self.nodes[key] = NodeGene(
        #     key=key
        #     type=node_type,
        #     bias=random.normalvariate(mu=0.0, sigma=config.bias_init_std_dev),
        #     activation=config.activation_defs.get(config.activation_func)
        # )
        self.nodes[key] = NodeGene(
            key=key,
            type=node_type,
            bias=bias,
            activation=self.config.activation_defs.get(self.config.activation_func)
        )

        if node_type == NodeType.INPUT:
            self.inputs.append(key)
        elif node_type == NodeType.OUTPUT:
            self.outputs.append(key)

        return key

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

        Note: This is a required interface method.

         Mutates the genome according to the mutation parameter values specified
         in the genome configuration.
        """
        if random.random() < self.config.weight_mutate_prob:
            self.mutate_weights(self.config.weight_replace_prob,
                                self.config.weight_init_std_dev,
                                self.config.weight_perturb_std_dev,
                                self.config.weight_min_value,
                                self.config.weight_max_value)

        if random.random() < self.config.bias_mutate_prob:
            self.mutate_biases(self.config.bias_replace_prob,
                               self.config.bias_init_std_dev,
                               self.config.bias_perturb_std_dev,
                               self.config.bias_min_value,
                               self.config.bias_max_value)

        if random.random() < self.config.conn_add_prob:
            self.mutate_add_connection(self.config.weight_init_std_dev)

        if random.random() < self.config.node_add_prob:
            activation_func = self.config.activation_defs.get(self.config.activation_func)
            self.mutate_add_node(activation_func)

    def mutate_add_connection(self, std_dev):
        """Performs an 'add connection' structural mutation.
        
        A single connection with a random weight is added between two previously
        unconnected nodes.

        Args:
            std_dev (float): The standard deviation for the normal distribution
                from which the weight of the new connection is chosen.
        """
        # Do not allow connections between output nodes, between input nodes
        possible_inputs = [k for k, g in self.nodes.items() if g.type != NodeType.OUTPUT]
        possible_outputs = [k for k, g in self.nodes.items() if g.type != NodeType.INPUT]

        node_in = random.choice(possible_inputs)
        node_out = random.choice(possible_outputs)

        # Check for existing connection, enable if disabled
        mutation = (node_in, node_out, InnovationType.NEW_CONNECTION)
        mutation_key = self.innovation_store.mutation_to_key.get(mutation)
        if mutation_key in self.connections:
            self.connections[mutation_key].expressed = True
            return

        # Add a new connection
        # connection_weight = random.normalvariate(mu=0.0, sigma=std_dev)
        connection_weight = random.uniform(-3.0, 3.0)
        self.add_connection(node_in, node_out, connection_weight)

    def mutate_add_node(self, activation):
        """Performs an 'add node' structural mutation.

        An existing connection is split and the new node is placed where the old
        connection used to be. The old connection is disabled and two new
        connection genes are added. The new connection leading into the new node
        receives a weight of 1.0 and the connection leading out of the new node
        receives the old connection weight.

        Args:
            activation (function): The activation function for the new node.
        """
        if self.connections:
            # Only add a new node if there are existing connections to replace

            # NOTE: Gene dictionaries could be replaced with RandomDict() for faster
            # random access (currently O(n)): https://github.com/robtandy/randomdict
            old_gene_key = random.choice(list(self.connections.keys()))
            old_connection_gene = self.connections[old_gene_key]

            mutation = (old_connection_gene.node_in,
                        old_connection_gene.node_out,
                        InnovationType.NEW_NODE)
            node_mutation_key = self.innovation_store.mutation_to_key.get(mutation)
            if node_mutation_key in self.nodes:
                # Skip if this mutation is already present in this genome
                return

            old_connection_gene.expressed = False

            node_key = self.add_node(old_connection_gene.node_in,
                                     old_connection_gene.node_out,
                                     bias=0.0,
                                     node_type=NodeType.HIDDEN)

            self.add_connection(node_in=old_connection_gene.node_in,
                                node_out=node_key,
                                weight=1.0)

            self.add_connection(node_in=node_key,
                                node_out=old_connection_gene.node_out,
                                weight=old_connection_gene.weight)

    def mutate_weights(self, replace_prob, init_std_dev, perturb_std_dev, min_val, max_val):
        """Performs weight mutations.

        Mutates (perturbs) each connection weight in the genome with some
        probability. Weights are either perturbed by an amount drawn from a
        normal distribution with mean=0 and standard deviation=perturb_std_dev
        or are replaced with a random value from the initialising normal
        distribution with mean=0 and standard_deviation=init_std_dev.

        TODO: Investigate how much perturbed weights/biases are changed by in other implementations.

        Args:
            replace_prob (float):  The probability of a weight being replaced as
                opposed to perturbed. Must be a value in the range [0, 1].
            init_std_dev (float): The standard deviation of the normal
                distribution from which to draw a replacement weight.
            perturb_std_dev (float): The standard deviation of the normal
                distribution from which to draw the amount to perturb each
                weight.
            min_val (float): The minimum allowed weight value.
            max_val (float): The maximum allowed weight value.
        """
        for key, gene in self.connections.items():
            if random.random() < replace_prob:
                # Replace weight
                # gene.weight = random.normalvariate(mu=0.0, sigma=init_std_dev)
                gene.weight = random.uniform(-3.0, 3.0)
            else:
                # Perturb weight
                # gene.weight += random.normalvariate(mu=0.0, sigma=perturb_std_dev)
                gene.weight += random.uniform(-perturb_std_dev, perturb_std_dev)
                gene.weight = max(min_val, gene.weight)
                gene.weight = min(max_val, gene.weight)

    def mutate_biases(self, replace_prob, init_std_dev, perturb_std_dev, min_val, max_val):
        """Performs bias mutations.

        Mutates (perturbs) each node bias in the genome with some probability.
        Biases are either perturbed by an amount drawn from a normal
        distribution with mean=0 and standard deviation=perturb_std_dev or are
        replaced with a random value from the initialising normal distribution
        with mean=0 and standard_deviation=init_std_dev.

        Args:
            replace_prob (float):  The probability of a bias being replaced as
                opposed to perturbed. Must be a value in the range [0, 1].
            init_std_dev (float): The standard deviation of the normal
                distribution from which to draw a replacement bias.
            perturb_std_dev (float): The standard deviation of the normal
                distribution from which to draw the amount to perturb each
                bias.
            min_val (float): The minimum allowed bias value.
            max_val (float): The maximum allowed bias value.
        """
        for key, gene in self.nodes.items():
            if random.random() < replace_prob:
                # Replace bias
                # gene.bias = random.normalvariate(mu=0.0, sigma=init_std_dev)
                gene.bias = random.uniform(-3.0, 3.0)
            else:
                # Perturb bias
                # TODO: Rename config param to reflect normal dist limits
                # gene.bias += random.normalvariate(mu=0.0, sigma=perturb_std_dev)
                gene.bias += random.uniform(-perturb_std_dev, perturb_std_dev)
                gene.bias = max(min_val, gene.bias)
                gene.bias = min(max_val, gene.bias)

    def configure_crossover(self, parent1, parent2):
        """Performs crossover between two genomes.

        Note: This is a required interface method.

        If the two genomes have equal fitness then the joint and excess genes
        are inherited from parent1. Since parent1 and parent2 are chosen at
        random, this choice is random.

        Args:
            parent1 (Genome): The first parent.
            parent2 (Genome): The second parent.
        """
        # Ensure parent1 is the fittest
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1

        # Inherit connection genes
        for key, gene1 in parent1.connections.items():
            gene2 = parent2.connections.get(key)

            if gene2 is None:
                # gene1 is excess or disjoint
                self.connections[key] = copy.deepcopy(gene1)
            else:
                # gene is mutual, randomly choose from parents
                if random.random() > 0.5:
                    self.connections[key] = copy.deepcopy(gene1)
                else:
                    self.connections[key] = copy.deepcopy(gene2)

                if (not gene1.expressed) or (not gene2.expressed):
                    # Probabilistically disable gene if disabled in at least one parent
                    if random.random() < self.config.gene_disable_prob:
                        self.connections[key].expressed = False

        # Inherit node genes
        for key, gene1 in parent1.nodes.items():
            gene2 = parent2.nodes.get(key)

            if gene2 is None:
                # gene1 is excess or disjoint
                self.nodes[key] = copy.deepcopy(gene1)
            else:
                # gene is mutual, randomly choose from parents
                if random.random() > 0.5:
                    self.nodes[key] = copy.deepcopy(gene1)
                else:
                    self.nodes[key] = copy.deepcopy(gene2)

            # Add to input/output nodes if applicable
            if gene1.type == NodeType.INPUT:
                self.inputs.append(key)
            elif gene1.type == NodeType.OUTPUT:
                self.outputs.append(key)

    def distance(self, other):
        """Computes the compatibility  (genetic) distance between two genomes.

        Note: This is a required interface method.

        This is used for deciding how to speciate the population. Distance is a
        function of the number of disjoint and excess genes, as well as the
        weight differences of matching genes.

        Update (11.02.20): Distance is measured as topological dissimilarity.
        It is implemented as a proportion of enabled and matching node and
        connection genes. 0 = topologically identical, 1 = no matching genes.

        TODO: Decide how distance should be calculated.

        Args:
            other (Genome): The other genome to compare itself to.

        Returns:
            float: The genetic distance between itself and the other genome.
        """
        c1 = self.config.compatibility_disjoint_coefficient
        c2 = self.config.compatibility_weight_coefficient

        # Find size of larger genome (count only expressed connections)
        N = max(len(self.nodes) + len([g for g in self.connections.values() if g.expressed]),
                len(other.nodes) + len([g for g in other.connections.values() if g.expressed]))

        # Node gene distance
        all_nodes = set(self.nodes.keys()).union(set(other.nodes.keys()))
        non_matching_nodes = set(self.nodes.keys()) ^ set(other.nodes.keys())
        matching_nodes = all_nodes - non_matching_nodes

        avg_bias_diff = 0.0
        for key in matching_nodes:
            avg_bias_diff += abs(self.nodes[key].bias - other.nodes[key].bias)

        if matching_nodes:
            avg_bias_diff = avg_bias_diff / len(matching_nodes)

        # Connection gene distance (count only expressed connections)
        all_connections = set([k for k, g in self.connections.items() if g.expressed]).union(set([k for k, g in other.connections.items() if g.expressed]))
        non_matching_connections = set([k for k, g in self.connections.items() if g.expressed]) ^ set([k for k, g in other.connections.items() if g.expressed])
        matching_connections = all_connections - non_matching_connections

        avg_weight_diff = 0.0
        for key in matching_connections:
            avg_weight_diff += abs(self.connections[key].weight - other.connections[key].weight)

        if matching_connections:
            avg_weight_diff = avg_weight_diff / len(matching_connections)

        gene_dist = c1 * (len(non_matching_nodes) + len(non_matching_connections)) / N
        weight_dist = c2 * (avg_weight_diff + avg_bias_diff) / 2

        return gene_dist + weight_dist

    def size(self):
        """Returns a measure of genome complexity.

        Note: This is a required interface function.

        Returns:
            tuple: A measure of the complexity of the genome given by
                (number of nodes, number of enabled connections)
        """
        num_nodes = len(self.nodes)
        num_enabled_connections = len([1 for key, gene in self.connections.items() if gene.expressed])

        return num_nodes, num_enabled_connections
