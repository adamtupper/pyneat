"""This module defines the genome encoding used by NEAT.
"""
from itertools import count
import random
import copy
from enum import Enum

from neat.activations import ActivationFunctionSet
from neat.config import ConfigParameter, write_pretty_params


__author__ = "Adam Tupper"
__copyright__ = "Adam Tupper"
__license__ = "mit"


class NodeTypes(Enum):
    """Define the types for nodes in the network.
    """
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class ConnectionGene:
    """Defines a connection gene used in the genome encoding.

    TODO: Complete class docstring.

    Attributes:
        ...
    """

    def __init__(self, in_node, out_node, weight, expressed):
        """Creates a ConnectionGene object with the required properties.

        Args:
            in_node:
            out_node:
            weight:
            expressed:
        """
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.expressed = expressed

    def __eq__(self, other):
        return (self.in_node, self.out_node, self.weight, self.expressed) == \
               (other.in_node, other.out_node, other.weight, other.expressed)


class NodeGene:
    """Defines a node gene used in the genome encoding.

    TODO: Complete class docstring.

    Attributes:
        ...
    """

    def __init__(self, type, bias, activation):
        """Creates a NodeGene object with the required properties.

        Args:
            type (NodeTypes):
            bias (float):
            activation (function):
        """
        self.type = type
        self.bias = bias
        self.activation = activation

    def __eq__(self, other):
        return (self.type, self.bias, self.activation) == \
               (other.type, other.bias, other.activation)


class GenomeConfig:
    """Sets up and holds configuration information for the Genome class.
    """
    def __init__(self, params):
        """GenomeConfig constructor.

        TODO: Complete method docstring.

        Args:
            params:
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

    def save(self, f):
        """

        TODO: Complete method docstring.

        Args:
            f:
        """
        write_pretty_params(f, self, self._params)


class Genome:
    """Defines a genome used to encode a neural network.
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
    def write_config(cls, f, config):
        """Takes a file-like object and the configuration object created by
        parse_config. This method should write the configuration item
        definitions to the given file.

        Note: This is a required interface method.

        TODO: Complete method docstring.

        Args:
            f:
            config (GenomeConfig):
        """
        config.save(f)

    def __init__(self, key):
        """Creates a Genome object with the required properties.

        Note: This is a required interface method.

        TODO: Write new test for when no input/output nodes are specified.
        TODO: Make tests more robust to weight and bias initialisations.

        Args:
            key (int): A unique identifier for the genome.
        """
        self.key = key
        self.fitness = None
        self.node_key_generator = count(0)

        # (gene key, gene) pairs for genes
        self.nodes = {}
        self.connections = {}

        # Store the keys for input and output node genes
        self.inputs = []
        self.outputs = []

    def configure_new(self, config):
        """Configure a new genome based on the given configuration.

        Note: This is a required interface method.

        Args:
            config (GenomeConfig): The genome configuration.
        """

        # Create the required number of input and output nodes
        for _ in range(config.num_inputs):
            key = next(self.node_key_generator)
            self.nodes[key] = NodeGene(
                type=NodeTypes.INPUT,
                bias=random.normalvariate(mu=0.0, sigma=config.bias_init_std_dev),
                activation=config.activation_defs.get(config.activation_func)
            )
            self.inputs.append(key)

        for _ in range(config.num_outputs):
            key = next(self.node_key_generator)
            self.nodes[key] = NodeGene(
                type=NodeTypes.OUTPUT,
                bias=random.normalvariate(mu=0.0, sigma=config.bias_init_std_dev),
                activation=config.activation_defs.get(config.activation_func)
            )
            self.outputs.append(key)

        # Add initial connections
        for in_node in self.inputs:
            for out_node in self.outputs:
                if random.random() < config.initial_conn_prob:
                    self.add_connection(
                        in_node,
                        out_node,
                        random.normalvariate(mu=0.0, sigma=config.weight_init_std_dev)
                    )

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

    def add_connection(self, in_node, out_node, weight, expressed=True):
        """Add a connection between two nodes.

        Args:
            in_node (int): The key of the input node for the new connection.
            out_node (int): The key of the output node for the new connection.
            weight (float): The weight of the connection. Must be a value
                between [0, 1].
            expressed (bool): True if the connection should be expressed in the
                phenotype, False otherwise.
        """
        new_connection_gene = ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=weight,
            expressed=expressed
        )
        self.connections[(in_node, out_node)] = new_connection_gene

    def mutate(self, config):
        """Mutate the genome.

        Note: This is a required interface method.

         Mutates the genome according to the mutation parameter values specified
         in the genome configuration.

        Args:
            config (GenomeConfig): The genome configuration.
        """
        if random.random() < config.weight_mutate_prob:
            self.mutate_weights(config.weight_replace_prob,
                                config.weight_init_std_dev,
                                config.weight_perturb_std_dev,
                                config.weight_min_value,
                                config.weight_max_value)

        if random.random() < config.bias_mutate_prob:
            self.mutate_biases(config.bias_replace_prob,
                               config.bias_init_std_dev,
                               config.bias_perturb_std_dev,
                               config.bias_min_value,
                               config.bias_max_value)

        if random.random() < config.conn_add_prob:
            self.mutate_add_connection(config.weight_init_std_dev)

        if random.random() < config.node_add_prob:
            activation_func = config.activation_defs.get(config.activation_func)
            self.mutate_add_node(activation_func)

    def mutate_add_connection(self, std_dev):
        """Performs an 'add connection' structural mutation.
        
        A single connection with a random weight is added between two previously
        unconnected nodes.

        Restrictions:
            - Output nodes cannot be the input node of a connection.
            - Input nodes cannot be the output node of a connection.
            - Don't allow connections between output nodes, or between input
              nodes.

        The restrictions prevent recurrent connections on the input and output
        nodes.

        Args:
            std_dev (float): The standard deviation for the normal distribution
                from which the weight of the new connection is chosen.
        """
        # Do not allow connections between output nodes, between input nodes
        possible_inputs = [k for k, g in self.nodes.items() if g.type != NodeTypes.OUTPUT]
        possible_outputs = [k for k, g in self.nodes.items() if g.type != NodeTypes.INPUT]

        in_node = random.choice(possible_inputs)
        out_node = random.choice(possible_outputs)

        # Check for existing connection, enable if disabled
        if (in_node, out_node) in self.connections.keys():
            self.connections[(in_node, out_node)].expressed = True
            return

        # Add a new connection
        connection_weight = random.normalvariate(mu=0.0, sigma=std_dev)
        self.add_connection(in_node, out_node, connection_weight)

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
        new_node_id = next(self.node_key_generator)
        self.nodes[new_node_id] = NodeGene(type=NodeTypes.HIDDEN,
                                           bias=0.0,
                                           activation=activation)

        # NOTE: Gene dictionaries could be replaced with RandomDict() for faster
        # random access (currently O(n)): https://github.com/robtandy/randomdict
        old_gene_key = random.choice(list(self.connections.keys()))
        old_connection_gene = self.connections[old_gene_key]
        old_connection_gene.expressed = False
        
        self.add_connection(
            in_node=old_connection_gene.in_node,
            out_node=new_node_id,
            weight=1.0
        )

        self.add_connection(
            in_node=new_node_id,
            out_node=old_connection_gene.out_node,
            weight=old_connection_gene.weight
        )

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
                gene.weight = random.normalvariate(mu=0.0, sigma=init_std_dev)
            else:
                # Perturb weight
                gene.weight += random.normalvariate(mu=0.0, sigma=perturb_std_dev)
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
                gene.bias = random.normalvariate(mu=0.0, sigma=init_std_dev)
            else:
                # Perturb bias
                gene.bias += random.normalvariate(mu=0.0, sigma=perturb_std_dev)
                gene.bias = max(min_val, gene.bias)
                gene.bias = min(max_val, gene.bias)

    def configure_crossover(self, genome1, genome2, config):
        """Performs crossover between two genomes.

        Note: This is a required interface method.

        If the two genomes have equal fitness then the joint and excess genes
        are inherited from genome1. Since genome1 and genome2 are chosen at
        random, this choice is random.

        Args:
            genome1 (Genome): The first parent.
            genome2 (Genome): The second parent.
            config (GenomeConfig): The genome configuration.
        """
        # Ensure genome1 is the fittest
        if genome1.fitness < genome2.fitness:
            genome1, genome2 = genome2, genome1

        # Inherit connection genes
        for key, gene1 in genome1.connections.items():
            gene2 = genome2.connections.get(key)

            if gene2 is None:
                # gene1 is excess or disjoint
                self.connections[key] = copy.deepcopy(gene1)
            else:
                # gene is mutual, randomly choose from parents
                if random.random() > 0.5:
                    self.connections[key] = copy.deepcopy(gene1)
                else:
                    self.connections[key] = copy.deepcopy(gene2)

                if not gene1.expressed or not gene2.expressed:
                    # Probabilistically disable gene
                    if random.random() < config.gene_disable_prob:
                        self.connections[key].expressed = False

        # Inherit node genes
        for key, gene1 in genome1.nodes.items():
            gene2 = genome2.nodes.get(key)

            if gene2 is None:
                # gene1 is excess or disjoint
                self.nodes[key] = copy.deepcopy(gene1)
            else:
                # gene is mutual, randomly choose from parents
                if random.random() > 0.5:
                    self.nodes[key] = copy.deepcopy(gene1)
                else:
                    self.nodes[key] = copy.deepcopy(gene2)

    def distance(self, other, config):
        """Computes the compatibility  (genetic) distance between two genomes.

        Note: This is a required interface method.

        This is used for deciding how to speciate the population. Distance is a
        function of the number of disjoint and excess genes, as well as the
        weight differences of matching genes.

        TODO: Investigate practical significance of the difference between the NEAT-Python and original measures of genetic distance.

        Args:
            other (Genome): The other genome to compare itself to.
            config (GenomeConfig): The genome configuration.

        Returns:
            float: The genetic distance between itself and the other genome.
        """
        c1 = config.compatibility_disjoint_coefficient
        c2 = config.compatibility_weight_coefficient

        # Node gene distance
        max_nodes = max(len(self.nodes), len(other.nodes))
        all_genes = set(self.nodes.keys()).union(set(other.nodes.keys()))
        non_matching_genes = set(self.nodes.keys()) ^ set(other.nodes.keys())
        matching_genes = all_genes - non_matching_genes

        avg_weight_diff = 0.0
        for key in matching_genes:
            avg_weight_diff += abs(self.nodes[key].bias - other.nodes[key].bias)
        avg_weight_diff = avg_weight_diff / len(matching_genes)

        node_distance = c1 * (len(non_matching_genes) / max_nodes) + c2 * avg_weight_diff

        # Connection gene distance
        max_connections = max(len(self.connections), len(other.connections))
        all_genes = set(self.connections.keys()).union(set(other.connections.keys()))
        non_matching_genes = set(self.connections.keys()) ^ set(other.connections.keys())
        matching_genes = all_genes - non_matching_genes

        avg_weight_diff = 0.0
        for key in matching_genes:
            avg_weight_diff += abs(self.connections[key].weight - other.connections[key].weight)
        avg_weight_diff = avg_weight_diff / len(matching_genes)

        connection_distance = c1 * (len(non_matching_genes) / max_connections) + c2 * avg_weight_diff

        return node_distance + connection_distance

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