"""Module for tracking structural innovations in genomes during evolution.
"""

from enum import Enum
from itertools import count


class InnovationType(Enum):
    """Define the types of structural innovations.
    """
    NEW_NODE = 0
    NEW_CONNECTION = 1


class InnovationRecord:
    """Record information about a structural mutations for comparison against
    other mutations.

    Attributes:
        key (int): A unique identifier for the record.
        innov_type (InnovationType): The type of structural mutation that
            occurred.
        node_in (int): The incoming node for the new connection or node (at
            the time of the mutation occurring in the case of new node
            mutations).
        node_out (int): The outgoing node for the new connection or node (at
            the time of the mutation occurring in the case of new node
            mutations).
    """

    def __init__(self, key, innov_type, node_in, node_out):
        """Create a new innovation record.

        Args:
            key (int): A unique identifier for the record.
            innov_type (InnovationType): The type of structural mutation that
                occurred.
            node_in (int): The incoming node for the new connection or node (at
                the time of the mutation occurring in the case of new node
                mutations).
            node_out (int): The outgoing node for the new connection or node (at
                the time of the mutation occurring in the case of new node
                mutations).
        """
        self.innov_type = innov_type
        self.node_in = node_in
        self.node_out = node_out


class InnovationStore:
    """Store records of new node and connection mutations for lookup. Also
    responsible for generating unique innovation keys.

    Attributes:
        key_to_record (dict): A dictionary containing innovation records for
            each new structural mutation, indexed by innovation keys.
        mutation_to_key (dict): A dictionary containing innovation keys, indexed
            by mutations (node_in, node_out, innovation_type).
        _innovation_key_generator (generator): Generates the next innovation
            key.
    """

    def __init__(self):
        """Create a new innovation record store.
        """
        self.key_to_record = {}
        self.mutation_to_key = {}
        self._innovation_key_generator = count(0)

    def get_innovation_key(self, node_in, node_out, innovation_type):
        """Get a new or existing innovation key for a structural mutation.

        Args:
            node_in (int): The input node to the new node/connection.
            node_out (int): The output node to the new node/connection.
            innovation_type (InnovationType): The type of structural mutation.

        Returns:
            int: The innovation key for the mutation.
        """
        key = self.mutation_to_key.get((node_in, node_out, innovation_type))
        if key:
            # This mutation has already occurred, return the existing key
            return key

        # Generate a new key and create a new record
        key = next(self._innovation_key_generator)
        record = InnovationRecord(key, innovation_type, node_in, node_out)
        self.key_to_record[key] = record
        self.mutation_to_key[(node_in, node_out, innovation_type)] = key

        return key
