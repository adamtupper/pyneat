"""Module for tracking structural innovations in genomes during evolution.
"""

from enum import Enum
from itertools import count


class InnovationType(Enum):
    """Define the types of structural innovations.
    """
    NEW_NODE = 0
    NEW_CONNECTION = 1


class InnovationRecord():
    """Record information about a structural mutations for comparison against
    other mutations.

    Attributes:
        innov_type (InnovationType): The type of structural mutation that
            occurred.
        node_in (int): The incoming node for the new connection or node (at
            the time of the mutation occurring in the case of new node
            mutations).
        node_out (int): The outgoing node for the new connection or node (at
            the time of the mutation occurring in the case of new node
            mutations).
    """

    def __init__(self, innov_type, node_in, node_out):
        """Create a new innovation record.

        Args:
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


class InnovationStore():
    """Store records of new node and connection mutations for lookup. Also
     responsible for generating unique innovation keys.

     Attributes:
         records (dict): A dictionary containing innovation records for each new
             structural mutation.
         innovation_key_generator (generator): Generates the next innovation
             key.
    """

    def __init__(self):
        """Create a new innovation record store.
        """
        self.records = {}
        self.innovation_key_generator = count(0)
