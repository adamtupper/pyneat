"""Tests for the innovation module.

A longer description of the module. Sections are created with a section
header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension
"""
import os

import neat

from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet
from custom_neat.genome import Genome
from custom_neat.innovation import *


class TestInnovation:
    """Tests for the innovation module.
    """
    def setup_method(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'configurations', 'test_configuration.ini')
        self.config = neat.Config(Genome,
                                  Reproduction,
                                  SpeciesSet,
                                  neat.DefaultStagnation,
                                  config_path)

    def test_innovation_record_constructor(self):
        """Test the InnovationRecord constructor correctly initialises
        innovation records.
        """
        key = 0
        innovation_type = InnovationType.NEW_NODE
        node_in = 0
        node_out = 1

        record = InnovationRecord(key, innovation_type, node_in, node_out)

        assert key == record.key
        assert innovation_type == record.innov_type
        assert node_in == record.node_in
        assert node_out == record.node_out

    def test_innovation_store_constructor(self):
        """Test the InnovationStore constructor correctly initialises
        innovation stores.
        """
        innovation_store = InnovationStore()

        assert {} == innovation_store.mutation_to_key
        assert {} == innovation_store.key_to_record
        assert 0 == next(innovation_store._innovation_key_generator)

    def test_get_innovation_key_new(self):
        """Test getting a new innovation key for a new mutation.
        """
        pass

    def test_get_innovation_key_existing(self):
        """Test getting an existing innovation key for a mutation that has
        already previously occurred.
        """
        pass
