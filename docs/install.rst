============
Installation
============

If you want to draw the architectures of the networks encoded by genomes,
you will need to install Graphviz:

.. code-block:: bash

    sudo apt update
    sudo apt install graphviz graphviz-dev

The remaining dependencies (listed in the requirements file) can then be
installed using pip:

.. code-block:: bash

    pip install -r requirements.txt

Finally, PyNEAT can be install via:

.. code-block:: bash

    python setup.py install
