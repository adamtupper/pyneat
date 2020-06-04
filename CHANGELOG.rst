=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html)>`_.

Current
=======

Added
#####

- Average crossover operation that averages the weights of mutual connections, instead of choosing from the parents at
  random. The probability of this occurring is governed by the :code:`crossover_avg_prob` config parameter.
- Custom stagnation behaviour that doesn't preserve elite species if the number of species is above the species elitism
  threshold. This is more inline with the original implementation of NEAT. Previously, the stagnation behaviour of
  NEAT-Python was used.
- The visualise module (for visualising results) that was included with examples has now been added to the library
  proper.

Changed
#######

- Changed :code:`crossover_prob` config parameter to :code:`mutate_only_prob` (equivalent to
  1 - :code:`crossover_prob`).
- Added retries for add node mutations.
- Reduced the maximum weight magnitudes for the XOR example to match the original implementation of NEAT. Now weights
  are restricted to the range [-8., 8.].
- The :code:`Population.run` function can now take additional keyword arguments that are passed to the provided
  fitness evaluation function.

Fixed
#####

- Fixed bug preventing :code:`survival_threshold` from being applied.
- Species adjusted fitness is now saved to the right attribute.

Version 0.2.0 (25-04-2020)
==========================

Added
#####

- Support for feed-forward networks.
- A configurable probability that offspring generated through crossover are not also mutated
  (:code:`crossover_only_prob`).
- Steepened sigmoid activation function.
- Markov and non-Markov double pole balancing examples.
- Non-Markov single pole balancing example.
- XOR example.

Changed
#######

- Node and connection gene keys are now generated and kept track of using an :code:`InnovationStore` to improve
  crossover.
- Nodes no longer have individual biases, instead bias nodes are (optionally) used. The matches the original
  specification of NEAT (and reduces the search space).
- "Add connection" mutations now retry a certain number of times if a connection cannot be added (or is already present)
  between the selected nodes.
- Switched from Gaussian to uniform connection weight initialisation and perturbation to match the original NEAT
  specification.
- Now ensure that weight mutations are not performed if a structural mutation has been performed.
- Improved the method for identifying nodes that are required to compute network outputs. Previously, some unused nodes
  were also included.

Fixed
#####

- Small errors in the implementation of adjusted fitness and the largest remainder allocation method that are used for
  calculating species offspring allocations.
- A bug causing the same parent sometimes being selected twice for crossover.
- A bug in the forward pass of RNNs that caused the current state of the outputs (instead of the previous state) to be
  returned.

Version 0.1.0 (13-02-2020)
==========================

Added
#####

- Initial implementation based on `NEAT-Python <https://github.com/CodeReclaimers/neat-python>`_.
