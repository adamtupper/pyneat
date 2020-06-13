"""Implements the core of the evolutionary algorithm.
"""
from neat.math_util import mean
from neat.reporting import ReporterSet

from pyneat.innovation import InnovationStore


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """This class implements the core evolution algorithm.

    The steps of the algorithm are as follows:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.

    Attributes:
        reporters (ReporterSet): The set of reporters used for logging.
        config (CustomConfig): The global configuration settings for the entire
            algorithm.
        reproduction (Reproduction): The reproduction scheme for generating
            genomes.
        innovation_store (InnovationStore): The store for innovation records for
            tracking structural mutations.
        fitness_criterion (function): The fitness function to assess the
            population with to test for termination.
        population (dict): The population of individuals. A dictionary of
            genome key, genome pairs.
        species (SpeciesSet): The speciation scheme for dividing the population
            into species.
        generation (int): The generation number.
        best_genome (Genome): The best genome discovered so far (according to
            fitness).
    """

    def __init__(self, config, initial_state=None):
        """Create a new population.

        Args:
            config (CustomConfig): The global configuration settings.
            initial_state (tuple): An optional starting point for the algorithm
                to continue from. Contains a Population, SpeciesSet and a
                generation number.
        """
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        self.innovation_store = InnovationStore()

        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size,
                                                           self.innovation_store)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None
        self.last_improved = 0

    def add_reporter(self, reporter):
        """Add a new reporter to the reporter set.

        Args:
            reporter (Reporter): The reporter to add to the reporter set.
        """
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        """Remove a reporter from the reporter set.

        Args:
            reporter (Reporter): The reporter to remove from the reporter set.
        """
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None, **kwargs):
        """Runs NEAT's genetic algorithm for at most n generations.  If n is
        None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.

        Args:
            fitness_function (function): The fitness function to assess genomes
                with.
            n (int): The maximum number of generations to run for.
            **kwargs: Extra arguments that are passed to the fitness function.

        Returns:
            Genome: The best genome found during the run(s).
        """
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Runtime check that the population size is always correct
            assert len(self.population.keys()) == self.config.pop_size

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config, **kwargs)

            # Draw genomes (useful for debugging)
            # for key, genome in self.population.items():
            #     visualize.draw_net(genome, filename=f'results/run_1/genomes/genome_{key}')

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            self.reporters.end_generation(self.config, self.population, self.species)

            # If the fitness of the entire population has not improved for more
            # than 20 generations, refocus the search into the most promising
            # spaces.
            if best.fitness == self.best_genome.fitness:
                self.last_improved = self.generation
            refocus = self.generation - self.last_improved > 20
            if refocus:
                self.last_improved = self.generation

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config,
                                                          self.species,
                                                          self.config.pop_size,
                                                          self.generation,
                                                          self.innovation_store,
                                                          refocus)

            # Runtime check to ensure that all genomes share an innovation store
            assert len(set([g.innovation_store for g in self.population.values()])) == 1

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size,
                                                                   self.innovation_store)
                else:
                    raise CompleteExtinctionException()

            # Dynamic compatibility distance thresholding
            # compat_threshold = self.config.species_set_config.compatibility_threshold
            # if len(self.species.species) > 35:
            #     compat_threshold += 0.3
            # elif len(self.species.species) < 25:
            #     compat_threshold -= 0.3
            # compat_threshold = max(0.3, compat_threshold)
            #
            # self.config.species_set_config.compatibility_threshold = compat_threshold
            # self.reporters.info(f'Compatibility threshold = {compat_threshold}')

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome
