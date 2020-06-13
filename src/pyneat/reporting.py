"""Additional reporters (on top of what are offered by NEAT-Python) that are
triggered on particular events.
"""
import logging
import time

from neat.reporting import BaseReporter
from numpy import mean, std


class LogFileReporter(BaseReporter):
    """Write the same information as the StdOutReporter, but instead of printing
    to standard out, the outputs are written to a log file.

    Attributes:
        filename (str): The name of the log file to use.
        log_level (logging.level): The log level (either DEBUG, INFO, WARNING,
            ERROR, or CRITICAL). Note: All LogFileReporter logs are INFO level.
        show_species_detail (bool): Whether to show detailed species
            information.
        generation (int): The current generation.
        generation_start_time (int): The start time of the current generation.
        generation_times (list): The length of time to complete each generation.
        num_extinctions (int): The number of extinctions that have occurred.
    """
    def __init__(self, filename, log_level=logging.INFO, show_species_detail=True):
        """Initialise a new LogFileReporter.

        Args:
            filename (str): The name of the log file to use.
            log_level (logging.level): The log level (either DEBUG, INFO,
                WARNING, ERROR, or CRITICAL). Note: All LogFileReporter logs are
                INFO level.
            show_species_detail (bool): Whether to show detailed species
                information.
        """
        self.filename = filename
        self.log_level = log_level
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

        logging.basicConfig(filename=self.filename, level=self.log_level)

    def start_generation(self, generation):
        """The log message to write when a generation is started.

        Args:
            generation (int): The current generation number.
        """
        self.generation = generation
        logging.info('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        """The log message to write when a generation is ended.

        Args:
            config (CustomConfig): The global configuration settings for the
                entire algorithm.
            population (dict): The population of individuals. A dictionary of
                genome key, genome pairs.
            species_set (SpeciesSet): The speciation scheme for dividing the
                population into species.
        """
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            logging.info('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            logging.info("   ID   age  size  fitness  adj fit  stag")
            logging.info("  ====  ===  ====  =======  =======  ====")
            for sid in sorted(species_set.species):
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                logging.info(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            logging.info('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        logging.info('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            logging.info("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            logging.info("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        """The log message to write after evaluating the fitness of the
        population.

        Args:
            config (CustomConfig): The global configuration settings for the
                entire algorithm.
            population (dict): The population of individuals. A dictionary of
                genome key, genome pairs.
            species (SpeciesSet): The speciation scheme for dividing the
                population into species.
            best_genome (Genome): The best genome from the population.
        """
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = std(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        logging.info('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        logging.info(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        """The log message to write after a complete extinction has occurred.
        """
        self.num_extinctions += 1
        logging.info('All species extinct.')

    def found_solution(self, config, generation, best):
        """The log message to write after a solution has been found.
        """
        logging.info('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        """The log message to write when a species has stagnated.
        """
        if self.show_species_detail:
            logging.info("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        """Write a custom log message.

        Args:
            msg (str): The log message to write.
        """
        logging.info(msg)
