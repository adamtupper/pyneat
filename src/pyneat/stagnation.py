"""Track the progress of species and remove those that have stalled.

TODO: Add stagnation tests.
TODO: Update module docstrings, and clean up comments and code.
"""
import sys

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import stat_functions

# TODO: Add a method for the user to change the "is stagnant" computation.


class Stagnation(DefaultClassConfig):
    """Track the progress of species and remove those that have stalled.
    """
    @classmethod
    def parse_config(cls, param_dict):
        """Parses the stagnation configuration parameters.

        Config Parameters:
            species_fitness_func (str): The function (mean, max) for
                aggregating the fitnesses of the members of each species.
            max_stagnation (int): The maximum number of generations a
                species can stall for before being deemed stagnant.
            species_elitism (int): The minimum number of species that should
                be retained.

        TODO: Refactor `species_elitism` as `min_species`

        Args:
            param_dict: ...

        Returns:
            DefaultClassConfig: ...
        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('species_fitness_func', str, 'mean'),
                                   ConfigParameter('max_stagnation', int, 15),
                                   ConfigParameter('species_elitism', int, 0)])

    def __init__(self, config, reporters):
        """

        Args:
            config:
            reporters:
        """
        # pylint: disable=super-init-not-called
        self.stagnation_config = config

        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(
                "Unexpected species fitness func: {0!r}".format(config.species_fitness_func))

        self.reporters = reporters

    def update(self, species_set, generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.

        Args:
            species_set: ...
            generation: ...

        Returns:
            list: ...
        """
        species_data = []
        for sid, s in species_set.species.items():
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            s.fitness = self.species_fitness_func(s.get_fitnesses())
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = generation

            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:
                is_stagnant = stagnant_time >= self.stagnation_config.max_stagnation

            # This allows the species with the highest fitness to remain even if
            # they're stagnant - not original behaviour
            # if (len(species_data) - idx) <= self.stagnation_config.species_elitism:
            #     is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result
