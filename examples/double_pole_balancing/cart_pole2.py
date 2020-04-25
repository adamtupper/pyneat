"""Cart pole module.
"""
import sys
from random import randint
from dpole import integrate  # wrapped from C++
from pyneat.nn.recurrent import RNN


class CartPole(object):
    def __init__(self, population, markov, print_status=True):
        self.__population = population
        self.__markov = markov
        self.__state = []
        self.print_status = print_status

        self.cart_limit = 2.4  # metres
        self.angle_limit = 0.628329  # 36 degrees (in radians)

        self.__initial_state()  # Initialise the environment

    state = property(lambda self: self.__state)

    def run(self, testing=False):
        """ Runs the cart-pole experiment and evaluates the population. """

        if self.__markov:
            # markov experiment: full system's information is provided to the network
            for genome in self.__population:
                # genome to phenotype
                assert len(genome.inputs) == 6, "There must be 6 inputs to the network"
                net = RNN.create(genome)

                self.__initial_state()

                if testing:
                    # cart's position, first pole's angle, second pole's angle
                    # print "\nInitial conditions:"
                    print(f"{self.__state[0]:.2f} \t {self.__state[2]:.2f} \t {self.__state[4]:.2f}")
                    pass

                steps = 0

                while steps < 100000:
                    inputs = [self.__state[0] / 4.80,  # cart's initial position
                              self.__state[1] / 2.00,  # cart's initial speed
                              self.__state[2] / 0.52,  # pole_1 initial angle
                              self.__state[3] / 2.00,  # pole_1 initial angular velocity
                              self.__state[4] / 0.52,  # pole_2 initial angle
                              self.__state[5] / 2.00]  # pole_2 initial angular velocity

                    # activate the neural network
                    output = net.forward(inputs)
                    # maps [-1,1] onto [0,1]
                    # action = 0.5 * (output[0] + 1.0)
                    action = output
                    # advances one time step
                    self.__state = integrate(action, self.__state, 1)

                    if self.__outside_bounds():
                        # network failed to solve the task
                        if testing:
                            print(f"Failed at step {steps} \t {self.__state[0]:.2f} \t {self.__state[2]:.2f} \t {self.__state[4]:.2f}")
                            sys.exit(0)
                        else:
                            break
                    steps += 1

                genome.fitness = float(steps)  # the higher the better
                # if self.print_status:
                #    print "Chromosome %3d evaluated with score: %d " %(genome.id, genome.fitness)

        else:
            # non-markovian: no velocity information is provided (only 3 inputs)
            for genome in self.__population:
                assert len(genome.inputs) == 3, "There must be 3 inputs to the network"
                net = RNN.create(genome)
                self.__initial_state()

                genome.fitness, score = self.__non_markov(net, 1000, testing)
                genome.score = score

                # print "Chromosome %3d %s evaluated with fitness %2.5f and score: %s" %(genome.id, genome.size(), genome.fitness, score)

            # we need to make sure that the found solution is robust enough and good at
            # generalizing for several different initial conditions, so the champion
            # from each generation (i.e., the one with the highest F) passes for a
            # generalization test (the criteria here was defined by Gruau)
            best = max(self.__population, key=lambda g: g.fitness)  # selects the best network
            if self.print_status:
                print(f"\t\nBest chromosome of generation: {best.key}")

            # ** *******************#
            #  GENERALIZATION TEST  #
            # **********************#

            # first: can it balance for at least 100k steps?
            best_net = RNN.create(best)
            best_net.reset()
            self.__initial_state()  # reset initial state
            # long non-markovian test
            if self.print_status:
                print("Starting the 100k test...")
            score = self.__non_markov(best_net, 100000, testing)[1]

            if score > 99999:
                if self.print_status:
                    print("\tWinner passed the 100k test! Starting the generalization test...")
                # second: now let's try 625 different initial conditions
                balanced = self.__generalization_test(best_net, testing)

                if balanced > 200:
                    if self.print_status:
                        print(f"\tWinner passed the generalization test with score: {balanced}\n")
                    # set chromosome's fitness to 100k (and ceases the simulation)
                    best.fitness = 100000
                    best.score = balanced
                else:
                    if self.print_status:
                        print(f"\tWinner failed the generalization test with score: {balanced}\n")

            else:
                if self.print_status:
                    print(f"\tWinner failed at the 100k test with score {score}\n")

    def get_state(self):
        """Return the state of the system.

        Returns:
            list: The state of the system.
        """
        return self.__state

    def get_scaled_state(self):
        """Return the scaled state of the system.

        Note: Velocities are not scaled!

        Returns:
            list: The scaled state of the system.
        """
        return [
            self.__state[0] / self.cart_limit,
            self.__state[1],
            self.__state[2] / self.angle_limit,
            self.__state[3],
            self.__state[4] / self.angle_limit,
            self.__state[5]
        ]

    def step(self, action):
        """Advance the system by one time step.

        Args:
            action (float): The action of the agent. A value in the range [0, 1].
        """
        self.__state = integrate(action, self.__state, 1)

    def outside_bounds(self):
        return self.__outside_bounds()

    def __non_markov(self, network, max_steps, testing):
        # variables used in Gruau's fitness function
        den = 0.0
        f1 = 0.0
        f2 = 0.0
        F = 0.0
        last_values = []

        steps = 0
        while steps < max_steps:
            inputs = [self.__state[0] / self.cart_limit,   # cart's initial position
                      self.__state[2] / self.angle_limit,  # pole_1 initial angle
                      self.__state[4] / self.angle_limit]  # pole_2 initial angle

            # Activate the neural network and advance system state
            action = network.forward(inputs)[0]

            if steps < 1 or steps % 2 == 0:
                # No action for first time step and only every second time step
                action = 0.5

            self.__state = integrate(action, self.__state, 1)

            if self.__outside_bounds():
                # network failed to solve the task
                if testing:
                    print(f"Failed at step {steps} \t {self.__state[0]:.2f} \t {self.__state[2]:.2f} \t {self.__state[4]:.2f}")
                    break
                else:
                    # print "Failed at step %d" %steps
                    break

            den = abs(self.__state[0]) + abs(self.__state[1]) + \
                  abs(self.__state[2]) + abs(self.__state[3])
            last_values.append(den)

            if len(last_values) == 100:
                last_values.pop(0)  # we only need to keep the last 100 values

            steps += 1

        # compute Gruau's fitness
        if steps > 100:
            # the denominator is computed only for the last 100 time steps
            jiggle = sum(last_values)
            F = 0.1 * steps / 1000.0 + 0.9 * 0.75 / (jiggle)
        else:
            F = 0.1 * steps / 1000.0

        return (steps, steps)

    def __generalization_test(self, best_net, testing):
        values = [0.05, 0.25, 0.5, 0.75, 0.95]

        balanced = 0
        test_number = 0
        for x in values:
            for x_dot in values:
                for theta in values:
                    for theta_dot in values:
                        self.__state = [x * 4.32 - 2.16,  # set cart's position
                                        x_dot * 2.70 - 1.35,  # set cart's velocity
                                        theta * 0.12566304 - 0.06283152,  # set pole_1 angle
                                        theta_dot * 0.30019504 - 0.15009752,  # set pole_1 angular velocity
                                        0.0,
                                        0.0]

                        test_number += 1
                        best_net.reset()
                        score = self.__non_markov(best_net, 1000, testing)[1]
                        if (score > 999):
                            balanced += 1
                            if self.print_status:
                                print(f"Test {test_number} succeeded with score: {score}")
                        else:
                            if self.print_status:
                                print(f"Test {test_number} failed with score...: {score}")
        return balanced

    def __initial_state(self):
        """ Sets the initial state of the system. """

        # according to Stanley (p. 45) the initial state condition is not random
        rand = False

        if rand:
            self.__state = [randint(0, 4799) / 1000.0 - 2.4,  # cart's initial position
                            randint(0, 1999) / 1000.0 - 1.0,  # cart's initial speed
                            randint(0, 399) / 1000.0 - 0.2,  # pole_1 initial angle
                            randint(0, 399) / 1000.0 - 0.2,  # pole_1 initial angular velocity
                            randint(0, 2999) / 1000.0 - 0.4,  # pole_2 initial angle
                            randint(0, 2999) / 1000.0 - 0.4]  # pole_2 initial angular velocity
        else:
            self.__state = [0.0,
                            0.0,
                            0.0174533,  # set pole_1 to one degree (in radians)
                            0.0,
                            0.0,
                            0.0]

    def __outside_bounds(self):
        """ Check if outside the bounds. """

        failureAngle = 0.628329  # thirty_six_degrees

        return abs(self.__state[0]) > 2.4 or \
               abs(self.__state[2]) > failureAngle or \
               abs(self.__state[4]) > failureAngle
