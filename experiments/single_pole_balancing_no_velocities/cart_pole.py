"""General settings and implementation of the single-pole cart system dynamics.

Adapted from the example provided with NEAT-Python.
"""

from math import cos, pi, sin
import random


class CartPole(object):
    gravity = 9.8  # Acceleration due to gravity, positive is downward, m/sec^2
    mcart = 1.0  # Cart mass in kg
    mpole = 0.1  # Pole mass in kg
    lpole = 0.5  # Half the pole length in meters
    time_step = 0.01  # Time step size in seconds

    def __init__(self, x=None, dx=None, theta=None, dtheta=None,
                 position_limit=4.8, angle_limit=36):
        """Initialise the system.

        Args:
            x: The cart position in meters.
            dx: The cart velocity in meters per second.
            theta: The pole angle in degrees.
            dtheta: The pole angular velocity in degrees per second.
            position_limit: The cart position limit in meters.
            angle_limit: The pole angle limit in degrees.
        """
        self.position_limit = position_limit
        self.angle_limit = angle_limit

        # Initialise system, randomly if starting state not given
        if x is None:
            x = random.uniform(-0.5 * self.position_limit, 0.5 * self.position_limit)

        if theta is None:
            theta = random.uniform(-0.5 * self.angle_limit, 0.5 * self.angle_limit)

        if dx is None:
            dx = random.uniform(-1.0, 1.0)

        if dtheta is None:
            dtheta = random.uniform(-1.0, 1.0)

        self.t = 0.0
        self.x = x
        self.dx = dx
        self.theta = theta
        self.dtheta = dtheta

        self.xacc = 0.0
        self.tacc = 0.0

        # Convert angular measurements to radians for dynamics calculations
        self.angle_limit_radians = angle_limit * (pi / 180)
        self.theta = self.theta * (pi / 180)
        self.dtheta = self.dtheta * (pi / 180)

    def step(self, force):
        """Update the system state using leapfrog integration.

        Equations:

            x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
            v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt

        Args:
            force (float): The force applied by the agent on the cart. Measured
                in newtons.
        """
        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole
        dt = self.time_step

        # Remember acceleration from previous step.
        tacc0 = self.tacc
        xacc0 = self.xacc

        # Update position/angle.
        self.x += dt * self.dx + 0.5 * xacc0 * dt ** 2
        self.theta += dt * self.dtheta + 0.5 * tacc0 * dt ** 2

        # Compute new accelerations as given in "Correct equations for the dynamics of the cart-pole system"
        # by Razvan V. Florian (http://florian.io).
        # http://coneural.org/florian/papers/05_cart_pole.pdf
        st = sin(self.theta)
        ct = cos(self.theta)
        tacc1 = (g * st + ct * (-force - mp * L * self.dtheta ** 2 * st) / mt) / (L * (4.0 / 3 - mp * ct ** 2 / mt))
        xacc1 = (force + mp * L * (self.dtheta ** 2 * st - tacc1 * ct)) / mt

        # Update velocities.
        self.dx += 0.5 * (xacc0 + xacc1) * dt
        self.dtheta += 0.5 * (tacc0 + tacc1) * dt

        # Remember current acceleration for next step.
        self.tacc = tacc1
        self.xacc = xacc1
        self.t += dt

    def get_scaled_state(self):
        """Get system state, scaled into (approximately) [-1, 1].

        Returns:
            list: The scaled system state [x, dx, theta, dtheta].
        """
        return [self.x / self.position_limit,
                self.dx / 4.0,  # Assuming max velocity = 4.0 m/s
                (self.theta + self.angle_limit_radians) / self.angle_limit_radians,  # TODO: Check max angular velocity
                self.dtheta / self.angle_limit_radians]

    def get_angle_limit(self):
        """Return the angle limit in degrees.

        Returns:
            float: The angle limit in degrees.
        """
        return self.angle_limit_radians * (180 / pi)

    def get_state(self):
        """Return the state of the system.

        Units:
            - x (the cart position) is measured in meters.
            - dx (the cart velocity) is measured in meters per second.
            - theta (the pole angle) is measure in degrees.
            - dtheta (the pole angular velocity) is measured in degrees per
              second.

        Returns:
            tuple: The state of the system (x, dx, theta, dtheta)
        """
        return self.x, self.dx, self.theta * (180 / pi), self.dtheta * (180 / pi)


def continuous_actuator_force(action):
    """Convert the network output to a continuous force to be applied to the
    cart.

    Args:
        action (list): A scalar float vector in the range [-1, 1].

    Returns:
        float: The force to be applied to the cart in the range [-10, 10] N.
    """
    return 10.0 * action[0]


def noisy_continuous_actuator_force(action):
    """

    # TODO: Complete function docstring.
    # TODO: Check that the implementation conforms to my requirements.

    Args:
        action:

    Returns:

    """
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0


def discrete_actuator_force(action):
    """Convert the network action to a discrete force applied to the cart.

    Args:
        action ([float]): The action of the agent. Must be a scalar value in the
            range [-1, 1].

    Returns:
        float: A force of either 5 N (move cart right) or -5 N (move cart left)
            to be applied to the cart.
    """
    return 5.0 if action[0] > 0.0 else -10.0


def noisy_discrete_actuator_force(action):
    """

    # TODO: Complete function docstring.
    # TODO: Check that the implementation conforms to my requirements.

    Args:
        action:

    Returns:

    """
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0
