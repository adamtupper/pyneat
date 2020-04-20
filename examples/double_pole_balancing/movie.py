"""Movie module from NEAT-Python single pole balancing example.
"""
import math

import gizeh as gz
import moviepy.editor as mpy
from cart_pole2 import CartPole


def make_movie(net, duration_seconds, output_filename):
    w, h = 480, 100
    scale = 480 / 10

    cart = gz.rectangle(lx=0.5 * scale, ly=0.25 * scale, xy=(240, 80), stroke_width=1, fill=(0, 1, 0))
    long_pole = gz.rectangle(lx=0.1 * scale, ly=1.0 * scale, xy=(240, 55), stroke_width=1, fill=(1, 0, 0))
    short_pole = gz.rectangle(lx=0.1 * scale, ly=0.5 * scale, xy=(240, 70), stroke_width=1, fill=(0, 0, 1))

    env = CartPole(population=None, markov=False)
    time_steps = 0

    x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot = env.get_state()
    x_vals = [x]                        # Cart position
    x_dot_vals = [x_dot]                # Cart velocity
    theta_1_vals = [theta_1]            # Long pole angle
    theta_1_dot_vals = [theta_1_dot]    # Long pole velocity
    theta_2_vals = [theta_2]            # Short pole angle
    theta_2_dot_vals = [theta_2_dot]    # Short pole velocity

    def make_frame(t):
        nonlocal time_steps, x_vals, x_dot_vals, theta_1_vals, theta_1_dot_vals, theta_2_vals, theta_2_dot_vals

        # Activate network and advance environment
        obs = env.get_scaled_state()
        obs = [obs[0], obs[2], obs[4]]  # Remove velocities
        action = net.forward(obs)[0]
        env.step(action)

        # Increment time step counter and store new system state
        time_steps += 1
        x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot = env.get_state()
        x_vals.append(x)
        x_dot_vals.append(x_dot)
        theta_1_vals.append(theta_1)
        theta_1_dot_vals.append(theta_1_dot)
        theta_2_vals.append(theta_2)
        theta_2_vals.append(theta_2_dot)

        surface = gz.Surface(w, h, bg_color=(1, 1, 1))

        # Add state information
        text = gz.text(f'{time_steps} time steps', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 10))
        text.draw(surface)
        text = gz.text(f'x = {x:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 20))
        text.draw(surface)
        text = gz.text(f'x_dot = {x_dot:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 26))
        text.draw(surface)
        text = gz.text(f'theta_1 = {theta_1 * 180 / math.pi:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 32))
        text.draw(surface)
        text = gz.text(f'theta_1_dot = {theta_1_dot * 180 / math.pi:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 38))
        text.draw(surface)
        text = gz.text(f'theta_2 = {theta_2 * 180 / math.pi:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 44))
        text.draw(surface)
        text = gz.text(f'theta_2_dot = {theta_2_dot * 180 / math.pi:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 50))
        text.draw(surface)

        # Convert position to display units
        visX = x * scale

        # Draw cart
        group = gz.Group((cart,)).translate((visX, 0))
        group.draw(surface)

        # Draw long pole
        group = gz.Group((long_pole,)).translate((visX, 0)).rotate(theta_1, center=(240 + visX, 80))
        group.draw(surface)

        # Draw short pole
        group = gz.Group((short_pole,)).translate((visX, 0)).rotate(theta_2, center=(240 + visX, 80))
        group.draw(surface)

        return surface.get_npimage()

    clip = mpy.VideoClip(make_frame, duration=duration_seconds)
    clip.write_videofile(output_filename, codec="mpeg4", fps=30)

    # print(x_vals)
    # print()
    # print(x_dot_vals)
    # print()
    # print(theta_1_vals)
    # print()
    # print(theta_1_dot_vals)