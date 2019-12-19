"""Movie module from NEAT-Python single pole balancing example.
"""
import gizeh as gz
import moviepy.editor as mpy
from cart_pole import CartPole


def make_movie(net, force_function, duration_seconds, output_filename):
    w, h = 300, 100
    scale = 300 / 6

    cart = gz.rectangle(scale * 0.5, scale * 0.25, xy=(150, 80), stroke_width=1, fill=(0, 1, 0))
    pole = gz.rectangle(scale * 0.1, scale * 1.0, xy=(150, 55), stroke_width=1, fill=(1, 1, 0))

    sim = CartPole()
    time_steps = 0

    def make_frame(t):
        nonlocal time_steps

        observation = sim.get_scaled_state()
        observation = [observation[0], observation[2]]  # Remove velocities
        action = net.forward(observation)

        sim.step(force_function(action))

        time_steps += 1

        surface = gz.Surface(w, h, bg_color=(1, 1, 1))

        # Add state information
        text = gz.text(f'{time_steps} time steps', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 10))
        text.draw(surface)
        text = gz.text(f'x = {sim.x:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 20))
        text.draw(surface)
        text = gz.text(f'x_dot = {sim.dx:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 26))
        text.draw(surface)
        text = gz.text(f'theta = {sim.theta:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 32))
        text.draw(surface)
        text = gz.text(f'theta_dot = {sim.dtheta:.4f}', fontfamily="Impact", fontsize=6, fill=(0, 0, 0), xy=(35, 38))
        text.draw(surface)

        # Convert position to display units
        visX = scale * sim.x

        # Draw cart.
        group = gz.Group((cart,)).translate((visX, 0))
        group.draw(surface)

        # Draw pole.
        group = gz.Group((pole,)).translate((visX, 0)).rotate(sim.theta, center=(150 + visX, 80))
        group.draw(surface)

        return surface.get_npimage()

    clip = mpy.VideoClip(make_frame, duration=duration_seconds)
    clip.write_videofile(output_filename, codec="mpeg4", fps=50)
