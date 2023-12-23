"""
Simple gravity pendulum with air resistance taken into account
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from matplotlib.animation import FuncAnimation

# Shortcuts for type hinting
num = int | float
vec = list | np.ndarray


def rk4(diff_eq: Callable, y0: vec, trange: np.ndarray, *args) -> np.ndarray:
    """
    Solves a system of first-order differential equations using the
    classic Runge-Kutta method
    :param diff_eq: Function which returns the righthandside of the
        equations making up the system of equations
    :param y0: Initial values for y and y' (i.e. the terms in the system
        of equations)
    :param trange: Time points for which the equation is solved
    :param args: Any additional paramaters for the differential equation
    :return: A m x n size matrix, where m is the amount of equations
        and n is the amount of timesteps. Contains values for each equation
        at each timestep.
    """
    m = trange.size
    dt = trange[1] - trange[0]
    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)
    n = y0.size
    sol = np.zeros((m, n))
    sol[0, :] = y0
    for i, t in enumerate(trange[1:], start=1):
        y = sol[i - 1, :]
        k1 = diff_eq(y, t, *args)
        k2 = diff_eq(y + dt * k1 / 2, t + dt / 2, *args)
        k3 = diff_eq(y + dt * k2 / 2, t + dt / 2, *args)
        k4 = diff_eq(y + dt * k3, t + dt, *args)
        y += 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        sol[i, :] = y
    return sol


class SinglePendulum:
    def __init__(self, init_conds: vec, k: num, m: num, l_len: num,
                 g: num, tspan: vec) -> None:
        """
        :param init_conds: Initial angle and velocity
        :param k: Constant parameeter for air resistance
        :param m: Mass of the pendulum, or the swinging weight
        :param l_len: Length of the line connecting the weight
        :param g: Gravitational acceleration
        :param tspan: Timesteps
        """
        self.init_conds = init_conds
        self.k = k
        self.m = m
        self.l_len = l_len
        self.g = g
        self.tspan = tspan
        self.sol = (None, None)

    @staticmethod
    def _pend(init_conds: list, _: vec, k: num, m: num, l_len: num,
              g: num) -> np.ndarray:
        """
        Differential equation of the pendulum
        :param init_conds: Angle and velocity
        :param _: Necessary, but unused, parameter indicating the
            timestep
        :param k: Value for the air resistance
        :param m: Mass of the swinging weight
        :param l_len: Length of the line
        :param g: Gravitational acceleration
        :return:
        """
        theta, omega = init_conds
        dydt = [omega, k / (m * l_len) * omega + (g / l_len) * np.sin(theta)]
        return np.array(dydt)

    def _solve(self) -> None:
        """
        :return:
        """
        self.sol = rk4(self._pend, self.init_conds, self.tspan,
                       self.k, self.m, self.l_len, self.g)

    def _create_graphs(self) -> None:
        """
        Graphs of theta(t) and omega(t)
        :return:
        """
        fig_graph, ax_graph = plt.subplots()
        ax_graph.grid()
        self._solve()
        plt.plot(self.tspan, self.sol[:, 0], label="theta(t)")
        plt.plot(self.tspan, self.sol[:, 1], label="omega(t)")
        plt.legend()

    def _create_plots(self) -> tuple:
        """
        :return:
        """
        fig_anim, ax_anim = plt.subplots()
        plt.grid()
        plt.xlim([-self.l_len - 0.25, self.l_len + 0.25])
        plt.ylim([-self.l_len - 0.25, self.l_len + 0.25])
        x_init = -self.l_len * np.sin(self.sol[:, 0][0])
        y_init = -self.l_len * -np.cos(self.sol[:, 0][0])
        line = plt.plot([0, x_init], [0, y_init])[0]
        ball = plt.plot([x_init], [y_init], "ro")[0]
        cross = plt.plot([0], [0], "kx")[0]
        plot_objs = (line, ball, cross)
        return fig_anim, plot_objs

    def _update_plots(self, n, plots, sol_coords):
        """
        Updates the plot objects with the new data
        :param n:
        :param plots:
        :param sol_coords:
        :return:
        """
        x = -self.l_len * np.sin(sol_coords[:, 0][n])
        y = -self.l_len * -np.cos(sol_coords[:, 0][n])
        for i in range(len(plots) - 1):
            if i == 0:
                plots[i].set_data([0, x], [0, y])
            elif i == 1:
                plots[i].set_data([x], [y])

        return plots

    def animate(self) -> None:
        """
        :return:
        """
        self._create_graphs()
        fig_anim, plot_objs = self._create_plots()
        anim = FuncAnimation(fig_anim, self._update_plots, self.tspan.size,
                             fargs=(plot_objs, self.sol), interval=10, blit=True)
        anim.save("pend_first.gif", writer="pillow")
        plt.show()


def main():
    # Initial conditions and constants
    g, k = 9.81, -0.9
    m, l_len = 2, 2
    initial_conds = [0.01, 0]  # [angle, velocity]
    start, end, timestep = 0, 30, 0.1  # In seconds
    tspan = np.linspace(start, end, int(end / timestep))
    pend = SinglePendulum(initial_conds, k, m, l_len, g, tspan)
    pend.animate()


if __name__ == "__main__":
    main()
