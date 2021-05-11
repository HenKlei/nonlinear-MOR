import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import partial

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import create_uniform_grid
from tent_pitching.visualization import (plot_1d_space_time_grid, plot_space_function,
                                         plot_space_time_function, plot_on_reference_tent)
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin, RungeKutta4

from nonlinear_mor import NonlinearReductor


GLOBAL_SPACE_GRID_SIZE = 0.3333333
T_MAX = 1.
MAX_SPEED = 5.

LOCAL_SPACE_GRID_SIZE = 1e-1
LOCAL_TIME_GRID_SIZE = 1e-1


grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)


def characteristic_speed(x):
    return MAX_SPEED


space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000, log=True)

plot_1d_space_time_grid(space_time_grid, title='Space time grid obtained via tent pitching')


def linear_transport_flux(u, mu=1.):
    return mu * u


def linear_transport_flux_derivative(u, mu=1.):
    return mu


def inverse_transformation(u, phi_2, phi_2_dt, phi_2_dx, mu=1.):
    return u / (1. - phi_2_dx * mu)


def u_0_function(x, jump=True):
    if jump:
        return 1. * (x <= 0.25)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)


parameters = [0.25, 0.5, 0.75, 1.]
solutions = []

for mu in parameters:
    discretization = DiscontinuousGalerkin(partial(linear_transport_flux, mu=mu),
                                           partial(linear_transport_flux_derivative, mu=mu),
                                           partial(inverse_transformation, mu=mu),
                                           LOCAL_SPACE_GRID_SIZE, LOCAL_TIME_GRID_SIZE)

    grid_operator = GridOperator(space_time_grid, discretization, DGFunction,
                                 TimeStepperType=RungeKutta4,
                                 local_space_grid_size=LOCAL_SPACE_GRID_SIZE,
                                 local_time_grid_size=LOCAL_TIME_GRID_SIZE)

    u_0 = grid_operator.interpolate(u_0_function)

    u = grid_operator.solve(u_0)

#    plot_space_time_function(u, partial(inverse_transformation, mu=mu),
#                             title=f'Space time solution for mu={mu}',
#                             three_d=True, space_time_grid=space_time_grid)

#    plt.show()

    solutions.append((mu, u))

reductor = NonlinearReductor(space_time_grid)
reductor.reduce(solutions)
