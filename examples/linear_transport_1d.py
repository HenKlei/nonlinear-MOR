import numpy as np
import matplotlib.pyplot as plt

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import create_uniform_grid
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin, LaxFriedrichsFlux, RungeKutta4

from nonlinear_mor.reductors import NonlinearReductor
from nonlinear_mor.models import SpacetimeModel


GLOBAL_SPACE_GRID_SIZE = 1.
T_MAX = 1.
MAX_SPEED = 1.

LOCAL_SPACE_GRID_SIZE = 1e-2
LOCAL_TIME_GRID_SIZE = 1e-2

N_X = 100
N_T = 100


def characteristic_speed(x):
    return MAX_SPEED


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


grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)

space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000)

lambda_ = LOCAL_TIME_GRID_SIZE / LOCAL_SPACE_GRID_SIZE
numerical_flux = LaxFriedrichsFlux(linear_transport_flux, linear_transport_flux_derivative,
                                   lambda_)

discretization = DiscontinuousGalerkin(numerical_flux, inverse_transformation,
                                       LOCAL_SPACE_GRID_SIZE)

grid_operator = GridOperator(space_time_grid, discretization, DGFunction, u_0_function,
                             TimeStepperType=RungeKutta4,
                             local_space_grid_size=LOCAL_SPACE_GRID_SIZE,
                             local_time_grid_size=LOCAL_TIME_GRID_SIZE)

fom = SpacetimeModel(grid_operator, inverse_transformation, n_x=N_X, n_t=N_T)

N_train = 20
parameters = np.linspace(0.25, 1., N_train)
reference_parameter = 1.

registration_params = {'sigma': 0.1, 'epsilon': 0.1, 'iterations': 100}

reductor = NonlinearReductor(fom, parameters, reference_parameter)
rom = reductor.reduce(registration_params=registration_params)

test_parameter = 0.5
u_red = rom.solve(test_parameter)
plt.matshow(u_red)
plt.show()
