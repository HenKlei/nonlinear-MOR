import pickle
import numpy as np

import time

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import create_uniform_grid
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin, LaxFriedrichsFlux, RungeKutta4

from nonlinear_mor.reductors.neural_network import NonlinearNeuralNetworkReductor
from nonlinear_mor.models import SpacetimeModel
from nonlinear_mor.utils.io import write_errors_to_file


GLOBAL_SPACE_GRID_SIZE = 1.
T_MAX = 1.
MAX_SPEED = 1.

LOCAL_SPACE_GRID_SIZE = 1e-2
LOCAL_TIME_GRID_SIZE = 1e-2

N_X = 100
N_T = 100


def characteristic_speed(x):
    return MAX_SPEED


def burgers_flux(u, mu=1.):
    return mu * u**2 / 2.


def burgers_flux_derivative(u, mu=1.):
    return mu * u


def inverse_transformation(u, phi_2, phi_2_dt, phi_2_dx, mu=1.):
    return 2 * u / (1 + np.sqrt(1 - 2 * u * phi_2_dx * mu))


def u_0_function(x, jump=False):
    if jump:
        return 1. * (x <= 0.25)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)


grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)

space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000)

lambda_ = LOCAL_TIME_GRID_SIZE / LOCAL_SPACE_GRID_SIZE
numerical_flux = LaxFriedrichsFlux(burgers_flux, burgers_flux_derivative, lambda_)

discretization = DiscontinuousGalerkin(numerical_flux, inverse_transformation,
                                       LOCAL_SPACE_GRID_SIZE)

grid_operator = GridOperator(space_time_grid, discretization, DGFunction, u_0_function,
                             TimeStepperType=RungeKutta4,
                             local_space_grid_size=LOCAL_SPACE_GRID_SIZE,
                             local_time_grid_size=LOCAL_TIME_GRID_SIZE)

fom = SpacetimeModel(grid_operator, inverse_transformation, n_x=N_X, n_t=N_T)

parameters = []
reference_parameter = 1.

gs_smoothing_params = {'alpha': 100., 'exponent': 3}
registration_params = {'sigma': 0.1, 'epsilon': 0.1, 'iterations': 5000}
trainer_params = {'learning_rate': 0.001}
training_params = {'number_of_epochs': int(1e5)}
restarts = 100

NUM_WORKERS = 2
MAX_BASIS_SIZE = 10

reductor = NonlinearNeuralNetworkReductor(fom, parameters, reference_parameter,
                                          gs_smoothing_params=gs_smoothing_params)
start = time.perf_counter()
rom, output_dict = reductor.reduce(max_basis_size=MAX_BASIS_SIZE, return_all=True,
                                   restarts=restarts, registration_params=registration_params,
                                   trainer_params=trainer_params, training_params=training_params,
                                   num_workers=NUM_WORKERS,
                                   full_solutions_file='outputs/full_solutions',
                                   full_velocity_fields_file='outputs/full_velocity_fields')
time_for_reduction = time.perf_counter() - start

print(output_dict['training_data'])

output_dict.pop('full_velocity_fields', None)

singular_values = output_dict.pop('singular_values', '')
with open('outputs/singular_values', 'w') as output_file:
    for s in singular_values:
        output_file.write(str(s) + '\n')

best_loss = output_dict.pop('best_loss', '')

with open('outputs/output_dict_rom', 'wb') as output_file:
    pickle.dump(output_dict, output_file)

pod_size = output_dict['reduced_velocity_fields'].shape[1]

test_parameters = [0.5, 0.75]
write_errors_to_file('results/relative_errors.txt', time_for_reduction, gs_smoothing_params,
                     registration_params, pod_size, singular_values, restarts, trainer_params,
                     training_params, best_loss, test_parameters, rom, fom)
