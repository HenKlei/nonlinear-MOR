from typer import Option, run
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import pathlib

from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor as NonlinearReductor
from nonlinear_mor.models import WrappedpyMORModel

from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.domaindescriptions import LineDomain, CircleDomain
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.discretizers.builtin import discretize_instationary_fv


def burgers_problem(circle=False, parameter_range=(.25, 1.5)):
    """One-dimensional Burgers-type problem.

    The problem is to solve ::
        ∂_t u(x, t, μ)  +  ∂_x (v * u(x, t, μ)^μ) = 0
                                       u(x, 0, μ) = u_0(x)
    for u with t in [0, 0.3] and x in [0, 2].

    Parameters
    ----------
    circle
        If `True`, impose periodic boundary conditions. Otherwise Dirichlet left,
        outflow right.
    parameter_range
        The interval in which μ is allowed to vary.
    """

    initial_data = ExpressionFunction('0.25 * (x[0] < -.75) + 1. * (-.75 <= x[0]) * (x[0] <= -.25)', 1)
    dirichlet_data = ConstantFunction(dim_domain=1, value=.25)

    return InstationaryProblem(

        StationaryProblem(
            domain=CircleDomain([-1, 1]) if circle else LineDomain([-1, 1], right=None),

            dirichlet_data=dirichlet_data,

            rhs=None,

            nonlinear_advection=ExpressionFunction('v[0] * x**2 / 2.',
                                                   1, {'v': 1}),

            nonlinear_advection_derivative=ExpressionFunction('v[0] * x',
                                                              1, {'v': 1}),
        ),

        T=2.,

        initial_data=initial_data,

        parameter_ranges={'v': parameter_range},

        name=f"burgers_problem({circle})"
    )


def create_fom(nx, nt):
    problem = burgers_problem()
    model, _ = discretize_instationary_fv(
        problem,
        diameter=2 / nx,
        num_flux='engquist_osher',
        nt=nt
    )
    return WrappedpyMORModel(model, spatial_shape=(nx, ), name='pyMOR Burgers 1d Model')


def main(N_X: int = Option(100, help='Number of pixels in x-direction'),
         N_T: int = Option(150, help='Number of pixels in time-direction'),
         N_train: int = Option(50, help='Number of training parameters'),
         reference_parameter: float = Option(0.25, help='Reference parameter'),
         alpha: float = Option(100., help='Alpha'),
         exponent: int = Option(2, help='Exponent'),
         sigma: float = Option(0.1, help='Sigma'),
         max_basis_size: int = Option(50, help='Maximum dimension of reduced basis'),
         restarts: int = Option(25, help='Maximum number of training restarts'),
         full_velocity_fields_filepath_prefix: str = Option(None, help='Filepath prefix for full velocity fields file')):

    fom = create_fom(N_X, N_T)

    parameters = np.linspace(0.25, 1.5, N_train)

    gs_smoothing_params = {'alpha': alpha, 'exponent': exponent}
    registration_params = {'sigma': sigma}
    basis_sizes = range(1, max_basis_size+1)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_geodesic_shooting_{timestr}'

    if full_velocity_fields_filepath_prefix:
        full_velocity_fields_file = f'{full_velocity_fields_filepath_prefix}/outputs/full_velocity_fields'
    else:
        full_velocity_fields_file = None

    reductor = NonlinearReductor(fom, parameters, reference_parameter,
                                 gs_smoothing_params=gs_smoothing_params)
    reductor.write_summary(filepath_prefix=filepath_prefix, registration_params=registration_params,
                           additional_text="------------------\n" +
                                           f"Number of elements in x-direction: {N_X}\n" +
                                           f"Number of elements in t-direction: {N_T}\n" +
                                           f"Reference parameter: {reference_parameter}\n" +
                                           f"Number of training parameters: {N_train}\n" +
                                           f"Maximum dimension of the reduced basis: {max_basis_size}\n" +
                                           f"Number of training restarts in neural network training: {restarts}")
    roms, output_dict = reductor.reduce(basis_sizes=basis_sizes, return_all=True, restarts=restarts,
                                        full_velocity_fields_file=full_velocity_fields_file,
                                        registration_params=registration_params, filepath_prefix=filepath_prefix)

    outputs_filepath = f'{filepath_prefix}/outputs'
    pathlib.Path(outputs_filepath).mkdir(parents=True, exist_ok=True)
    with open(f'{outputs_filepath}/output_dict_rom', 'wb') as output_file:
        pickle.dump(output_dict, output_file)
    with open(f'{outputs_filepath}/full_velocity_fields', 'wb') as output_file:
        pickle.dump(output_dict['full_velocity_fields'], output_file)

    results_filepath = f'{filepath_prefix}/results'
    pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
    test_parameters = [0.5, 0.75, 1., 1.25]
    for basis_size in basis_sizes:
        results_filepath = f'{filepath_prefix}/results/basis_size_{basis_size}'
        pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
        pathlib.Path(results_filepath + '/figures_tex').mkdir(parents=True, exist_ok=True)

        rom = roms[basis_size-1][0]
        for test_parameter in test_parameters:
            tic = time.perf_counter()
            u_red = rom.solve(test_parameter, filepath_prefix=filepath_prefix)
            time_rom = time.perf_counter() - tic
            u_red.save(f'{results_filepath}/result_mu_{str(test_parameter).replace(".", "_")}.png')
            u_red.plot()
            tikzplotlib.save(f'{results_filepath}/figures_tex/result_mu_{str(test_parameter).replace(".", "_")}.tex')
            plt.close()

            tic = time.perf_counter()
            u_full = fom.solve(test_parameter)
            time_fom = time.perf_counter() - tic
            u_full.save(f'{results_filepath}/full_solution_mu_{str(test_parameter).replace(".", "_")}.png')
            u_full.plot()
            tikzplotlib.save(f'{results_filepath}/figures_tex/full_solution_mu_{str(test_parameter).replace(".", "_")}.tex')
            plt.close()
            (u_red - u_full).save(f'{results_filepath}/difference_mu_{str(test_parameter).replace(".", "_")}.png')
            (u_red - u_full).plot()
            tikzplotlib.save(f'{results_filepath}/figures_tex/difference_mu_{str(test_parameter).replace(".", "_")}.tex')
            plt.close()

            with open(f'{results_filepath}/relative_errors.txt', 'a') as errors_file:
                errors_file.write(f"{test_parameter}\t{(u_red - u_full).norm / u_full.norm}\t{time_fom}\t{time_rom}\n")


if __name__ == "__main__":
    run(main)
