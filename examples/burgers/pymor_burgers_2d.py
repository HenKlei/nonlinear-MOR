from typer import Option, run
import pickle
import numpy as np
import pathlib
import math

from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import discretize_instationary_fv, RectGrid

from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor as NonlinearReductor
from nonlinear_mor.models import WrappedpyMORModel


def create_fom(grid, nt):
    problem = burgers_problem_2d(vx=1., vy=1., initial_data_type='sin',
                                 parameter_range=(0, 1e42), torus=True)
    nx = grid
    print('Discretize ...')
    grid *= 1. / math.sqrt(2)
    model, _ = discretize_instationary_fv(
        problem,
        diameter=1. / grid,
        grid_type=RectGrid,
        num_flux='engquist_osher',
        lxf_lambda=1.,
        nt=nt
    )
    return WrappedpyMORModel(model, spatial_shape=(2*nx, nx), name='pyMOR Burgers 2d Model')


def main(grid: int = Option(60, help='Use grid with (2*NI)*NI elements.'),
         nt: int = Option(100, help='Number of time steps.'),
         N_train: int = Option(50, help='Number of training parameters'),
         reference_parameter: float = Option(1., help='Reference parameter'),
         alpha: float = Option(100., help='Alpha'),
         exponent: int = Option(2, help='Exponent'),
         sigma: float = Option(0.1, help='Sigma'),
         max_basis_size: int = Option(50, help='Maximum dimension of reduced basis'),
         restarts: int = Option(25, help='Maximum number of training restarts')):

    fom = create_fom(grid, nt)

    parameters = np.linspace(1., 2., N_train)

    gs_smoothing_params = {'alpha': alpha, 'exponent': exponent}
    registration_params = {'sigma': sigma}
    basis_sizes = range(1, max_basis_size+1)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_geodesic_shooting_2d_{timestr}'

    reductor = NonlinearReductor(fom, parameters, reference_parameter,
                                 gs_smoothing_params=gs_smoothing_params)
    reductor.write_summary(filepath_prefix=filepath_prefix, registration_params=registration_params,
                           additional_text="------------------\n" +
                                           f"Number of elements in x-direction: {2*grid}\n" +
                                           f"Number of elements in y-direction: {grid}\n" +
                                           f"Number of elements in t-direction: {nt}\n" +
                                           f"Reference parameter: {reference_parameter}\n" +
                                           f"Number of training parameters: {N_train}\n" +
                                           f"Maximum dimension of the reduced basis: {max_basis_size}\n" +
                                           f"Number of training restarts in neural network training: {restarts}")
    roms, output_dict = reductor.reduce(basis_sizes=basis_sizes, return_all=True, restarts=restarts,
                                        save_intermediate_results=False,
                                        registration_params=registration_params, filepath_prefix=filepath_prefix)

    outputs_filepath = f'{filepath_prefix}/outputs'
    pathlib.Path(outputs_filepath).mkdir(parents=True, exist_ok=True)
    with open(f'{outputs_filepath}/output_dict_rom', 'wb') as output_file:
        pickle.dump(output_dict, output_file)

    results_filepath = f'{filepath_prefix}/results'
    pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
    test_parameters = [1.2, 1.4, 1.6, 1.8]
    for basis_size in basis_sizes:
        results_filepath = f'{filepath_prefix}/results/basis_size_{basis_size}'
        pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)

        rom = roms[basis_size-1][0]
        for test_parameter in test_parameters:
            tic = time.perf_counter()
            u_red = rom.solve(test_parameter, filepath_prefix=filepath_prefix)
            time_rom = time.perf_counter() - tic
#            u_red.save(f'{results_filepath}/result_mu_{str(test_parameter).replace(".", "_")}.png')
            tic = time.perf_counter()
            u_full = fom.solve(test_parameter)
            time_fom = time.perf_counter() - tic
#            u_full.save(f'{results_filepath}/full_solution_mu_{str(test_parameter).replace(".", "_")}.png')
#            (u_red - u_full).save(f'{results_filepath}/difference_mu_{str(test_parameter).replace(".", "_")}.png')

            with open(f'{results_filepath}/relative_errors.txt', 'a') as errors_file:
                errors_file.write(f"{test_parameter}\t{(u_red - u_full).norm / u_full.norm}\t{time_fom}\t{time_rom}\n")


if __name__ == "__main__":
    run(main)
