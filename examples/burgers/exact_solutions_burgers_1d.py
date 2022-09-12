from typer import Option, run
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor as NonlinearReductor
from nonlinear_mor.models import AnalyticalModel


def exact_solution(x, *, mu=0.25):
    s_l = 1.5 * mu
    s_m = mu
    s_r = 0.5 * mu
    t_intersection = 0.25 / (s_l - s_r)
    return ScalarFunction(data=(2. * (x[..., 1] <= t_intersection) * (0.25 + s_l * x[..., 1] - x[..., 0] >= 0.)
                                + (2. * (x[..., 1] > t_intersection)
                                   * (0.25 + (s_l - s_m) * t_intersection + s_m * x[..., 1] - x[..., 0] >= 0.))
                                + (1. * (0.25 + s_l * x[..., 1] - x[..., 0] < 0.)
                                   * (0.5 + s_r * x[..., 1] - x[..., 0] > 0.))))


def create_fom(N_X, N_T):
    return AnalyticalModel(exact_solution, n_x=N_X, n_t=N_T, name='Analytical Burgers Model')


def main(N_X: int = Option(100, help='Number of pixels in x-direction'),
         N_T: int = Option(100, help='Number of pixels in time-direction'),
         N_train: int = Option(50, help='Number of training parameters'),
         reference_parameter: float = Option(0.25, help='Reference parameter'),
         alpha: float = Option(100., help='Alpha'),
         exponent: int = Option(2, help='Exponent'),
         sigma: float = Option(0.1, help='Sigma'),
         max_basis_size: int = Option(50, help='Maximum dimension of reduced basis'),
         restarts: int = Option(25, help='Maximum number of training restarts')):

    fom = create_fom(N_X, N_T)

    parameters = np.linspace(0.25, 1.5, N_train)

    gs_smoothing_params = {'alpha': alpha, 'exponent': exponent}
    registration_params = {'sigma': sigma}
    basis_sizes = range(1, max_basis_size+1)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_geodesic_shooting_{timestr}'

    reductor = NonlinearReductor(fom, parameters, reference_parameter,
                                 gs_smoothing_params=gs_smoothing_params)
    roms, output_dict = reductor.reduce(basis_sizes=basis_sizes, return_all=True, restarts=restarts,
                                        registration_params=registration_params, filepath_prefix=filepath_prefix)
    reductor.write_summary(filepath_prefix=filepath_prefix, registration_params=registration_params,
                           additional_text="------------------\n" +
                                           f"Number of elements in x-direction: {N_X}\n" +
                                           f"Number of elements in t-direction: {N_T}\n" +
                                           f"Reference parameter: {reference_parameter}\n" +
                                           f"Number of training parameters: {N_train}\n" +
                                           f"Maximum dimension of the reduced basis: {max_basis_size}\n" +
                                           f"Number of training restarts in neural network training: {restarts}")

    outputs_filepath = f'{filepath_prefix}/outputs'
    pathlib.Path(outputs_filepath).mkdir(parents=True, exist_ok=True)
    with open(f'{outputs_filepath}/output_dict_rom', 'wb') as output_file:
        pickle.dump(output_dict, output_file)

    results_filepath = f'{filepath_prefix}/results'
    pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
    test_parameters = [0.5, 0.75, 1., 1.25]
    for basis_size in basis_sizes:
        results_filepath = f'{filepath_prefix}/results/basis_size_{basis_size}'
        pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)

        rom = roms[basis_size-1][0]
        for test_parameter in test_parameters:
            tic = time.perf_counter()
            u_red = rom.solve(test_parameter, filepath_prefix=filepath_prefix)
            time_rom = time.perf_counter() - tic
            u_red.save(f'{results_filepath}/result_mu_{str(test_parameter).replace(".", "_")}.png')
            tic = time.perf_counter()
            u_full = fom.solve(test_parameter)
            time_fom = time.perf_counter() - tic
            u_full.save(f'{results_filepath}/full_solution_mu_{str(test_parameter).replace(".", "_")}.png')
            (u_red - u_full).save(f'{results_filepath}/difference_mu_{str(test_parameter).replace(".", "_")}.png')

            with open(f'{results_filepath}/relative_errors.txt', 'a') as errors_file:
                errors_file.write(f"{test_parameter}\t{(u_red - u_full).norm / u_full.norm}\t{time_fom}\t{time_rom}\n")


if __name__ == "__main__":
    run(main)
