from typer import Option, run
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import pathlib

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.reductors import GreedyDictionaryReductor as NonlinearReductor
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
         N_train: int = Option(20, help='Number of training parameters'),
         tol: float = Option(5e-2, help='Greedy tolerance'),
         alpha: float = Option(200., help='Alpha'),
         exponent: int = Option(2, help='Exponent'),
         sigma: float = Option(0.1, help='Sigma'),
         max_basis_size: int = Option(5, help='Maximum dimension of reduced basis'),
         restarts: int = Option(25, help='Maximum number of training restarts'),
         full_velocity_fields_filepath_prefix: str = Option(None, help='Filepath prefix for full velocity fields file')):

    fom = create_fom(N_X, N_T)

    parameters = np.linspace(0.25, 1.5, N_train)

    gs_smoothing_params = {'alpha': alpha, 'exponent': exponent}
    registration_params = {'sigma': sigma}

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_geodesic_shooting_{timestr}'

    geodesic_shooter = geodesic_shooting.GeodesicShooting(**gs_smoothing_params)
    reductor = NonlinearReductor(fom, parameters, tol, geodesic_shooter)
    reductor.write_summary(filepath_prefix=filepath_prefix, registration_params=registration_params,
                           additional_text="------------------\n" +
                                           f"Number of elements in x-direction: {N_X}\n" +
                                           f"Number of elements in t-direction: {N_T}\n" +
                                           f"Number of training parameters: {N_train}\n" +
                                           f"Tolerance used in Greedy: {tol}\n" +
                                           f"Maximum dimension of the reduced basis: {max_basis_size}\n" +
                                           f"Number of training restarts in neural network training: {restarts}")
    rom = reductor.reduce(max_basis_size=max_basis_size, l2_prod=False, registration_params=registration_params,
                          save_intermediate_results=True, filepath_prefix=filepath_prefix)

    results_filepath = f'{filepath_prefix}/results'
    pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
    test_parameters = [0.5, 0.75, 1., 1.25]

    results_filepath = f'{filepath_prefix}/results/basis_size_{max_basis_size}'
    pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
    pathlib.Path(results_filepath + '/figures_tex').mkdir(parents=True, exist_ok=True)

    for test_parameter in test_parameters:
        tic = time.perf_counter()
        u_red = rom.solve(test_parameter)
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
