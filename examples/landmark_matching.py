import ast
import pathlib
import dill as pickle
import time
from typer import Argument, Option, run
from typing import List
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.visualization import plot_landmark_matchings

from nonlinear_mor.utils.versioning import get_git_hash, get_version
from load_model import load_full_order_model, load_landmark_function


def main(example: str = Argument(..., help='Path to the example to execute, for instance '
                                           'example="1d.burgers.piecewise_constant.burgers_1d_landmarks_analytical"'),
         spatial_shape: List[int] = Argument(..., help='Number of unknowns in the spatial coordinate directions'),
         num_time_steps: int = Option(100, help='Number of time steps in the high-fidelity solutions'),
         additional_parameters: str = Option('{}', help='Additional parameters to pass to the full-order model',
                                             callback=ast.literal_eval),
         num_test_parameters: int = Option(50, help='Number of test parameters'),
         sampling_mode: str = Option('uniform', help='Sampling mode for sampling the training parameters'),
         reference_parameter: str = Option('0.625', help='Reference parameter, either a number or a list of numbers',
                                           callback=ast.literal_eval),
         sigma: float = Option(0.1, help='Registration parameter `sigma`'),
         kernel_sigma: float = Option(1., help='Kernel parameter `sigma`'),
         all_landmarks: bool = Option(True, help='If `True`, all landmarks are used, otherwise only a certain subset'
                                                 '(might not be supported by every model)'),
         write_results: bool = Option(True, help='Determines whether or not to write results to disc (useful during '
                                                 'development)')):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_landmark_matching_{timestr}'
    if write_results:
        pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)

    spatial_shape = tuple(spatial_shape)
    fom = load_full_order_model(example, spatial_shape, num_time_steps, additional_parameters)
    get_landmarks = load_landmark_function(example)

    if write_results:
        full_order_model_filepath = f'{filepath_prefix}/full_order_model'
        pathlib.Path(full_order_model_filepath).mkdir(parents=True, exist_ok=True)
        fom_dictionary = {'example': example, 'spatial_shape': spatial_shape,
                          'num_time_steps': num_time_steps, 'additional_parameters': additional_parameters}
        with open(f'{full_order_model_filepath}/model.pickle', 'wb') as fom_file:
            pickle.dump(fom_dictionary, fom_file)

    u_ref = fom.solve(reference_parameter)
    reference_landmarks = get_landmarks(mu=reference_parameter, all_landmarks=all_landmarks)

    parameters = fom.parameter_space.sample(num_test_parameters, sampling_mode)

    kwargs_kernel = {"sigma": kernel_sigma}
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel=kwargs_kernel, sampler_options={'order': 1})
    mins = np.zeros(u_ref.dim)
    maxs = np.ones(u_ref.dim)

    if write_results:
        results_filepath = f'{filepath_prefix}/results'
        pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
        with open(f'{filepath_prefix}/summary.txt', 'a') as summary_file:
            summary_file.write('========================================================\n')
            summary_file.write('Git hash: ' + get_git_hash() + '\n')
            summary_file.write('========================================================\n')
            summary_file.write('FOM: ' + str(fom) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Reference parameter: ' + str(reference_parameter) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Geodesic Shooting:\n')
            summary_file.write('------------------\n')
            summary_file.write('Version: ' + get_version(geodesic_shooting) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Sigma in landmark shooting: ' + str(sigma) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Sigma of kernel: ' + str(kernel_sigma) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Reference landmarks (' + str(reference_landmarks.shape[0]) + ' in total):\n')
            for reference_landmark in reference_landmarks:
                summary_file.write(str(reference_landmark) + '\n')

    for mu in parameters:
        target_landmarks = get_landmarks(mu=mu, all_landmarks=all_landmarks)
        initial_momenta = np.zeros(reference_landmarks.shape)
        result = gs.register(reference_landmarks, target_landmarks, initial_momenta=initial_momenta, sigma=sigma,
                             return_all=True)
        registered_landmarks = result['registered_landmarks']

        def compute_average_distance(target_landmarks, registered_landmarks):
            dist = 0.
            for x, y in zip(registered_landmarks, target_landmarks):
                dist += np.linalg.norm(x - y)
            dist /= registered_landmarks.shape[0]
            return dist

        print(f"Norm of difference: {compute_average_distance(target_landmarks, registered_landmarks)}")

        initial_momenta = result['initial_momenta']
        flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta, reference_landmarks,
                                                            mins=mins, maxs=maxs, spatial_shape=u_ref.full_shape)
        u = fom.solve(mu)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u.plot("Solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        if write_results:
            fig.savefig(f"{results_filepath}/solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        u_ref_transformed = u_ref.push_forward(flow)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u_ref_transformed.plot("Transformed reference solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        if write_results:
            fig.savefig(f"{results_filepath}/transformed_ref_solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u_ref.plot("Reference solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        if write_results:
            fig.savefig(f"{results_filepath}/reference_solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = (u - u_ref_transformed).plot("Difference", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        if write_results:
            fig.savefig(f"{results_filepath}/difference_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        if write_results:
            u.save(f"{results_filepath}/solution_mu_{str(mu).replace('.', '_')}.png")
            u_ref_transformed.save(f"{results_filepath}/transformed_reference_solution_mu_{str(mu).replace('.', '_')}.png")
            (u - u_ref_transformed).save(f"{results_filepath}/difference_mu_{str(mu).replace('.', '_')}.png")


if __name__ == "__main__":
    run(main)
