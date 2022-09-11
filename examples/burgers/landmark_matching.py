from typer import Option, run
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.visualization import plot_landmark_matchings

from nonlinear_mor.models import AnalyticalModel
from nonlinear_mor.utils.versioning import get_git_hash, get_version


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


def get_landmarks(mu=0.25, all_landmarks=True):
    t_central_intersection = 0.25 / mu
    x_central_intersection = 0.625
    x_boundary_intersection = 1.
    t_boundary_intersection = 0.625 / mu
    if t_boundary_intersection > 1.:
        x_boundary_intersection = mu + 0.375
        t_boundary_intersection = 1.
    if all_landmarks:
        return np.array([[x_central_intersection, t_central_intersection],
                         [x_boundary_intersection, t_boundary_intersection],
                         [x_central_intersection + 0.125 * (x_boundary_intersection - x_central_intersection),
                          t_central_intersection + 0.125 * (t_boundary_intersection - t_central_intersection)],
                         [x_central_intersection + 0.25 * (x_boundary_intersection - x_central_intersection),
                          t_central_intersection + 0.25 * (t_boundary_intersection - t_central_intersection)],
                         [x_central_intersection + 0.375 * (x_boundary_intersection - x_central_intersection),
                          t_central_intersection + 0.375 * (t_boundary_intersection - t_central_intersection)],
                         [x_central_intersection + 0.5 * (x_boundary_intersection - x_central_intersection),
                          t_central_intersection + 0.5 * (t_boundary_intersection - t_central_intersection)],
                         [x_central_intersection + 0.625 * (x_boundary_intersection - x_central_intersection),
                          t_central_intersection + 0.625 * (t_boundary_intersection - t_central_intersection)],
                         [x_central_intersection + 0.75 * (x_boundary_intersection - x_central_intersection),
                          t_central_intersection + 0.75 * (t_boundary_intersection - t_central_intersection)],
                         [x_central_intersection + 0.875 * (x_boundary_intersection - x_central_intersection),
                          t_central_intersection + 0.875 * (t_boundary_intersection - t_central_intersection)],
                         [0.25, 0.],
                         [0.25 + 0.25 * (x_central_intersection - 0.25), 0.25 * t_central_intersection],
                         [0.25 + 0.5 * (x_central_intersection - 0.25), 0.5 * t_central_intersection],
                         [0.25 + 0.75 * (x_central_intersection - 0.25), 0.75 * t_central_intersection],
                         [0.5 + 0.25 * (x_central_intersection - 0.5), 0.25 * t_central_intersection],
                         [0.5 + 0.5 * (x_central_intersection - 0.5), 0.5 * t_central_intersection],
                         [0.5 + 0.75 * (x_central_intersection - 0.5), 0.75 * t_central_intersection],
                         [0.5, 0.]])
    else:
        return np.array([[x_central_intersection, t_central_intersection],
                         [x_boundary_intersection, t_boundary_intersection]])


def main(N_X: int = Option(100, help='Number of pixels in x-direction'),
         N_T: int = Option(100, help='Number of pixels in time-direction'),
         reference_parameter: float = Option(0.625, help='Reference parameter'),
         sigma: float = Option(0.1, help='Sigma'),
         kernel_sigma: float = Option(1., help='Sigma for kernel'),
         all_landmarks: bool = Option(True, help='If `True`, all landmarks are used, otherwise only two')):
    fom = create_fom(N_X, N_T)
    u_ref = fom.solve(reference_parameter)
    reference_landmarks = get_landmarks(mu=reference_parameter, all_landmarks=all_landmarks)

    parameters = [0.5, 0.75, 1., 1.25, 1.5]

    kwargs_kernel = {"sigma": kernel_sigma}
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel=kwargs_kernel, sampler_options={'order': 1})
    mins = np.zeros(2)
    maxs = np.ones(2)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_landmarks_nx_{N_X}_nt_{N_T}_{timestr}'

    import pathlib
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

    results = []

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
        spatial_shape = (N_X, N_T)
        flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta, reference_landmarks,
                                                            mins=mins, maxs=maxs, spatial_shape=spatial_shape)
        u = fom.solve(mu)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u.plot("Solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"{results_filepath}/solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        u_ref_transformed = u_ref.push_forward(flow)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u_ref_transformed.plot("Transformed reference solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"{results_filepath}/transformed_ref_solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u_ref.plot("Reference solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"{results_filepath}/reference_solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = (u - u_ref_transformed).plot("Difference", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"{results_filepath}/difference_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        u.save(f"{results_filepath}/solution_mu_{str(mu).replace('.', '_')}.png")
        u_ref_transformed.save(f"{results_filepath}/transformed_reference_solution_mu_{str(mu).replace('.', '_')}.png")
        (u - u_ref_transformed).save(f"{results_filepath}/difference_mu_{str(mu).replace('.', '_')}.png")

        rel_error = (u_ref_transformed - u).norm / u.norm
        print(f"Relative error for mu={mu}: {rel_error}")
        rest = np.s_[int(N_X*0.1):int(N_X*0.9), int(N_T*0.1):int(N_T*0.9)]
        rel_error_small = (u_ref_transformed - u).get_norm(restriction=rest) / u.get_norm(restriction=rest)
        print(f"Relative error on smaller domain for mu={mu}: {rel_error_small}")
        results.append((mu, rel_error))
        with open(f'{results_filepath}/relative_errors.txt', 'a') as errors_file:
            errors_file.write(f"{mu}\t{rel_error}\t{rel_error_small}\n")


if __name__ == "__main__":
    run(main)
