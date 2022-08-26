from typer import Option, run
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.visualization import (animate_landmark_trajectories,
                                                   plot_initial_momenta_and_landmarks,
                                                   plot_landmark_trajectories,
                                                   plot_landmark_matchings, animate_warpgrids)

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


def get_intersection_points(mu=0.25):
    t_central_intersection = 0.25 / mu
    x_central_intersection = 0.625
    x_boundary_intersection = 1.
    t_boundary_intersection = 0.625 / mu
    if t_boundary_intersection > 1.:
        x_boundary_intersection = mu + 0.375
        t_boundary_intersection = 1.
    return np.array([[x_central_intersection, t_central_intersection],
                     [x_boundary_intersection, t_boundary_intersection],#])
                     [x_central_intersection + 0.125 * (x_boundary_intersection - x_central_intersection), t_central_intersection + 0.125 * (t_boundary_intersection - t_central_intersection)],
                     [x_central_intersection + 0.25 * (x_boundary_intersection - x_central_intersection), t_central_intersection + 0.25 * (t_boundary_intersection - t_central_intersection)],
                     [x_central_intersection + 0.375 * (x_boundary_intersection - x_central_intersection), t_central_intersection + 0.375 * (t_boundary_intersection - t_central_intersection)],
                     [x_central_intersection + 0.5 * (x_boundary_intersection - x_central_intersection), t_central_intersection + 0.5 * (t_boundary_intersection - t_central_intersection)],
                     [x_central_intersection + 0.625 * (x_boundary_intersection - x_central_intersection), t_central_intersection + 0.625 * (t_boundary_intersection - t_central_intersection)],
                     [x_central_intersection + 0.75 * (x_boundary_intersection - x_central_intersection), t_central_intersection + 0.75 * (t_boundary_intersection - t_central_intersection)],
                     [x_central_intersection + 0.875 * (x_boundary_intersection - x_central_intersection), t_central_intersection + 0.875 * (t_boundary_intersection - t_central_intersection)],
                     [0.25, 0.],
                     [0.25 + 0.25 * (x_central_intersection - 0.25), 0.25 * t_central_intersection],
                     [0.25 + 0.5 * (x_central_intersection - 0.25), 0.5 * t_central_intersection],
                     [0.25 + 0.75 * (x_central_intersection - 0.25), 0.75 * t_central_intersection],
                     [0.5 + 0.25 * (x_central_intersection - 0.5), 0.25 * t_central_intersection],
                     [0.5 + 0.5 * (x_central_intersection - 0.5), 0.5 * t_central_intersection],
                     [0.5 + 0.75 * (x_central_intersection - 0.5), 0.75 * t_central_intersection],
                     [0.5, 0.]])


def main(N_X: int = Option(100, help='Number of pixels in x-direction'),
         N_T: int = Option(100, help='Number of pixels in time-direction'),
         N_train: int = Option(10, help='Number of training parameters'),
         reference_parameter: float = Option(0.625, help='Reference parameter'),
         sigma: float = Option(0.1, help='Sigma')):
    fom = create_fom(N_X, N_T)
    u_ref = fom.solve(reference_parameter)
    reference_landmarks = get_intersection_points(mu=reference_parameter)

    parameters = [0.75, 1., 1.25, 1.5]

    kwargs_kernel = {"sigma": 5.}
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel=kwargs_kernel, sampler_options={'order': 1})
    mins = np.zeros(2)
    maxs = np.ones(2)

    results = []

    for mu in parameters:
        target_landmarks = get_intersection_points(mu=mu)
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
#        flow.plot("Flow")
        u = fom.solve(mu)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u.plot("Solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"results_landmark_matching/solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        u_ref_transformed = u_ref.push_forward(flow)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u_ref_transformed.plot("Transformed reference solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"results_landmark_matching/transformed_reference_solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = u_ref.plot("Reference solution", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"results_landmark_matching/reference_solution_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis, vals = (u - u_ref_transformed).plot("Difference", axis=axis)
        fig.colorbar(vals, ax=axis)
        plot_landmark_matchings(reference_landmarks, target_landmarks, registered_landmarks, axis=axis)
        fig.savefig(f"results_landmark_matching/difference_with_landmarks_mu_{str(mu).replace('.', '_')}.png")
        plt.close(fig)

        vf = gs.get_vector_field(initial_momenta, reference_landmarks)
        vf.plot(interval=5, scale=1.)
        time_evolution_momenta = result['time_evolution_momenta']
        time_evolution_positions = result['time_evolution_positions']
        plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                                   min_x=mins[0], max_x=maxs[0], min_y=mins[1], max_y=maxs[1])
        flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta, reference_landmarks,
                                                            mins=mins, maxs=maxs, spatial_shape=spatial_shape)
        flow.plot("Flow")
        plt.show()

        u.plot(f"Full solution for mu={mu}")
        u_ref_transformed.plot(f"Transformed reference solution for mu={mu}")
        (u - u_ref_transformed).plot(f"Difference for mu={mu}")
        plt.show()

        rel_error = (u_ref_transformed - u).norm / u.norm
        print(f"Relative error for mu={mu}: {rel_error}")
        results.append((mu, rel_error))

    for mu, err in results:
        print(f"Parameter: {mu}; relative error: {err}")


if __name__ == "__main__":
    run(main)
