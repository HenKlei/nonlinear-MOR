import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (plot_initial_momenta_and_landmarks,
                                                   plot_landmark_matchings,
                                                   plot_landmark_trajectories)
from geodesic_shooting.core import ScalarFunction


if __name__ == "__main__":
    from load_model import load_landmark_function
    get_landmarks = load_landmark_function("1d.burgers.piecewise_constant.burgers_landmarks_analytical")
    mu_in = 0.75
    mu_tar = 1.5
    input_landmarks = get_landmarks(mu=mu_in, many_landmarks=False)
    target_landmarks = get_landmarks(mu=mu_tar, many_landmarks=False)

    # perform the registration using landmark shooting algorithm
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={"sigma": 4.})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.1, return_all=True, landmarks_labeled=False)
    initial_momenta = result['initial_momenta']
    registered_landmarks = result['registered_landmarks']

    # plot results
    plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks)

    plot_initial_momenta_and_landmarks(initial_momenta, registered_landmarks,
                                       min_x=0., max_x=1., min_y=0., max_y=1.)

    time_evolution_momenta = result['time_evolution_momenta']
    time_evolution_positions = result['time_evolution_positions']
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                               min_x=0., max_x=1., min_y=0., max_y=1.)

    nx = 250
    ny = 250
    mins = np.array([0., 0.])
    maxs = np.array([1., 1.])
    spatial_shape = (nx, ny)

    flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta, input_landmarks,
                                                        mins=mins, maxs=maxs, spatial_shape=spatial_shape,
                                                        get_time_dependent_diffeomorphism=True)
    flow.plot("Flow")

    vf = gs.get_vector_field(initial_momenta, input_landmarks, mins=mins, maxs=maxs, spatial_shape=spatial_shape)
    vf.save_tikz("test_folder/vf.tex", interval=5)
    vf.plot(color_length=True)

    def exact_solution(x, *, mu=1.):
        s_l = 1.5 * mu
        s_m = mu
        s_r = 0.5 * mu
        t_intersection = 0.25 / (s_l - s_r)
        return ScalarFunction(data=(2. * (x[..., 1] <= t_intersection) * (0.25 + s_l * x[..., 1] - x[..., 0] >= 0.)
                                    + (2. * (x[..., 1] > t_intersection)
                                       * (0.25 + (s_l - s_m) * t_intersection + s_m * x[..., 1] - x[..., 0] >= 0.))
                                    + (1. * (0.25 + s_l * x[..., 1] - x[..., 0] < 0.)
                                       * (0.5 + s_r * x[..., 1] - x[..., 0] > 0.))))

    linspace_list = [np.linspace(s_min, s_max, num) for (s_min, s_max), num in zip([(0., 1.)],
                                                                                   spatial_shape)]
    temp = np.meshgrid(*linspace_list, np.linspace(0., 1., ny))
    coordinates = np.stack([temp_.T for temp_ in temp], axis=-1)
    coordinates_list = coordinates.reshape((-1, 2))

    image = exact_solution(coordinates, mu=mu_in)
    target_image = exact_solution(coordinates, mu=mu_tar)

    image.plot("Original image")
    target_image.plot("Target image")
    resulting_image = image.push_forward(flow[-1])
    resulting_image.plot("Transformed image")
    (target_image - resulting_image).plot("Difference")
    plt.show()

    print(f"Error in image matching: {(target_image - resulting_image).norm / target_image.norm}")
