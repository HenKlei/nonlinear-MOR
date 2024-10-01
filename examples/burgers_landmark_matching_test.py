import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting

from geodesic_shooting.utils.visualization import (plot_initial_momenta_and_landmarks,
                                                   plot_landmark_matchings,
                                                   plot_landmark_trajectories)

from automatic_landmark_detection import place_landmarks_on_edges
from load_model import load_full_order_model, load_landmark_function


if __name__ == "__main__":
    get_landmarks = load_landmark_function("1d.burgers.piecewise_constant.burgers_landmarks_analytical")
    mu_in = 0.75
    mu_tar = 1.5
    place_landmarks_automatically = True
    nx = 200
    ny = 200
    mins = np.array([0., 0.])
    maxs = np.array([1., 1.])
    spatial_shape = (nx, ny)

    fom = load_full_order_model("1d.burgers.piecewise_constant.burgers_landmarks_analytical", (nx, ), ny, {})
    u_in = fom.solve(mu_in)
    u_tar = fom.solve(mu_tar)
    if place_landmarks_automatically:
        num_landmarks = 20
        input_landmarks = place_landmarks_on_edges(u_in.to_numpy().T, num_landmarks)
        target_landmarks = place_landmarks_on_edges(u_tar.to_numpy().T, num_landmarks)
    else:
        input_landmarks = get_landmarks(mu=mu_in, many_landmarks=False)
        target_landmarks = get_landmarks(mu=mu_tar, many_landmarks=False)

    landmarks_labeled = False
    # perform the registration using landmark shooting algorithm
    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={"sigma": 6.})
    result = gs.register(input_landmarks, target_landmarks, sigma=0.1, return_all=True,
                         landmarks_labeled=landmarks_labeled, kwargs_kernel_dist={"sigma": 4.})
    initial_momenta = result['initial_momenta']
    registered_landmarks = result['registered_landmarks']

    # plot results
    plot_landmark_matchings(input_landmarks, target_landmarks, registered_landmarks,
                            landmarks_labeled=landmarks_labeled)

    plot_initial_momenta_and_landmarks(initial_momenta, registered_landmarks,
                                       min_x=0., max_x=1., min_y=0., max_y=1.)

    time_evolution_momenta = result['time_evolution_momenta']
    time_evolution_positions = result['time_evolution_positions']
    plot_landmark_trajectories(time_evolution_momenta, time_evolution_positions,
                               min_x=0., max_x=1., min_y=0., max_y=1.)

    flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta, input_landmarks,
                                                        mins=mins, maxs=maxs, spatial_shape=spatial_shape,
                                                        get_time_dependent_diffeomorphism=True)
    flow.plot("Flow")

    vf = gs.get_vector_field(initial_momenta, input_landmarks, mins=mins, maxs=maxs, spatial_shape=spatial_shape)
    vf.save_tikz("test_folder/vf.tex", interval=5)
    vf.plot(color_length=True)

    u_in.plot("Original image")
    u_tar.plot("Target image")
    resulting_image = u_in.push_forward(flow[-1])
    resulting_image.plot("Transformed image")
    (u_tar - resulting_image).plot("Difference")
    plt.show()

    print(f"Error in image matching: {(u_tar - resulting_image).norm / u_tar.norm}")
