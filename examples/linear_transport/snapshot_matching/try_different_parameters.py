import pickle
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.io import save_image
from geodesic_shooting.utils.visualization import plot_warpgrid, plot_vector_field


if __name__ == "__main__":
    with open("full_solution", 'rb') as input_file:
        full_solution = pickle.load(input_file)

    with open("reference_solution", 'rb') as input_file:
        reference_solution = pickle.load(input_file)

    alphas = [1, 10, 100, 1000]
    exponents = [1, 3, 5]
    sigmas = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
    epsilons = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]

    # perform the registration
    for alpha in alphas:
        for exponent in exponents:
            for sigma in sigmas:
                for epsilon in epsilons:
                    gs = geodesic_shooting.GeodesicShooting(alpha=alpha, exponent=exponent)
                    image, _, _, _, _ = gs.register(reference_solution, full_solution,
                                                    sigma=sigma, epsilon=epsilon, iterations=5000,
                                                    return_all=True)

                    norm = (np.linalg.norm((full_solution - image).flatten())
                            / np.linalg.norm(full_solution.flatten()))
                    with open('relative_errors.txt', 'a') as errors_file:
                        errors_file.write(f"{alpha}\t{exponent}\t{sigma}\t{epsilon}\t{norm}\n")
