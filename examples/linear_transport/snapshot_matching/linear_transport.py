import pickle
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.visualization import plot_vector_field


if __name__ == "__main__":
    with open("full_solution", 'rb') as input_file:
        full_solution = pickle.load(input_file)

    with open("reference_solution", 'rb') as input_file:
        reference_solution = pickle.load(input_file)

    # perform the registration
    geodesic_shooting = geodesic_shooting.GeodesicShooting(alpha=1000., exponent=3.)
    image, v0, energies, Phi0, length = geodesic_shooting.register(reference_solution,
                                                                   full_solution, sigma=0.1,
                                                                   epsilon=0.01, iterations=5000,
                                                                   return_all=True)

    norm = (np.linalg.norm((full_solution - image).flatten())
            / np.linalg.norm(full_solution.flatten()))
    print(f'Relative norm of difference: {norm}')

    plot_vector_field(v0)

    plt.matshow(reference_solution)
    plt.title("Input")
    plt.matshow(full_solution)
    plt.title("Target")
    plt.matshow(image)
    plt.title("Result")
    plt.show()
