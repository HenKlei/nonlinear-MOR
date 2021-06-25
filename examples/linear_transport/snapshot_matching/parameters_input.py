import pickle
import numpy as np
from typing import List, Optional
from typer import Option, run

import geodesic_shooting


def main(alphas: Optional[List[float]] = Option([1., ], help=''),
         exponents: Optional[List[int]] = Option([1., ], help=''),
         sigmas: Optional[List[float]] = Option([0.1, ], help=''),
         epsilons: Optional[List[float]] = Option([0.1, ], help='')):

    with open("full_solution", 'rb') as input_file:
        full_solution = pickle.load(input_file)

    with open("reference_solution", 'rb') as input_file:
        reference_solution = pickle.load(input_file)

    # perform the registration
    for alpha in alphas:
        for exponent in exponents:
            for sigma in sigmas:
                for epsilon in epsilons:
                    gs = geodesic_shooting.GeodesicShooting(alpha=alpha, exponent=exponent)
                    result = gs.register(reference_solution, full_solution, sigma=sigma,
                                         epsilon=epsilon, iterations=5000, return_all=True)

                    transformed_input = result['transformed_input']

                    norm = (np.linalg.norm((full_solution - transformed_input).flatten())
                            / np.linalg.norm(full_solution.flatten()))
                    with open('relative_errors.txt', 'a') as errors_file:
                        errors_file.write(f"{alpha}\t{exponent}\t{sigma}\t{epsilon}\t{norm}\t"
                                          f"{result['iterations']}\t{result['time']}\t"
                                          f"{result['reason_registration_ended']}\n")


if __name__ == "__main__":
    run(main)