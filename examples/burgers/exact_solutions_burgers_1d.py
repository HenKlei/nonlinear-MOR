import pickle
import numpy as np
import matplotlib.pyplot as plt

from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor as NonlinearReductor
from nonlinear_mor.models import AnalyticalModel


N_X = 10
N_T = 10


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


def create_fom():
    return AnalyticalModel(exact_solution, n_x=N_X, n_t=N_T)


def main():
    fom = create_fom()

    N_train = 3
    parameters = np.linspace(0.25, 1.5, N_train)
    reference_parameter = 0.25

    gs_smoothing_params = {'alpha': 100., 'exponent': 2}
    registration_params = {'sigma': 0.1}
    max_basis_size = 2
    restarts = 10

    reductor = NonlinearReductor(fom, parameters, reference_parameter,
                                 gs_smoothing_params=gs_smoothing_params)
    rom, output_dict = reductor.reduce(max_basis_size=max_basis_size, return_all=True, restarts=restarts,
                                       registration_params=registration_params)

    with open('outputs/output_dict_rom', 'wb') as output_file:
        pickle.dump(output_dict, output_file)

    test_parameters = [0.5, 0.75]
    for test_parameter in test_parameters:
        u_red = rom.solve(test_parameter)
        u_red.save(f'results/result_mu_{str(test_parameter).replace(".", "_")}.png')
        u_full = fom.solve(test_parameter)
        u_full.save(f'results/full_solution_mu_{str(test_parameter).replace(".", "_")}.png')
        (u_red - u_full).save(f'results/difference_mu_{str(test_parameter).replace(".", "_")}.png')

        with open('results/relative_errors.txt', 'a') as errors_file:
            errors_file.write(f"{test_parameter}\t{(u_red - u_full).norm / u_full.norm}\n")


if __name__ == "__main__":
    main()
