import pickle
import numpy as np
import matplotlib.pyplot as plt

from nonlinear_mor.reductors import NonlinearReductor
from nonlinear_mor.models import AnalyticalModel


N_X = 50
N_T = 50


def exact_solution(x, *, mu=0.25):
    return 1. * (x[..., 0] <= mu * x[..., 1])


fom = AnalyticalModel(exact_solution, n_x=N_X, n_t=N_T)


N_train = 5
parameters = np.linspace(0.25, 1., N_train)
reference_parameter = 0.25

gs_smoothing_params = {'alpha': 1000., 'exponent': 3}
registration_params = {'sigma': 0.1, 'epsilon': 0.1, 'iterations': 1500}
restarts = 10

reductor = NonlinearReductor(fom, parameters, reference_parameter,
                             gs_smoothing_params=gs_smoothing_params)
rom, output_dict = reductor.reduce(return_all=True, restarts=restarts,
                                   registration_params=registration_params)

with open('outputs/output_dict_rom', 'wb') as output_file:
    pickle.dump(output_dict, output_file)

test_parameters = [0.5, 0.75]
for test_parameter in test_parameters:
    u_red = rom.solve(test_parameter)
    plt.matshow(u_red)
    plt.savefig(f'results/result_mu_{str(test_parameter).replace(".", "_")}.png')
    plt.close()
    u_full = fom.solve(test_parameter)
    plt.matshow(u_full)
    plt.savefig(f'results/full_solution_mu_{str(test_parameter).replace(".", "_")}.png')
    plt.close()
    plt.matshow(u_red - u_full)
    plt.savefig(f'results/difference_mu_{str(test_parameter).replace(".", "_")}.png')
    plt.close()
    with open('results/relative_errors.txt', 'a') as errors_file:
        errors_file.write(f"{test_parameter}\t"
                          f"{np.linalg.norm(u_red - u_full) / np.linalg.norm(u_full)}\n")
