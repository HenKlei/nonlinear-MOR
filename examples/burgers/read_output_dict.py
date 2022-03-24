import numpy as np
import matplotlib.pyplot as plt

from nonlinear_mor.utils.torch.early_stopping import SimpleEarlyStoppingScheduler
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer
from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor as NonlinearReductor


def main():
    restarts = 10

    from exact_solutions_burgers_1d import create_fom
    fom = create_fom()

    N_train = 50
    training_set = np.linspace(0.25, 1., N_train)
    reference_parameter = 0.25

    max_basis_size = 15

    reductor = NonlinearReductor(fom, training_set, reference_parameter)
    rom, output_dict = reductor.reduce(max_basis_size=max_basis_size, return_all=True, restarts=restarts,
                                       full_velocity_fields_file='outputs/full_velocity_fields')

    test_parameters = [0.5, 0.75]
    for test_parameter in test_parameters:
        u_red = rom.solve(test_parameter)
        """
        u_red.save()
        plt.matshow(u_red)
        plt.savefig(f'results/result_mu_{str(test_parameter).replace(".", "_")}.png')
        plt.close()
        """
        u_full = fom.solve(test_parameter)
        """
        plt.matshow(u_full)
        plt.savefig(f'results/full_solution_mu_{str(test_parameter).replace(".", "_")}.png')
        plt.close()
        plt.matshow(u_red - u_full)
        plt.savefig(f'results/difference_mu_{str(test_parameter).replace(".", "_")}.png')
        plt.close()
        """
        relative_error = (u_red - u_full).norm / u_full.norm
        print(f"Parameter: {test_parameter}; Relative error: {relative_error}")
        with open('results/relative_errors.txt', 'a') as errors_file:
            errors_file.write(f"{test_parameter}\t"
                              f"{relative_error}\n")


if __name__ == '__main__':
    main()
