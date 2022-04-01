import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from nonlinear_mor.utils.torch.early_stopping import SimpleEarlyStoppingScheduler
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer
from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor as NonlinearReductor


def main():
    restarts = 1

    from exact_solutions_burgers_1d import create_fom
    fom = create_fom()

    N_train = 50
    training_set = np.linspace(0.25, 1., N_train)
    reference_parameter = 0.25

    max_basis_size = 15

    filepath_prefix = 'tmp_results'

    reductor = NonlinearReductor(fom, training_set, reference_parameter)
    rom, output_dict = reductor.reduce(max_basis_size=max_basis_size, return_all=True, restarts=restarts,
                                       full_velocity_fields_file=f'{filepath_prefix}/outputs/full_velocity_fields',
                                       filepath_prefix=filepath_prefix)

    test_parameters = [0.5, 0.75, 1., 1.25]
    for test_parameter in test_parameters:
        u_full = fom.solve(test_parameter)
        u_full.plot(f"Full solution $u_\mu$ for $\mu={test_parameter}$")
        tikzplotlib.save(f"{filepath_prefix}/figures_tex/full_solution_mu_{str(test_parameter).replace('.', '_')}.tex")
        u_red = rom.solve(test_parameter, filepath_prefix=filepath_prefix)
        u_red.plot("Reduced solution $u_\mu^\mathrm{red}$ for " + f"$\mu={test_parameter}$")
        tikzplotlib.save(f"{filepath_prefix}/figures_tex/reduced_solution_mu_{str(test_parameter).replace('.', '_')}.tex")
        relative_error = (u_red - u_full).norm / u_full.norm
        (u_red - u_full).plot("Difference $u_\mu^\mathrm{red}-u_\mu$ for " + f"$\mu={test_parameter}$")
        tikzplotlib.save(f"{filepath_prefix}/figures_tex/difference_mu_{str(test_parameter).replace('.', '_')}.tex")
        plt.show()
        print(f"Parameter: {test_parameter}; Relative error: {relative_error}")
        with open(filepath_prefix + '/results/relative_errors.txt', 'a') as errors_file:
            errors_file.write(f"{test_parameter}\t{relative_error}\n")


if __name__ == '__main__':
    main()
