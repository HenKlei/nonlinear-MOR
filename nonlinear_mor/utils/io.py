import time
import matplotlib.pyplot as plt
import numpy as np


def write_errors_to_file(filename, time_for_reduction, gs_smoothing_params, registration_params,
                         pod_size, singular_values, restarts, trainer_params, training_params,
                         best_loss, test_parameters, rom, fom, save_figs=True):
    with open(filename, 'a') as errors_file:
        errors_file.write("Reduction:\n----------\n")
        errors_file.write(f"Total time for reduction: {time_for_reduction}\n")
        errors_file.write("Registration:\n"
                          f"  Alpha: {gs_smoothing_params['alpha']}\n"
                          f"  Exponent: {gs_smoothing_params['exponent']}\n"
                          f"  Sigma: {registration_params['sigma']}\n"
                          f"  Epsilon: {registration_params['epsilon']}\n"
                          f"  Max. Iterations: {registration_params['iterations']}\n"
                          "POD of vector fields:\n"
                          f"  Basis size: {pod_size}\n"
                          "  Largest truncated singular value: "
                          f"{singular_values[pod_size]}\n"
                          "Training of neural networks:\n"
                          f"  Restarts: {restarts}\n"
                          f"  Learning rate: {trainer_params['learning_rate']}\n"
                          f"  Number of epochs: {training_params['number_of_epochs']}\n"
                          f"  Best loss: {best_loss}\n\n")
        errors_file.write("Errors and computation times:\n-------\n")
        errors_file.write("Test parameter\tAbsolute L2-error\tRelative L2-error\t"
                          "Time full solution\tTime reduced solution\tSpeedup\n")

    for test_parameter in test_parameters:
        start = time.perf_counter()
        u_red = rom.solve(test_parameter)
        time_reduced_solution = time.perf_counter() - start
        start = time.perf_counter()
        u_full = fom.solve(test_parameter)
        time_full_solution = time.perf_counter() - start
        if save_figs:
            plt.matshow(u_red)
            plt.savefig(f'results/result_mu_{str(test_parameter).replace(".", "_")}.png')
            plt.close()
            plt.matshow(u_full)
            plt.savefig(f'results/full_solution_mu_{str(test_parameter).replace(".", "_")}.png')
            plt.close()
            plt.matshow(u_red - u_full)
            plt.savefig(f'results/difference_mu_{str(test_parameter).replace(".", "_")}.png')
            plt.close()
        with open(filename, 'a') as errors_file:
            errors_file.write(f"{test_parameter}\t"
                              f"{np.linalg.norm(u_red - u_full)}\t"
                              f"{np.linalg.norm(u_red - u_full) / np.linalg.norm(u_full)}\t"
                              f"{time_full_solution}\t"
                              f"{time_reduced_solution}\t"
                              f"{time_full_solution / time_reduced_solution}\n")

    with open(filename, 'a') as errors_file:
        errors_file.write("\n\n\n")
