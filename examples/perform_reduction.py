import ast
import pathlib
import dill as pickle
import time
from typer import Argument, Option, run
from typing import List

from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor

from load_model import load_full_order_model


def main(example: str = Argument(..., help='Path to the example to execute, for instance '
                                           'example="1d.burgers.piecewise_constant.burgers_1d_analytical"'),
         spatial_shape: List[int] = Argument(..., help='Number of unknowns in the spatial coordinate directions'),
         num_time_steps: int = Option(100, help='Number of time steps in the high-fidelity solutions'),
         additional_parameters: str = Option('{}', help='Additional parameters to pass to the full-order model',
                                             callback=ast.literal_eval),
         num_training_parameters: int = Option(50, help='Number of training parameters'),
         sampling_mode: str = Option('uniform', help='Sampling mode for sampling the training parameters'),
         reference_parameter: str = Option('0.25', help='Reference parameter, either a number or a list of numbers',
                                           callback=ast.literal_eval),
         alpha: float = Option(100., help='Registration parameter `alpha`'),
         exponent: int = Option(2, help='Registration parameter `exponent`'),
         sigma: float = Option(0.1, help='Registration parameter `sigma`'),
         max_reduced_basis_size_vector_fields: int = Option(50, help='Maximum dimension of reduced basis for '
                                                                     'vector fields'),
         reduce_snapshots: bool = Option(True, help='Determines whether or not to reduce the snapshots as well'),
         max_reduced_basis_size_snapshots: int = Option(50, help='Maximum dimension of reduced basis for snapshots'),
         num_workers: int = Option(1, help='Number of cores to use during registration; if greater than 1, the former '
                                           'vector field is not reused, otherwise the former vector field is used as '
                                           'initialization for the registration'),
         l2_prod: bool = Option(False, help='Determines whether or not to use the L2-product as inner product for '
                                            'orthonormalizing the vector fields'),
         neural_network_training_restarts: int = Option(1, help='Maximum number of training restarts'),
         hidden_layers_vector_field: List[int] = Option([20, 20, 20], help='Number of neurons in each hidden layer '
                                                        '(neural network for reduced coefficients of vector fields)'),
         hidden_layers_snapshots: List[int] = Option([20, 20, 20], help='Number of neurons in each hidden layer '
                                                     '(neural network for reduced coefficients of snapshots)'),
         full_vector_fields_filepath_prefix: str = Option(None, help='Filepath prefix for full vector fields file'),
         write_results: bool = Option(True, help='Determines whether or not to write results to disc (useful during '
                                                 'development)')):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_nonlinear_reductor_{timestr}'
    if write_results:
        pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)

    spatial_shape = tuple(spatial_shape)
    fom = load_full_order_model(example, spatial_shape, num_time_steps, additional_parameters)

    if write_results:
        full_order_model_filepath = f'{filepath_prefix}/full_order_model'
        pathlib.Path(full_order_model_filepath).mkdir(parents=True, exist_ok=True)
        fom_dictionary = {'example': example, 'spatial_shape': spatial_shape,
                          'num_time_steps': num_time_steps, 'additional_parameters': additional_parameters}
        with open(f'{full_order_model_filepath}/model.pickle', 'wb') as fom_file:
            pickle.dump(fom_dictionary, fom_file)

    parameters = fom.parameter_space.sample(num_training_parameters, sampling_mode)

    gs_smoothing_params = {'alpha': alpha, 'exponent': exponent}
    registration_params = {'sigma': sigma}
    assert max_reduced_basis_size_vector_fields <= num_training_parameters
    basis_sizes_vector_fields = range(1, max_reduced_basis_size_vector_fields + 1)
    if reduce_snapshots:
        assert max_reduced_basis_size_snapshots <= num_training_parameters
        basis_sizes_snapshots = range(1, max_reduced_basis_size_snapshots + 1)
    else:
        basis_sizes_snapshots = [1]
        max_reduced_basis_size_snapshots = "1 (not performing reduction of snapshots)"

    if full_vector_fields_filepath_prefix:
        full_vector_fields_file = f'{full_vector_fields_filepath_prefix}/outputs/full_vector_fields'
    else:
        full_vector_fields_file = None

    reductor = NonlinearNeuralNetworkReductor(fom, parameters, reference_parameter,
                                              gs_smoothing_params=gs_smoothing_params)

    if write_results:
        reductor_summary = reductor.create_summary(registration_params=registration_params)
        fom_summary = fom.create_summary()
        reduction_summary = ('Maximum dimension of the reduced basis (vector fields): '
                             + str(max_reduced_basis_size_vector_fields) + '\n' +
                             'Maximum dimension of the reduced basis (snapshots): '
                             + str(max_reduced_basis_size_snapshots) + '\n' +
                             'Number of training parameters: ' + str(num_training_parameters) + '\n' +
                             'Parameter sampling mode: ' + str(sampling_mode) + '\n' +
                             'Number of training restarts in neural network training: ' +
                             str(neural_network_training_restarts) + '\n' +
                             'Hidden layers of neural network (vector fields): ' + str(hidden_layers_vector_field)
                             + '\n' + 'Hidden layers of neural network (snapshots): ' + str(hidden_layers_snapshots)
                             + '\n' + 'L2-product used: ' + str(l2_prod) + '\n')

        if not reduce_snapshots:
            reduction_summary += '++ Not performing reduction of snapshots!\n'

        with open(f'{filepath_prefix}/summary.txt', 'a') as summary_file:
            summary_file.write('Example: ' + example)
            summary_file.write('\n\n========================================================\n\n')
            summary_file.write(fom_summary)
            summary_file.write('\n\n========================================================\n\n')
            summary_file.write(reductor_summary)
            summary_file.write('\n\n========================================================\n\n')
            summary_file.write(reduction_summary)

    roms, output_dict = reductor.reduce(basis_sizes_vector_fields=basis_sizes_vector_fields, l2_prod=l2_prod,
                                        reduce_snapshots=reduce_snapshots, basis_sizes_snapshots=basis_sizes_snapshots,
                                        return_all=True, save_intermediate_results=write_results,
                                        restarts=neural_network_training_restarts, num_workers=num_workers,
                                        full_vector_fields_file=full_vector_fields_file,
                                        registration_params=registration_params,
                                        hidden_layers_vf=hidden_layers_vector_field,
                                        hidden_layers_s=hidden_layers_snapshots, filepath_prefix=filepath_prefix)

    if write_results:
        outputs_filepath = f'{filepath_prefix}/outputs'
        pathlib.Path(outputs_filepath).mkdir(parents=True, exist_ok=True)
        with open(f'{outputs_filepath}/output_dict_rom', 'wb') as output_file:
            pickle.dump(output_dict, output_file)
        with open(f'{outputs_filepath}/full_vector_fields', 'wb') as output_file:
            pickle.dump(output_dict['full_vector_fields'], output_file)

        for basis_size_vf in basis_sizes_vector_fields:
            for basis_size_s in basis_sizes_snapshots:
                rom = roms[basis_size_vf-1][basis_size_s-1][0]
                reduced_models_filepath = (f'{filepath_prefix}/reduced_models/'
                                           + f'basis_sizes_vf_{basis_size_vf}_s_{basis_size_s}')
                pathlib.Path(reduced_models_filepath).mkdir(parents=True, exist_ok=True)
                rom.save_model(reduced_models_filepath)


if __name__ == "__main__":
    run(main)
