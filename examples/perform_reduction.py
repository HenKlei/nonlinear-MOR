import ast
import pathlib
import dill as pickle
import time
from typer import Argument, Option, run
from typing import List

from nonlinear_mor.reductors import NonlinearNeuralNetworkReductor as NonlinearReductor

from load_model import load_full_order_model


def main(example: str = Argument(..., help='For instance example="1d.burgers.piecewise_constant.analytical"'),
         spatial_shape: List[int] = Argument(..., help=''),
         num_time_steps: int = Option(100, help='Number of pixels in time-direction'),
         additional_parameters: str = Option("{}", help='', callback=ast.literal_eval),
         num_training_parameters: int = Option(50, help='Number of training parameters'),
         sampling_mode: str = Option('uniform', help=''),
         reference_parameter: str = Option("0.25", help='Reference parameter', callback=ast.literal_eval),
         alpha: float = Option(100., help='Alpha'),
         exponent: int = Option(2, help='Exponent'),
         sigma: float = Option(0.1, help='Sigma'),
         max_reduced_basis_size: int = Option(50, help='Maximum dimension of reduced basis for vector fields'),
         l2_prod: bool = Option(False, help=''),
         neural_network_training_restarts: int = Option(25, help='Maximum number of training restarts'),
         hidden_layers: List[int] = Option([20, 20, 20], help=''),
         full_vector_fields_filepath_prefix: str = Option(None, help='Filepath prefix for full vector fields file'),
         write_results: bool = Option(True, help='')):

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
    assert max_reduced_basis_size <= num_training_parameters
    basis_sizes = range(1, max_reduced_basis_size + 1)

    if full_vector_fields_filepath_prefix:
        full_vector_fields_file = f'{full_vector_fields_filepath_prefix}/outputs/full_vector_fields'
    else:
        full_vector_fields_file = None

    reductor = NonlinearReductor(fom, parameters, reference_parameter,
                                 gs_smoothing_params=gs_smoothing_params)

    if write_results:
        reductor_summary = reductor.create_summary(registration_params=registration_params)
        fom_summary = fom.create_summary()
        reduction_summary = ('Maximum dimension of the reduced basis: ' + str(max_reduced_basis_size) + '\n' +
                             'Number of training parameters: ' + str(num_training_parameters) + '\n' +
                             'Parameter sampling mode: ' + str(sampling_mode) + '\n' +
                             'Number of training restarts in neural network training: ' +
                             str(neural_network_training_restarts) + '\n' +
                             'Hidden layers of neural network: ' + str(hidden_layers) + '\n' +
                             'L2-product used: ' + str(l2_prod) + '\n')

        with open(f'{filepath_prefix}/summary.txt', 'a') as summary_file:
            summary_file.write('Example: ' + example)
            summary_file.write('\n\n========================================================\n\n')
            summary_file.write(fom_summary)
            summary_file.write('\n\n========================================================\n\n')
            summary_file.write(reductor_summary)
            summary_file.write('\n\n========================================================\n\n')
            summary_file.write(reduction_summary)

    roms, output_dict = reductor.reduce(basis_sizes=basis_sizes, l2_prod=l2_prod, return_all=True,
                                        save_intermediate_results=write_results,
                                        restarts=neural_network_training_restarts,
                                        full_vector_fields_file=full_vector_fields_file,
                                        registration_params=registration_params, hidden_layers=hidden_layers,
                                        filepath_prefix=filepath_prefix)

    if write_results:
        outputs_filepath = f'{filepath_prefix}/outputs'
        pathlib.Path(outputs_filepath).mkdir(parents=True, exist_ok=True)
        with open(f'{outputs_filepath}/output_dict_rom', 'wb') as output_file:
            pickle.dump(output_dict, output_file)
        with open(f'{outputs_filepath}/full_vector_fields', 'wb') as output_file:
            pickle.dump(output_dict['full_vector_fields'], output_file)

        for basis_size in basis_sizes:
            rom = roms[basis_size-1][0]
            reduced_models_filepath = f'{filepath_prefix}/reduced_models/basis_size_{basis_size}'
            pathlib.Path(reduced_models_filepath).mkdir(parents=True, exist_ok=True)
            rom.save_model(reduced_models_filepath)


if __name__ == "__main__":
    run(main)
