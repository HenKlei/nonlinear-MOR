import ast
import numpy as np
import pathlib
import time
from typer import Argument, Option, run
from typing import List

import geodesic_shooting
from geodesic_shooting.utils.reduced import pod

from load_model import load_full_order_model
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results
from nonlinear_mor.utils.versioning import get_git_hash, get_version


def main(example: str = Argument(..., help='Path to the example to execute, for instance '
                                           'example="1d.burgers.piecewise_constant.burgers_analytical"'),
         spatial_shape: List[int] = Argument(..., help='Number of unknowns in the spatial coordinate directions'),
         num_time_steps: int = Option(100, help='Number of time steps in the high-fidelity solutions'),
         additional_parameters: str = Option('{}', help='Additional parameters to pass to the full-order model',
                                             callback=ast.literal_eval),
         num_training_parameters: int = Option(3, help='Number of training parameters'),
         sampling_mode: str = Option('uniform', help='Sampling mode for sampling the training parameters'),
         reference_parameters: str = Option('[[0.5], [1.5]]', help='Reference parameter, either a number or a list of numbers',
                                            callback=ast.literal_eval),
         oversampling_size: int = Option(10, help='Margin in pixels used for oversampling'),
         value_on_oversampling: float = Option(default=None, help='Value to set for the target snapshots '
                                                                  'on the oversampling domain'),
         optimization_method: str = Option('GD', help='Optimizer used for geodesic shooting'),
         alpha: float = Option(0.01, help='Registration parameter `alpha`'),
         exponent: int = Option(1, help='Registration parameter `exponent`'),
         gamma: float = Option(1.0, help='Registration parameter `gamma`'),
         sigma: float = Option(0.01, help='Registration parameter `sigma`'),
         l2_prod: bool = Option(False, help='Use L2 product for POD'),
         reuse_initial_vector_field: bool = Option(True, help='Reuse the previous initial vector field as guess for '
                                                              'the next registration'),
         write_results: bool = Option(True, help='Determines whether or not to write results to disc (useful during '
                                                 'development)')):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_registration_test_{timestr}'
    if write_results:
        pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)

    spatial_shape = tuple(spatial_shape)
    fom = load_full_order_model(example, spatial_shape, num_time_steps, additional_parameters)

    parameters = fom.parameter_space.sample(num_training_parameters, sampling_mode)

    if fom.dim == 1:
        restriction = np.s_[oversampling_size:-oversampling_size, oversampling_size:-oversampling_size]
    elif fom.dim == 2:
        restriction = np.s_[oversampling_size:-oversampling_size, oversampling_size:-oversampling_size,
                            oversampling_size:-oversampling_size]
    elif fom.dim == 3:
        restriction = np.s_[oversampling_size:-oversampling_size, oversampling_size:-oversampling_size,
                            oversampling_size:-oversampling_size, oversampling_size:-oversampling_size]
    else:
        raise NotImplementedError
    if oversampling_size == 0:
        restriction = np.s_[...]

    gs_smoothing_params = {'alpha': alpha, 'exponent': exponent, 'gamma': gamma}
    registration_params = {'sigma': sigma, 'restriction': restriction, 'optimization_method': optimization_method}

    if not (isinstance(reference_parameters, list) and len(reference_parameters) > 1):
        reference_parameters = [reference_parameters]
    u_refs = [fom.solve(mu_ref) for mu_ref in reference_parameters]

    if value_on_oversampling is not None:
        for u_ref in u_refs:
            mask = np.ones(u_ref.full_shape, bool)
            mask[restriction] = 0
            u_ref[mask] = value_on_oversampling
    geodesic_shooter = geodesic_shooting.GeodesicShooting(**gs_smoothing_params)

    # write summary
    summary = '========================================================\n'
    summary += 'Git hash of nonlinear_mor-module: ' + get_git_hash() + '\n'
    summary += '========================================================\n'
    summary += 'FOM: ' + str(fom) + '\n'
    summary += 'Geodesic Shooting:\n'
    summary += '------------------\n'
    summary += 'Version: ' + get_version(geodesic_shooting) + '\n'
    summary += str(geodesic_shooter) + '\n'
    summary += '------------------\n'
    summary += 'Registration parameters: ' + str(registration_params) + '\n'
    summary += '------------------\n'
    summary += 'Reference parameters: ' + str(reference_parameters) + '\n'
    summary += 'Parameters (' + str(len(parameters)) + '): ' + str(parameters) + '\n'
    if write_results:
        with open(f'{filepath_prefix}/summary.txt', 'a') as f:
            f.write(summary)

    full_vector_fields = []
    snapshots = []

    initial_vector_field = None
    S = []

    for mu in parameters:
        print(f"mu: {mu}")
        u = fom.solve(mu)
        snapshots.append(u)
        _, S = pod(snapshots, num_modes=1, product_operator=None, return_singular_values='all')

        i = np.argmin(np.linalg.norm(np.array(reference_parameters) - mu, axis=-1))
        u_ref = u_refs[i]
        print(f"Reference parameter: {reference_parameters[i]}")
        result = geodesic_shooter.register(u_ref, u, **registration_params, return_all=True,
                                           initial_vector_field=initial_vector_field)
        if reuse_initial_vector_field:
            initial_vector_field = result['initial_vector_field']
        full_vector_fields.append(result['initial_vector_field'])
#        plot_registration_results(result, show_restriction_boundary=True)
        if write_results:
            save_plots_registration_results(result, filepath=f'{filepath_prefix}/mu_{str(mu).replace(".", "_")}/',
                                            show_restriction_boundary=True)
            transformed_input = result['transformed_input']
            absolute_error = (u - transformed_input).norm
            relative_error = absolute_error / u.norm
            restriction = registration_params.get('restriction')
            if restriction:
                absolute_error_restricted = (u - transformed_input).get_norm(restriction=restriction)
                relative_error_restricted = absolute_error_restricted / u.get_norm(restriction=restriction)
            else:
                absolute_error_restricted = absolute_error
                relative_error_restricted = relative_error
            with open(f'{filepath_prefix}/relative_mapping_errors.txt', 'a') as errors_file:
                errors_file.write(f"{mu}\t{absolute_error_restricted}\t{relative_error_restricted}\t"
                                  f"{absolute_error}\t{relative_error}\t"
                                  f"{result['iterations']}\t{result['time']}\t{result['reason_registration_ended']}\t"
                                  f"{result['energy_regularizer']}\t{result['energy_intensity_unscaled']}\t"
                                  f"{result['energy_intensity']}\t{result['energy']}\t{result['norm_gradient']}\n")

    if l2_prod:
        product_operator = None
    else:
        product_operator = geodesic_shooter.regularizer.cauchy_navier

    all_reduced_vector_fields, singular_values = pod(full_vector_fields,
                                                     num_modes=num_training_parameters,
                                                     product_operator=product_operator,
                                                     return_singular_values='all')
    print("Singular values of the initial vector fields with respect to the parameter:")
    print(singular_values)
    if write_results:
        filepath = filepath_prefix + '/singular_values'
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
        with open(f'{filepath}/singular_values_snapshots.txt', 'a') as singular_values_file:
            for s in S:
                singular_values_file.write(f"{s}\n")
        with open(f'{filepath}/singular_values_initial_vector_fields.txt', 'a') as singular_values_file:
            for val in singular_values:
                singular_values_file.write(f"{val}\n")

    filepath = filepath_prefix + '/singular_vectors'
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    for i, mode in enumerate(all_reduced_vector_fields):
        mode.save(f'{filepath}/mode_{i}.png', plot_args={'title': f'Mode {i}'})


if __name__ == "__main__":
    run(main)
