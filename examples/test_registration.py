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


def main(example: str = Argument(..., help='Path to the example to execute, for instance '
                                           'example="1d.burgers.piecewise_constant.burgers_analytical"'),
         spatial_shape: List[int] = Argument(..., help='Number of unknowns in the spatial coordinate directions'),
         num_time_steps: int = Option(100, help='Number of time steps in the high-fidelity solutions'),
         additional_parameters: str = Option('{}', help='Additional parameters to pass to the full-order model',
                                             callback=ast.literal_eval),
         num_training_parameters: int = Option(3, help='Number of training parameters'),
         sampling_mode: str = Option('uniform', help='Sampling mode for sampling the training parameters'),
         reference_parameter: str = Option('0.625', help='Reference parameter, either a number or a list of numbers',
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

    u_ref = fom.solve(reference_parameter)
    if value_on_oversampling is not None:
        mask = np.ones(u_ref.full_shape, bool)
        mask[restriction] = 0
        u_ref[mask] = value_on_oversampling
    geodesic_shooter = geodesic_shooting.GeodesicShooting(**gs_smoothing_params)

    full_vector_fields = []

    initial_vector_field = None

    for mu in parameters:
        print(f"mu: {mu}")
        u = fom.solve(mu)
        result = geodesic_shooter.register(u_ref, u, **registration_params, return_all=True,
                                           initial_vector_field=initial_vector_field)
        if reuse_initial_vector_field:
            initial_vector_field = result['initial_vector_field']
        full_vector_fields.append(result['initial_vector_field'])
        plot_registration_results(result, show_restriction_boundary=True)
        if write_results:
            save_plots_registration_results(result, filepath=f'{filepath_prefix}/mu_{str(mu).replace(".", "_")}/',
                                            show_restriction_boundary=True)

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
        with open(f'{filepath}/singular_values.txt', 'a') as singular_values_file:
            for val in singular_values:
                singular_values_file.write(f"{val}\n")

    filepath = filepath_prefix + '/singular_vectors'
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    for i, mode in enumerate(all_reduced_vector_fields):
        mode.save(f'{filepath}/mode_{i}.png', plot_args={'title': f'Mode {i}'})


if __name__ == "__main__":
    run(main)
