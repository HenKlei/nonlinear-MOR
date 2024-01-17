import ast
import numpy as np
import pathlib
import time
from typer import Argument, Option, run
from typing import List

import geodesic_shooting

from load_model import load_full_order_model
from geodesic_shooting.utils.summary import plot_registration_results, save_plots_registration_results


def main(example: str = Argument(..., help='Path to the example to execute, for instance '
                                           'example="1d.burgers.piecewise_constant.burgers_1d_analytical"'),
         spatial_shape: List[int] = Argument(..., help='Number of unknowns in the spatial coordinate directions'),
         num_time_steps: int = Option(100, help='Number of time steps in the high-fidelity solutions'),
         additional_parameters: str = Option('{}', help='Additional parameters to pass to the full-order model',
                                             callback=ast.literal_eval),
         num_training_parameters: int = Option(50, help='Number of training parameters'),
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
         write_results: bool = Option(True, help='Determines whether or not to write results to disc (useful during '
                                                 'development)')):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_restriction_test_{timestr}'
    if write_results:
        pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)

    spatial_shape = tuple(spatial_shape)
    fom = load_full_order_model(example, spatial_shape, num_time_steps, additional_parameters)

    parameters = fom.parameter_space.sample(num_training_parameters, sampling_mode)
    parameters = [1.]

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

    # x = np.linspace(0., 1., spatial_shape[0])
    # y = np.linspace(0., 1., num_time_steps)
    # data_initial_vector_field = np.zeros((*spatial_shape, num_time_steps, fom.dim + 1))
    # for i in range(spatial_shape[0]):
    #     for j in range(num_time_steps):
    #         data_initial_vector_field[i, j, 0] = y[j] #* 1e-5 #* (1.-4.*(x[i]-0.5)**2)
    # from geodesic_shooting.core import VectorField
    # initial_vector_field = VectorField(data=data_initial_vector_field)
    # initial_vector_field.plot()
    # initial_vector_field.get_magnitude().plot(title="Magnitude")
    # initial_vector_field.div.plot(title="Divergence")
    # import matplotlib.pyplot as plt
    # plt.show()
#    import sys
#    sys.exit(0)

    u_ref = fom.solve(reference_parameter)
    if value_on_oversampling is not None:
        mask = np.ones(u_ref.full_shape, bool)
        mask[restriction] = 0
        u_ref[mask] = value_on_oversampling
    geodesic_shooter = geodesic_shooting.GeodesicShooting(**gs_smoothing_params)

    # ivf_time_series = geodesic_shooter.integrate_forward_vector_field(initial_vector_field)
    # time_dependent_diffeomorphism = ivf_time_series.integrate(get_time_dependent_diffeomorphism=True)
    # interval = 1
    # _ = time_dependent_diffeomorphism.animate("Animation of time-evolution of diffeomorphism", interval=interval)
    # plt.show()
    # diffeomorphism = time_dependent_diffeomorphism[-1]
    # u_ref.push_forward(diffeomorphism).plot()
    # plt.show()

    for mu in parameters:
        print(f"mu: {mu}")
        u = fom.solve(mu)
        result = geodesic_shooter.register(u_ref, u, **registration_params, return_all=True)#, initial_vector_field=initial_vector_field)
        plot_registration_results(result, show_restriction_boundary=True)
        if write_results:
            save_plots_registration_results(result, filepath=f'{filepath_prefix}/mu_{str(mu).replace(".", "_")}/',
                                            show_restriction_boundary=True)


if __name__ == "__main__":
    run(main)
