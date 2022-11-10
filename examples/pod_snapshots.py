import ast
import numpy as np
import pathlib
import time
from typer import Argument, Option, run
from typing import List

from geodesic_shooting.utils.reduced import pod

from load_model import load_full_order_model


def main(example: str = Argument(..., help='For instance example="1d.burgers.piecewise_constant.analytical"'),
         spatial_shape: List[int] = Argument(..., help=''),
         num_time_steps: int = Option(100, help='Number of pixels in time-direction'),
         additional_parameters: str = Option("{}", help='', callback=ast.literal_eval),
         num_training_parameters: int = Option(50, help='Number of training parameters'),
         sampling_mode: str = Option('uniform', help='')):

    spatial_shape = tuple(spatial_shape)
    fom = load_full_order_model(example, spatial_shape, num_time_steps, additional_parameters)
    fom_summary = fom.create_summary()

    parameters = fom.parameter_space.sample(num_training_parameters, sampling_mode)

    snapshots = [fom.solve(mu) for mu in parameters]

    _, S = pod(snapshots, num_modes=1, product_operator=None, return_singular_values='all')

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_pod_snapshots_{timestr}'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)

    with open(f'{filepath_prefix}/singular_values_snapshots.txt', 'a') as singular_values_file:
        for s in S:
            singular_values_file.write(f"{s}\n")

    with open(f'{filepath_prefix}/summary.txt', 'a') as summary_file:
        summary_file.write('Example: ' + example + '\n')
        summary_file.write('\n========================================================\n\n')
        summary_file.write('Number of training parameters: ' + str(num_training_parameters) + '\n')
        summary_file.write('Parameter sampling mode: ' + str(sampling_mode) + '\n')
        summary_file.write('\n========================================================\n\n')
        summary_file.write(fom_summary + '\n')


if __name__ == "__main__":
    run(main)
