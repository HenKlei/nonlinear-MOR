from typer import Option, run
import numpy as np
import pathlib

from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.models import AnalyticalModel
from nonlinear_mor.utils.pod import pod


def main(N_X: int = Option(100, help='Number of pixels in x-direction'),
         N_T: int = Option(100, help='Number of pixels in time-direction'),
         N_train: int = Option(50, help='Number of training parameters')):

    def exact_solution(x, *, mu=0.25):
        s_l = 1.5 * mu
        s_m = mu
        s_r = 0.5 * mu
        t_intersection = 0.25 / (s_l - s_r)
        return ScalarFunction(data=(2. * (x[..., 1] <= t_intersection) * (0.25 + s_l * x[..., 1] - x[..., 0] >= 0.)
                                    + (2. * (x[..., 1] > t_intersection)
                                       * (0.25 + (s_l - s_m) * t_intersection + s_m * x[..., 1] - x[..., 0] >= 0.))
                                    + (1. * (0.25 + s_l * x[..., 1] - x[..., 0] < 0.)
                                       * (0.5 + s_r * x[..., 1] - x[..., 0] > 0.))))

    def create_fom():
        return AnalyticalModel(exact_solution, n_x=N_X, n_t=N_T, name='Analytical Burgers Model')

    fom = create_fom()

    parameters = np.linspace(0.25, 1.5, N_train)

    snapshots = np.vstack([fom.solve(p).to_numpy().flatten() for p in parameters]).T
    print(snapshots.shape)

    _, S = pod(snapshots, modes=1, product_operator=None, return_singular_values=True)
    filepath_prefix = f'results_nx_{N_X}_nt_{N_T}_Ntrain_{N_train}'
    results_filepath = f'{filepath_prefix}/results'
    pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)

    with open(f'{results_filepath}/singular_values_snapshots.txt', 'a') as singular_values_file:
        for s in S:
            singular_values_file.write(f"{s}\n")


if __name__ == "__main__":
    run(main)
