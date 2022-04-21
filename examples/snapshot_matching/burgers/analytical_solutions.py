from typer import Option, run

from geodesic_shooting.core import ScalarFunction
from nonlinear_mor.utils.example_scripts.analytical_solutions import run_analytical_solution_tests


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


BASE_FILEPATH_RESULTS = 'results/'


def main(reference_parameter: float = Option(0.25, help='Reference parameter'),
         num_parameters: int = Option(10, help='Number of training snapshots'),
         alpha: float = Option(100., help='Alpha'),
         exponent: int = Option(2, help='Exponent'),
         sigma: float = Option(0.1, help='Sigma'),
         min_stepsize: float = Option(5e-5, help='Minimum stepsize'),
         max_stepsize: float = Option(1e-1, help='Maximum stepsize'),
         iterations: int = Option(5000, help='Maximum number of iterations'),
         modes: int = Option(5, help='Number of modes to visualize'),
         N_x: int = Option(50, help='Number of pixels in x-direction'),
         N_y: int = Option(50, help='Number of pixels in y-direction'),
         reuse_vector_fields: bool = Option(True, help='Reuse former vector field for next registration?')):

    if iterations < 0:
        iterations = None

    run_analytical_solution_tests(exact_solution, BASE_FILEPATH_RESULTS, reference_parameter=reference_parameter,
                                  num_parameters=num_parameters, alpha=alpha, exponent=exponent, sigma=sigma,
                                  parameters_line_search={'min_stepsize': min_stepsize, 'max_stepsize': max_stepsize},
                                  iterations=iterations, modes=modes, N_x=N_x, N_y=N_y,
                                  reuse_vector_fields=reuse_vector_fields)


if __name__ == "__main__":
    run(main)
