from typer import Option, run

from nonlinear_mor.utils.example_scripts.analytical_solutions import run_analytical_solution_tests


def exact_solution(x, *, mu=0.25):
    return 1. * ((x[..., 1] - mu * x[..., 0]) > 0.)


BASE_FILEPATH_RESULTS = 'results/'


def main(reference_parameter: float = Option(0.25, help='Reference parameter'),
         num_parameters: int = Option(10, help='Number of training snapshots'),
         alpha: float = Option(100., help='Alpha'),
         exponent: int = Option(1, help='Exponent'),
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
