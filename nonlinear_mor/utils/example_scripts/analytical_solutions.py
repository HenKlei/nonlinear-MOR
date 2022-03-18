import os
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting


def run_analytical_solution_tests(exact_solution, BASE_FILEPATH_RESULTS='results/', reference_parameter=0.25,
                                  num_parameters=10, alpha=100., exponent=2, sigma=0.1,
                                  parameters_line_search={'min_stepsize': 5e-5, 'max_stepsize': 1e-1},
                                  iterations=5000, modes=5, N_x=50, N_y=50, reuse_vector_fields=True):

    XX, YY = np.meshgrid(np.linspace(0., 1., N_x), np.linspace(0., 1., N_y))
    XY = np.stack([XX.T, YY.T], axis=-1)

    if not os.path.exists(BASE_FILEPATH_RESULTS):
        os.makedirs(BASE_FILEPATH_RESULTS)

    min_stepsize = parameters_line_search['min_stepsize']
    max_stepsize = parameters_line_search['max_stepsize']

    FILEPATH_RESULTS = (BASE_FILEPATH_RESULTS + f'num_parameters{num_parameters}_alpha{str(alpha).replace(".", "_")}'
                        + f'_exponent{exponent}_sigma{str(sigma).replace(".", "_")}'
                        + f'_min_stepsize{str(min_stepsize).replace(".", "_")}'
                        + f'_max_stepsize{str(max_stepsize).replace(".", "_")}'
                        + f'_reuse{reuse_vector_fields}/')
    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)
    FILEPATH_IMAGES = FILEPATH_RESULTS + 'images/'
    if not os.path.exists(FILEPATH_IMAGES):
        os.makedirs(FILEPATH_IMAGES)

    # perform the registration
    gs = geodesic_shooting.GeodesicShooting(alpha=alpha, exponent=exponent)

    u_ref = exact_solution(XY, mu=reference_parameter)
    u_ref.plot(f"Reference solution mu={reference_parameter}")
    u_ref.save(FILEPATH_IMAGES + f'u_mu_{str(reference_parameter).replace(".", "_")}',
               title=f"Reference solution mu={reference_parameter}")

    def compute_registration(u, initial_vector_field=None):
        result = gs.register(u_ref, u, sigma=sigma, iterations=iterations,
                             parameters_line_search=parameters_line_search,
                             return_all=True, initial_vector_field=initial_vector_field)
        u_transformed = result['transformed_input']
        norm = (u - u_transformed).norm / u.norm
        return result, norm

    mus = np.linspace(1., 0.25, num_parameters, endpoint=False)[::-1]

    vector_fields = []
    solutions = []
    max_error = None
    result = {}

    with open(FILEPATH_RESULTS + 'results.txt', 'a') as errors_file:
        errors_file.write("Parameter\tRelative error of mapping\tIterations\tTime in seconds\t"
                          "Length of path\tReason for end of registration\n")

    for mu in mus:
        u = exact_solution(XY, mu=mu)
        u.plot(f"Exact solution for parameter mu={mu}")
        u.save(FILEPATH_IMAGES + f'u_mu_{str(mu).replace(".", "_")}',
               title=f"Exact solution for parameter mu={mu}")

        solutions.append(u)
        if reuse_vector_fields and vector_fields:
            result, error = compute_registration(u, initial_vector_field=result['initial_vector_field'])
        else:
            result, error = compute_registration(u)
        if max_error is None or error > max_error:
            max_error = error
        vector_fields.append(result['initial_vector_field'])

        with open(FILEPATH_RESULTS + 'results.txt', 'a') as errors_file:
            errors_file.write(f"{mu:.3e}\t{error:.3e}\t{result['iterations']}\t{result['time']}\t"
                              f"{result['length']:.3e}\t{result['reason_registration_ended']}\n")

        u_transformed = result['transformed_input']
        u_transformed.plot(f"Transformed reference solution for parameter mu={mu}")
        u_transformed.save(FILEPATH_IMAGES + f'u_approx_mu_{str(mu).replace(".", "_")}',
                           title=f"Transformed reference solution for parameter mu={mu}")

        v0 = result['initial_vector_field']
        v0.plot(f"Initial vector field for mu={mu}")
        v0.save(FILEPATH_IMAGES + f'initial_vector_field_mu_{str(mu).replace(".", "_")}',
                title=f"Initial vector field for mu={mu}")
        plt.show()

    def pod(A, modes=10, return_singular_values=True):
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        if return_singular_values:
            return U[:, :min(modes, U.shape[1])], S
        else:
            return U[:, :min(modes, U.shape[1])]

    A = np.array(vector_fields).T
    A_red, S_vector_fields = pod(A, modes=modes)
    for i, v in enumerate(A_red.T):
        v.plot()
        plt.title(f"Reduced initial vector field {i}")
        plt.savefig(FILEPATH_IMAGES + f'reduced_initial_vector_field_{i}')
        plt.close()
    _, S_snapshots = pod(np.array(solutions).T)

    with open(FILEPATH_RESULTS + 'results.txt', 'a') as errors_file:
        np.set_printoptions(formatter={'float': '{:.3e}'.format})
        errors_file.write("\n\nReuse vector fields\tReference parameter\tNumber of snapshots\talpha\texponent\tsigma\t"
                          "Minimum stepsize\tMaximum stepsize\tMaximum registration error\t"
                          "Singular values vector fields\tSingular values snapshots\n")
        errors_file.write(f"{reuse_vector_fields}\t{reference_parameter}\t{num_parameters}\t{alpha}\t{exponent}\t"
                          f"{sigma}\t{min_stepsize}\t{max_stepsize}\t{max_error:.3e}\t{S_vector_fields}\t"
                          f"{S_snapshots}\n")
