import ast
import pathlib
import dill as pickle
import time
from typer import Argument, Option, run
from typing import List
import numpy as np
import random
import torch

import geodesic_shooting
from geodesic_shooting.utils.reduced import pod

from nonlinear_mor.utils.versioning import get_git_hash, get_version
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer
from load_model import load_full_order_model, load_landmark_function
from automatic_landmark_detection import place_landmarks_on_edges

trainer_params = {}
hidden_layers = [50, 50]
training_params = {}
restarts = 10


def train_neural_network(layers_sizes, training_data, validation_data,
                         trainer_params={}, training_params={}):
    neural_network = FullyConnectedNetwork(layers_sizes)
    trainer = Trainer(neural_network, **trainer_params)
    best_loss, _ = trainer.train(training_data, validation_data, **training_params)
    return trainer.network, best_loss


def multiple_restarts_training(training_data, validation_data, layers_sizes, restarts,
                               trainer_params={}, training_params={}):
    best_neural_network = None
    best_loss = None

    for _ in range(restarts):
        neural_network, loss = train_neural_network(layers_sizes, training_data, validation_data,
                                                    trainer_params, training_params)
        if best_loss is None or loss is None or best_loss > loss:
            best_neural_network = neural_network
            best_loss = loss

    return best_neural_network, best_loss


def main(example: str = Argument(..., help='Path to the example to execute, for instance '
                                           'example="1d.burgers.piecewise_constant.burgers_1d_landmarks_analytical"'),
         spatial_shape: List[int] = Argument(..., help='Number of unknowns in the spatial coordinate directions'),
         num_time_steps: int = Option(100, help='Number of time steps in the high-fidelity solutions'),
         place_landmarks_automatically: bool = Option(False, help='Determines whether to run an automatic procedure to '
                                                                  'place landmarks or whether they are provided by the '
                                                                  'problem'),
         num_landmarks: int = Option(20, help='Number of landmarks to use when placing them automatically'),
         landmarks_labeled: bool = Option(True, help='Determines whether the landmarks are labeled'),
         many_landmarks: bool = Option(False, help='Determines whether to use many landmarks provided by the FOM'),
         additional_parameters: str = Option('{}', help='Additional parameters to pass to the full-order model',
                                             callback=ast.literal_eval),
         num_training_parameters: int = Option(50, help='Number of test parameters'),
         num_test_parameters: int = Option(50, help='Number of test parameters'),
         sampling_mode: str = Option('uniform', help='Sampling mode for sampling the training parameters'),
         oversampling_size: int = Option(10, help='Margin in pixels used for oversampling'),
         reference_parameter: str = Option('1.', help='Reference parameter, either a number or a list of numbers',
                                           callback=ast.literal_eval),
         sigma: float = Option(0.1, help='Registration parameter `sigma`'),
         kernel_sigma: float = Option(4., help='Kernel shape parameter `sigma`'),
         kernel_dist_sigma: float = Option(-1, help='Kernel shape parameter `sigma` for unlabeled case (if -1, the same value as `kernel_sigma` is used)'),
         write_results: bool = Option(True, help='Determines whether or not to write results to disc (useful during '
                                                 'development)')):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath_prefix = f'results_landmarks_registration_test_{timestr}'
    if write_results:
        pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)

    spatial_shape = tuple(spatial_shape)
    fom = load_full_order_model(example, spatial_shape, num_time_steps, additional_parameters)

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

    if write_results:
        full_order_model_filepath = f'{filepath_prefix}/full_order_model'
        pathlib.Path(full_order_model_filepath).mkdir(parents=True, exist_ok=True)
        fom_dictionary = {'example': example, 'spatial_shape': spatial_shape,
                          'num_time_steps': num_time_steps, 'additional_parameters': additional_parameters}
        with open(f'{full_order_model_filepath}/model.pickle', 'wb') as fom_file:
            pickle.dump(fom_dictionary, fom_file)

    u_ref = fom.solve(reference_parameter)

    if not place_landmarks_automatically:
        get_landmarks = load_landmark_function(example)
        reference_landmarks = get_landmarks(mu=reference_parameter, many_landmarks=many_landmarks)
    else:
        assert landmarks_labeled is False
        assert u_ref.dim == 2
        reference_landmarks = place_landmarks_on_edges(u_ref.to_numpy(), num_landmarks)

    num_landmarks = len(reference_landmarks)

    parameters = fom.parameter_space.sample(num_training_parameters, sampling_mode)

    gs = geodesic_shooting.LandmarkShooting(kwargs_kernel={"sigma": kernel_sigma})
    mins = np.zeros(u_ref.dim)
    maxs = np.ones(u_ref.dim)

    if write_results:
        results_filepath = f'{filepath_prefix}/results'
        pathlib.Path(results_filepath).mkdir(parents=True, exist_ok=True)
        with open(results_filepath + '/registration_errors.txt', 'w') as f:
            f.write('Parameter\tRelative error\n')
        with open(results_filepath + '/reference_landmarks.txt', 'w') as f:
            for l in reference_landmarks:
                f.write(f"{' '.join(str(x) for x in l)}\n")
        with open(results_filepath + '/reference_snapshot.txt', 'w') as f:
            coords = np.meshgrid(np.linspace(0, 1, u_ref.full_shape[0]), np.linspace(0, 1, u_ref.full_shape[1]))
            for x, y, z in zip(coords[0].flatten(), coords[1].flatten(), u_ref.flatten()):
                f.write(f'{x}\t{y}\t{z}\n')
        with open(f'{filepath_prefix}/summary.txt', 'a') as summary_file:
            summary_file.write('========================================================\n')
            summary_file.write('Git hash: ' + get_git_hash() + '\n')
            summary_file.write('========================================================\n')
            summary_file.write('FOM: ' + str(fom) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Reference parameter: ' + str(reference_parameter) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Oversampling size: ' + str(oversampling_size) + '\n')
            summary_file.write('Restriction: ' + str(restriction) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Geodesic Shooting:\n')
            summary_file.write('------------------\n')
            summary_file.write('Version: ' + get_version(geodesic_shooting) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Sigma in landmark shooting: ' + str(sigma) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Shape parameter of kernel: ' + str(kernel_sigma) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Place landmarks automatically: ' + str(place_landmarks_automatically) + '\n')
            summary_file.write('Number of landmarks: ' + str(num_landmarks) + '\n')
            summary_file.write('Landmarks labeled: ' + str(landmarks_labeled) + '\n')
            summary_file.write('Many landmarks: ' + str(many_landmarks) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Number of training parameters: ' + str(num_training_parameters) + '\n')
            summary_file.write('Sampling mode for training parameters: ' + str(sampling_mode) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Reference landmarks (' + str(reference_landmarks.shape[0]) + ' in total):\n')
            for reference_landmark in reference_landmarks:
                summary_file.write(str(reference_landmark) + '\n')

    list_initial_momenta = []
    snapshots = []

    for mu in parameters:
        if not place_landmarks_automatically:
            get_landmarks = load_landmark_function(example)
            target_landmarks = get_landmarks(mu=mu)
        else:
            target_landmarks = place_landmarks_on_edges(u_ref.to_numpy(), num_landmarks)
        initial_momenta = None
        if kernel_dist_sigma == -1:
            kernel_dist_sigma = kernel_sigma
        result = gs.register(reference_landmarks, target_landmarks, initial_momenta=initial_momenta, sigma=sigma,
                             return_all=True, landmarks_labeled=landmarks_labeled,
                             kwargs_kernel_dist={"sigma": kernel_dist_sigma})
        registered_landmarks = result['registered_landmarks']
        initial_momenta = result['initial_momenta']
        list_initial_momenta.append(initial_momenta)

        filepath = f'{filepath_prefix}/mu_{str(mu).replace(".", "_")}/'
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

        time_evolution_momenta = result['time_evolution_momenta'].reshape((-1, len(registered_landmarks), gs.dim))
        time_evolution_positions = result['time_evolution_positions'].reshape((-1, len(registered_landmarks), gs.dim))

        flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta, reference_landmarks,
                                                            mins=mins, maxs=maxs, spatial_shape=u_ref.full_shape)

        u_mu = fom.solve(mu)
        snapshots.append(u_mu)
        u_red = u_ref.push_forward(flow)

        if write_results:
            # Compute error in transformed snapshot and write to file
            with open(results_filepath + '/registration_errors.txt', 'a') as f:
                f.write(f'{mu}\t'
                        f'{(u_mu-u_red).get_norm(restriction=restriction) / u_mu.get_norm(restriction=restriction)}\n')
            with open(filepath + 'initial_momenta.txt', 'w') as f:
                for l in initial_momenta:
                    f.write(f"{' '.join(str(x) for x in l)}\n")
                    f.write(f"{' '.join(str(x) for x in l)}\n")
            with open(filepath + 'target_landmarks.txt', 'w') as f:
                for l in target_landmarks:
                    f.write(f"{' '.join(str(x) for x in l)}\n")
            with open(filepath + 'registered_landmarks.txt', 'w') as f:
                for l in registered_landmarks:
                    f.write(f"{' '.join(str(x) for x in l)}\n")
            with open(filepath + 'target_snapshot.txt', 'w') as f:
                for x, y, z in zip(coords[0].flatten(), coords[1].flatten(), u_mu.flatten()):
                    f.write(f'{x}\t{y}\t{z}\n')
            with open(filepath + 'registered_snapshot.txt', 'w') as f:
                for x, y, z in zip(coords[0].flatten(), coords[1].flatten(), u_red.flatten()):
                    f.write(f'{x}\t{y}\t{z}\n')
            with open(filepath + 'difference_target_registered_snapshot.txt', 'w') as f:
                for x, y, z in zip(coords[0].flatten(), coords[1].flatten(), (u_red - u_mu).flatten()):
                    f.write(f'{x}\t{y}\t{z}\n')
            with open(filepath + 'landmarks_over_time.txt', 'w') as f:
                for t in time_evolution_positions:
                    for l in t:
                        f.write(f"{' '.join(str(x) for x in l)}\n")
                    f.write("\n\n")
            with open(filepath + 'momenta_over_time.txt', 'w') as f:
                for t in time_evolution_momenta:
                    for l in t:
                        f.write(f"{' '.join(str(x) for x in l)}\n")
                    f.write("\n\n")
            with open(filepath + 'initial_vector_field.txt', 'w') as f:
                vf = gs.get_vector_field(initial_momenta, reference_landmarks, mins=mins, maxs=maxs,
                                         spatial_shape=u_ref.full_shape)
                for x, y, u, v in zip(coords[0].flatten(), coords[1].flatten(), vf.to_numpy()[..., 0].flatten(),
                                      vf.to_numpy()[..., 1].flatten()):
                    f.write(f'{x}\t{y}\t{u}\t{v}\n')
            with open(filepath + 'transformation_as_warpgrid.txt', 'w') as f:
                for x, y, u, v in zip(coords[0].flatten(), coords[1].flatten(),
                                      flow.to_numpy()[..., 1].flatten() / flow.full_shape[1],
                                      flow.to_numpy()[..., 0].flatten() / flow.full_shape[0]):
                    f.write(f'{x}\t{y}\t{u}\t{v}\n')

    test_parameters = fom.parameter_space.sample(num_test_parameters, sampling_mode)

    # Learn mapping directly to momenta
    training_data = [(torch.Tensor([mu, ]), torch.Tensor(momenta.flatten())) for mu, momenta in
                     zip(parameters, list_initial_momenta)]
    random.shuffle(training_data)
    validation_data = training_data[:int(0.1 * len(training_data)) + 1]
    training_data = training_data[int(0.1 * len(training_data)) + 2:]

    layers_sizes = [fom.parameter_space.dim] + list(hidden_layers) + [list_initial_momenta[0].flatten().shape[0]]
    best_ann_momenta, best_loss_momenta = multiple_restarts_training(training_data, validation_data, layers_sizes,
                                                                     restarts, trainer_params, training_params)
    print(f"NN training momenta: {best_loss_momenta}")

    momenta_sum_absolute_error_restricted = 0.
    momenta_sum_relative_error_restricted = 0.
    momenta_sum_absolute_error = 0.
    momenta_sum_relative_error = 0.

    for mu in test_parameters:
        initial_momenta = best_ann_momenta(torch.Tensor([mu, ])).detach().numpy().flatten()
        flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta.reshape(reference_landmarks.shape),
                                                            reference_landmarks,
                                                            mins=mins, maxs=maxs, spatial_shape=u_ref.full_shape)
        transformed_input = u_ref.push_forward(flow)
        u = fom.solve(mu)
        absolute_error = (u - transformed_input).norm
        relative_error = absolute_error / u.norm
        if restriction:
            absolute_error_restricted = (u - transformed_input).get_norm(restriction=restriction)
            relative_error_restricted = absolute_error_restricted / u.get_norm(restriction=restriction)
        else:
            absolute_error_restricted = absolute_error
            relative_error_restricted = relative_error
        momenta_sum_absolute_error_restricted += absolute_error_restricted
        momenta_sum_relative_error_restricted += relative_error_restricted
        momenta_sum_absolute_error += absolute_error
        momenta_sum_relative_error += relative_error
        print(f"Relative error with trained initial momenta for parameter mu={mu}: "
              f"{relative_error_restricted}")
        if write_results:
            with open(f'{filepath_prefix}/test_errors_neural_network_momenta.txt', 'a') as f:
                f.write(f"{mu}\t{absolute_error_restricted}\t{relative_error_restricted}\t"
                        f"{absolute_error}\t{relative_error}\n")

    if write_results:
        num_params = len(test_parameters)
        with open(f'{filepath_prefix}/average_test_errors_neural_network_momenta.txt', 'a') as f:
            f.write(f"{momenta_sum_absolute_error_restricted / num_params}\t"
                    f"{momenta_sum_relative_error_restricted / num_params}\t"
                    f"{momenta_sum_absolute_error / num_params}\t{momenta_sum_relative_error / num_params}\n")

    singular_vectors, singular_values = pod(list_initial_momenta, num_modes=len(parameters),
                                            return_singular_values='all')

    if write_results:
        with open(f'{filepath_prefix}/singular_values_initial_momenta.txt', 'w') as f:
            for s in singular_values:
                f.write(f'{s}\n')

    for basis_size in range(1, len(singular_vectors)):
        reduced_initial_momenta = singular_vectors[:basis_size]
        snapshot_matrix = np.stack([a.flatten() for a in reduced_initial_momenta])
        prod_reduced_initial_momenta = np.stack([a.flatten() for a in list_initial_momenta])
        reduced_coefficients = snapshot_matrix.dot(prod_reduced_initial_momenta.T).T
        projected_initial_momenta = (snapshot_matrix.T.dot(reduced_coefficients.T)).T  # shape: (len(training_set), dim)

        if write_results:
            with open(f'{filepath_prefix}/test_errors.txt', 'a') as f:
                f.write(f"\n\nReduced basis size: {basis_size}\n")

        sum_absolute_error_restricted = 0.
        sum_relative_error_restricted = 0.
        sum_absolute_error = 0.
        sum_relative_error = 0.

        for (mu, initial_momenta, u) in zip(parameters, projected_initial_momenta, snapshots):
            flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta.reshape(reference_landmarks.shape),
                                                                reference_landmarks,
                                                                mins=mins, maxs=maxs, spatial_shape=u_ref.full_shape)
            transformed_input = u_ref.push_forward(flow)
            absolute_error = (u - transformed_input).norm
            relative_error = absolute_error / u.norm
            if restriction:
                absolute_error_restricted = (u - transformed_input).get_norm(restriction=restriction)
                relative_error_restricted = absolute_error_restricted / u.get_norm(restriction=restriction)
            else:
                absolute_error_restricted = absolute_error
                relative_error_restricted = relative_error
            sum_absolute_error_restricted += absolute_error_restricted
            sum_relative_error_restricted += relative_error_restricted
            sum_absolute_error += absolute_error
            sum_relative_error += relative_error
            print(f"Relative error with projected initial momenta for parameter mu={mu}: "
                  f"{relative_error_restricted}")
            if write_results:
                with open(f'{filepath_prefix}/test_errors.txt', 'a') as f:
                    f.write(f"{mu}\t{absolute_error_restricted}\t{relative_error_restricted}\t"
                            f"{absolute_error}\t{relative_error}\n")

        if write_results:
            num_params = len(parameters)
            with open(f'{filepath_prefix}/average_test_errors.txt', 'a') as f:
                f.write(f"{basis_size}\t{sum_absolute_error_restricted / num_params}\t"
                        f"{sum_relative_error_restricted / num_params}\t"
                        f"{sum_absolute_error / num_params}\t{sum_relative_error / num_params}\n")

        # Train neural network for reduced coefficients
        training_data = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for mu, coeff in
                         zip(parameters, reduced_coefficients)]
        random.shuffle(training_data)
        validation_data = training_data[:int(0.1 * len(training_data)) + 1]
        training_data = training_data[int(0.1 * len(training_data)) + 2:]

        #self.compute_normalization(training_data, validation_data)
        #training_data = self.normalize(training_data)
        #validation_data = self.normalize(validation_data)

        layers_sizes = [fom.parameter_space.dim] + list(hidden_layers) + [basis_size]
        best_ann, best_loss = multiple_restarts_training(training_data, validation_data, layers_sizes,
                                                         restarts, trainer_params, training_params)
        print(f"NN training: {basis_size}: {best_loss}")

        sum_absolute_error_restricted = 0.
        sum_relative_error_restricted = 0.
        sum_absolute_error = 0.
        sum_relative_error = 0
        for mu in test_parameters:
            initial_momenta = np.array(singular_vectors[:basis_size]).T @ best_ann(torch.Tensor([mu, ])).detach().numpy().flatten()
            flow = gs.compute_time_evolution_of_diffeomorphisms(initial_momenta.reshape(reference_landmarks.shape),
                                                                reference_landmarks,
                                                                mins=mins, maxs=maxs, spatial_shape=u_ref.full_shape)
            transformed_input = u_ref.push_forward(flow)
            u = fom.solve(mu)
            absolute_error = (u - transformed_input).norm
            relative_error = absolute_error / u.norm
            if restriction:
                absolute_error_restricted = (u - transformed_input).get_norm(restriction=restriction)
                relative_error_restricted = absolute_error_restricted / u.get_norm(restriction=restriction)
            else:
                absolute_error_restricted = absolute_error
                relative_error_restricted = relative_error
            sum_absolute_error_restricted += absolute_error_restricted
            sum_relative_error_restricted += relative_error_restricted
            sum_absolute_error += absolute_error
            sum_relative_error += relative_error
            print(f"Relative error with projected initial momenta for parameter mu={mu}: "
                  f"{relative_error_restricted}")
            if write_results:
                with open(f'{filepath_prefix}/test_errors_neural_network.txt', 'a') as f:
                    f.write(f"{mu}\t{absolute_error_restricted}\t{relative_error_restricted}\t"
                            f"{absolute_error}\t{relative_error}\n")

        if write_results:
            num_params = len(test_parameters)
            with open(f'{filepath_prefix}/average_test_errors_neural_network.txt', 'a') as f:
                f.write(f"{basis_size}\t{sum_absolute_error_restricted / num_params}\t"
                        f"{sum_relative_error_restricted / num_params}\t"
                        f"{sum_absolute_error / num_params}\t{sum_relative_error / num_params}\n")

if __name__ == "__main__":
    run(main)
