import pickle
import random
import torch
import numpy as np
from functools import partial
import multiprocessing
from copy import deepcopy
import pathlib

import geodesic_shooting
from geodesic_shooting.core import TimeDependentVectorField
from geodesic_shooting.utils.reduced import pod
from geodesic_shooting.utils.summary import save_plots_registration_results

from nonlinear_mor.models import ReducedSpacetimeModel
from nonlinear_mor.utils.logger import getLogger
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer
from nonlinear_mor.utils.versioning import get_git_hash, get_version


class NonlinearNeuralNetworkReductor:
    def __init__(self, fom, training_set, reference_parameter,
                 gs_smoothing_params={'alpha': 1000., 'exponent': 3},
                 restriction=np.s_[...], value_on_oversampling=None):
        self.fom = fom
        self.training_set = training_set
        self.reference_parameter = reference_parameter
        self.reference_solution = self.fom.solve(reference_parameter)

        if value_on_oversampling is not None:
            mask = np.ones(self.reference_solution.full_shape, bool)
            mask[restriction] = 0
            self.reference_solution[mask] = value_on_oversampling

        self.geodesic_shooter = geodesic_shooting.GeodesicShooting(**gs_smoothing_params)

        self.logger = getLogger('nonlinear_mor.NonlinearNeuralNetworkReductor')

    def create_summary(self, registration_params={}):
        summary = '========================================================\n'
        summary += 'Git hash of nonlinear_mor-module: ' + get_git_hash() + '\n'
        summary += '========================================================\n'
        summary += 'FOM: ' + str(self.fom) + '\n'
        summary += 'Reductor: ' + str(self.__class__.__name__) + '\n'
        summary += 'Geodesic Shooting:\n'
        summary += '------------------\n'
        summary += 'Version: ' + get_version(geodesic_shooting) + '\n'
        summary += str(self.geodesic_shooter) + '\n'
        summary += '------------------\n'
        summary += 'Registration parameters: ' + str(registration_params) + '\n'
        summary += '------------------\n'
        summary += 'Reference parameter: ' + str(self.reference_parameter) + '\n'
        summary += 'Training parameters (' + str(len(self.training_set)) + '): ' + str(self.training_set) + '\n'
        return summary

    def compute_full_solutions(self, full_solutions_file=None):
        if full_solutions_file:
            with open(full_solutions_file, 'rb') as solution_file:
                return pickle.load(solution_file)
        return [(mu, self.fom.solve(mu)) for mu in self.training_set]

    def perform_single_registration(self, input_, initial_vector_field=None, save_intermediate_results=True,
                                    registration_params={}, filepath_prefix='', interval=10):
        assert len(input_) == 2
        mu, u = input_
        result = self.geodesic_shooter.register(self.reference_solution, u,
                                                initial_vector_field=initial_vector_field,
                                                **registration_params, return_all=True)

        v0 = result['initial_vector_field']

        if save_intermediate_results:
            filepath = filepath_prefix + '/intermediate_results'
            pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
            transformed_input = result['transformed_input']
            mu_as_string = str(mu).replace(".", "_")
            save_plots_registration_results(result, filepath=f'{filepath}/mu_{mu_as_string}/', postfix=f' mu={mu}')

            absolute_error = (u - transformed_input).norm
            relative_error = absolute_error / u.norm
            restriction = registration_params.get('restriction')
            if restriction:
                absolute_error_restricted = (u - transformed_input).get_norm(restriction=restriction)
                relative_error_restricted = absolute_error_restricted / u.get_norm(restriction=restriction)
            else:
                absolute_error_restricted = absolute_error
                relative_error_restricted = relative_error
            with open(f'{filepath}/relative_mapping_errors.txt', 'a') as errors_file:
                errors_file.write(f"{mu}\t{absolute_error_restricted}\t{relative_error_restricted}\t"
                                  f"{absolute_error}\t{relative_error}\t"
                                  f"{result['iterations']}\t{result['time']}\t{result['reason_registration_ended']}\t"
                                  f"{result['energy_regularizer']}\t{result['energy_intensity_unscaled']}\t"
                                  f"{result['energy_intensity']}\t{result['energy']}\t{result['norm_gradient']}\n")

        return v0

    def register_full_solutions(self, full_solutions, save_intermediate_results=True, registration_params={},
                                num_workers=1, full_vector_fields_file=None, reuse_vector_fields=True,
                                filepath_prefix='', interval=10):
        if full_vector_fields_file:
            with open(full_vector_fields_file, 'rb') as vector_fields_file:
                return pickle.load(vector_fields_file)

        filepath = filepath_prefix + '/intermediate_results'
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
        with open(f'{filepath}/relative_mapping_errors.txt', 'a') as errors_file:
            errors_file.write("Parameter\tAbsolute error on restriction\tRelative error on restriction\t"
                              "Absolute error on full domain\tRelative error on full domain\t"
                              "Number of iterations\tRequired time for registration\t"
                              "Reason for registration algorithm to stop\tEnergy regularizer\t"
                              "Energy intensity unscaled\tEnergy intensity\tEnergy\tNorm of gradient\n")

        with self.logger.block("Computing mappings and vector fields ..."):
            if num_workers > 1:
                if reuse_vector_fields:
                    self.logger.warning(f"Reusing vector fields not possible with {num_workers} workers ...")
                with multiprocessing.Pool(num_workers) as pool:
                    exact_solution = None
                    if hasattr(self.fom, 'exact_solution'):
                        exact_solution = self.fom.exact_solution
                        del self.fom.exact_solution  # necessary since otherwise pickling is not possible
                    perform_registration = partial(self.perform_single_registration,
                                                   initial_vector_field=None,
                                                   save_intermediate_results=save_intermediate_results,
                                                   registration_params=deepcopy(registration_params),
                                                   filepath_prefix=filepath_prefix,
                                                   interval=interval)
                    full_vector_fields = pool.map(perform_registration, full_solutions)
                    if exact_solution is not None:
                        self.fom.exact_solution = exact_solution
            else:
                full_vector_fields = []
                for i, (mu, u) in enumerate(full_solutions):
                    if reuse_vector_fields and i > 0:
                        initial_vector_field = full_vector_fields[-1]
                        self.logger.info("Reusing vector field from previous registration ...")
                    else:
                        initial_vector_field = None
                    full_vector_fields.append(self.perform_single_registration((mu, u),
                                              initial_vector_field=initial_vector_field,
                                              save_intermediate_results=save_intermediate_results,
                                              registration_params=deepcopy(registration_params),
                                              filepath_prefix=filepath_prefix,
                                              interval=interval))
        return full_vector_fields

    def reduce(self, basis_sizes=range(1, 11), l2_prod=False, return_all=True, restarts=10,
               save_intermediate_results=True, registration_params={}, trainer_params={}, hidden_layers=[20, 20, 20],
               training_params={}, num_workers=1, full_solutions_file=None, full_vector_fields_file=None,
               reuse_vector_fields=True, filepath_prefix='', interval=10):
        assert isinstance(restarts, int) and restarts > 0

        with self.logger.block("Computing full solutions ..."):
            full_solutions = self.compute_full_solutions(full_solutions_file)

        full_vector_fields = self.register_full_solutions(full_solutions,
                                                          save_intermediate_results,
                                                          registration_params, num_workers,
                                                          full_vector_fields_file,
                                                          reuse_vector_fields,
                                                          filepath_prefix,
                                                          interval)

        if save_intermediate_results:
            filepath = filepath_prefix + '/intermediate_results'
            pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

            parameter_dependent_vector_field = TimeDependentVectorField(data=full_vector_fields)
            ani = parameter_dependent_vector_field.animate(title="Parameter dependent vector field", interval=interval)
            try:
                ani.save(f'{filepath}/parameter_dependent_vector_field.gif', writer='imagemagick',
                         fps=max(1, len(self.training_set) // 10))
            except Exception as e:
                self.logger.warning(f"Could not save animation! Error: {e}")
            ani = parameter_dependent_vector_field.get_magnitude_series().animate(title="Magnitude of parameter-"
                                                                                        "dependent vector field")
            try:
                ani.save(f'{filepath}/magnitude_parameter_dependent_vector_field.gif', writer='imagemagick',
                         fps=max(1, len(self.training_set) // 10))
            except Exception as e:
                self.logger.warning(f"Could not save animation! Error: {e}")

        with self.logger.block("Reducing vector fields using POD ..."):
            if l2_prod:
                product_operator = None
            else:
                product_operator = self.geodesic_shooter.regularizer.cauchy_navier
            all_reduced_vector_fields, singular_values = pod(full_vector_fields,
                                                             num_modes=max(list(basis_sizes)),
                                                             product_operator=product_operator,
                                                             return_singular_values='all')

        if not l2_prod:
            norms = []
            for i, v in enumerate(all_reduced_vector_fields):
                v_norm = np.sqrt(product_operator(v).flatten().dot(v.flatten()))
                norms.append(v_norm)
                all_reduced_vector_fields[i] = v / v_norm

        if save_intermediate_results:
            filepath = filepath_prefix + '/intermediate_results'
            pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
            with open(f'{filepath}/singular_values.txt', 'a') as singular_values_file:
                for val in singular_values:
                    singular_values_file.write(f"{val}\n")

        roms = []

        for basis_size in basis_sizes:
            reduced_vector_fields = all_reduced_vector_fields[:basis_size]
            self.logger.info("Computing reduced coefficients ...")
            snapshot_matrix = np.stack([a.flatten() for a in reduced_vector_fields])
            if l2_prod:
                prod_reduced_vector_fields = np.stack([a.flatten() for a in full_vector_fields])
            else:
                prod_reduced_vector_fields = np.stack([product_operator(a).flatten() for a in full_vector_fields])
            reduced_coefficients = snapshot_matrix.dot(prod_reduced_vector_fields.T).T
            assert reduced_coefficients.shape == (len(self.training_set), len(reduced_vector_fields))

            self.logger.info("Approximating mapping from parameters to reduced coefficients ...")
            training_data = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for mu, coeff in
                             zip(self.training_set, reduced_coefficients)]
            random.shuffle(training_data)
            validation_data = training_data[:int(0.1 * len(training_data)) + 1]
            training_data = training_data[int(0.1 * len(training_data)) + 2:]

            self.compute_normalization(training_data, validation_data)
            training_data = self.normalize(training_data)
            validation_data = self.normalize(validation_data)

            layers_sizes = [self.fom.parameter_space.dim] + list(hidden_layers) + [basis_size]

            best_ann, best_loss = self.multiple_restarts_training(training_data, validation_data, layers_sizes,
                                                                  restarts, trainer_params, training_params)

            if not l2_prod:
                for i, v in enumerate(reduced_vector_fields):
                    reduced_vector_fields[i] = v * norms[i]

            self.logger.info("Building reduced model ...")
            rom = self.build_rom(reduced_vector_fields, best_ann)

            if return_all:
                roms.append((rom, {'training_data': training_data,
                                   'validation_data': validation_data,
                                   'best_loss': best_loss}))
            else:
                roms.append(rom)

        if return_all:
            return roms, {'all_reduced_vector_fields': all_reduced_vector_fields,
                          'singular_values': singular_values,
                          'full_vector_fields': full_vector_fields}

        return roms

    def compute_normalization(self, training_data, validation_data):
        self.min_input = np.min([elem[0].numpy() for elem in training_data + validation_data])
        self.max_input = np.max([elem[0].numpy() for elem in training_data + validation_data])
        self.min_output = np.min([elem[1].numpy() for elem in training_data + validation_data])
        self.max_output = np.max([elem[1].numpy() for elem in training_data + validation_data])

    def normalize(self, data):
        assert hasattr(self, 'min_input') and hasattr(self, 'max_input')
        assert hasattr(self, 'min_output') and hasattr(self, 'max_output')
        return [(self.normalize_input(elem[0]), self.normalize_output(elem[1])) for elem in data]

    def normalize_input(self, data):
        return (data - self.min_input) / (self.max_input - self.min_input)

    def normalize_output(self, data):
        return (data - self.min_output) / (self.max_output - self.min_output)

    def denormalize_output(self, data):
        return data * (self.max_output - self.min_output) + self.min_output

    def multiple_restarts_training(self, training_data, validation_data, layers_sizes, restarts,
                                   trainer_params={}, training_params={}):
        best_neural_network = None
        best_loss = None

        with self.logger.block(f"Performing {restarts} restarts of neural network training ..."):
            for _ in range(restarts):
                neural_network, loss = self.train_neural_network(layers_sizes, training_data,
                                                                 validation_data,
                                                                 trainer_params, training_params)
                if best_loss is None or loss is None or best_loss > loss:
                    best_neural_network = neural_network
                    best_loss = loss

            self.logger.info(f"Trained neural network with loss of {best_loss} ...")

        return best_neural_network, best_loss

    def train_neural_network(self, layers_sizes, training_data, validation_data,
                             trainer_params={}, training_params={}):
        neural_network = FullyConnectedNetwork(layers_sizes)
        trainer = Trainer(neural_network, **trainer_params)
        best_loss, _ = trainer.train(training_data, validation_data, **training_params)
        return trainer.network, best_loss

    def build_rom(self, vector_fields, neural_network):
        rom = ReducedSpacetimeModel(self.reference_solution, vector_fields, neural_network,
                                    self.geodesic_shooter, self.normalize_input,
                                    self.denormalize_output)
        return rom
