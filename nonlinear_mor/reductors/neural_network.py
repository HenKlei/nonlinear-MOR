import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing
from copy import deepcopy

import geodesic_shooting
from geodesic_shooting.utils.visualization import plot_vector_field

from nonlinear_mor.models import ReducedSpacetimeModel
from nonlinear_mor.utils import pod
from nonlinear_mor.utils.logger import getLogger
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer


class NonlinearNeuralNetworkReductor:
    def __init__(self, fom, training_set, reference_parameter,
                 gs_smoothing_params={'alpha': 1000., 'exponent': 3}):
        self.fom = fom
        self.training_set = training_set
        self.reference_parameter = reference_parameter
        self.reference_solution = self.fom.solve(reference_parameter)

        self.geodesic_shooter = geodesic_shooting.GeodesicShooting(**gs_smoothing_params,
                                                                   dim=self.reference_solution.ndim,
                                                                   shape=self.reference_solution.shape)

        self.logger = getLogger('nonlinear_mor.NonlinearNeuralNetworkReductor.reduce')

    def compute_full_solutions(self, full_solutions_file=None):
        if full_solutions_file:
            with open(full_solutions_file, 'rb') as solution_file:
                return pickle.load(solution_file)
        return [(mu, self.fom.solve(mu)) for mu in self.training_set]

    def perform_single_registration(self, input_, initial_velocity_field=None, save_intermediate_results=True,
                                    registration_params={'sigma': 0.1, 'iterations': 20,
                                                         'parameters_line_search': {'max_stepsize': 1.,
                                                                                    'min_stepsize': 1e-4}}):
        assert len(input_) == 2
        mu, u = input_
        result = self.geodesic_shooter.register(self.reference_solution, u,
                                                initial_velocity_field=initial_velocity_field,
                                                **registration_params, return_all=True)

        v0 = result['initial_velocity_field']

        if save_intermediate_results:
            transformed_input = result['transformed_input']

            plt.matshow(u)
            plt.savefig(f'intermediate_results/full_solution_mu_'
                        f'{str(mu).replace(".", "_")}.png')
            plt.close()
            plt.matshow(transformed_input)
            plt.savefig(f'intermediate_results/mapped_solution_mu_'
                        f'{str(mu).replace(".", "_")}.png')
            plt.close()
            #plot_vector_field(v0, title="Initial vector field (C2S)", interval=2)
            plt.savefig('intermediate_results/full_vector_field_mu_'
                        f'{str(mu).replace(".", "_")}.png')
            plt.close()
            norm = np.linalg.norm(u - transformed_input) / np.linalg.norm(u)
            with open('intermediate_results/'
                      'relative_mapping_errors.txt', 'a') as errors_file:
                errors_file.write(f"{mu}\t{norm}\t{result['iterations']}\t{result['time']}\t"
                                  f"{result['reason_registration_ended']}\n")

        return v0.flatten()

    def register_full_solutions(self, full_solutions, save_intermediate_results=True,
                                registration_params={'sigma': 0.1, 'iterations': 20},
                                num_workers=1, full_velocity_fields_file=None,
                                reuse_vector_fields=True):
        if full_velocity_fields_file:
            with open(full_velocity_fields_file, 'rb') as velocity_fields_file:
                return pickle.load(velocity_fields_file)
        with self.logger.block("Computing mappings and vector fields ..."):
            if num_workers > 1:
                if reuse_vector_fields:
                    self.logger.warning(f"Reusing velocity fields not possible with {num_workers} workers ...")
                with multiprocessing.Pool(num_workers) as pool:
                    perform_registration = partial(self.perform_single_registration,
                                                   initial_velocity_field=None,
                                                   save_intermediate_results=save_intermediate_results,
                                                   registration_params=deepcopy(registration_params))
                    full_velocity_fields = pool.map(perform_registration, full_solutions)
            else:
                full_velocity_fields = []
                for i, (mu, u) in enumerate(full_solutions):
                    if reuse_vector_fields and i > 0:
                        initial_velocity_field = full_velocity_fields[-1]
                        self.logger.info("Reusing velocity field from previous registration ...")
                    else:
                        initial_velocity_field = None
                    full_velocity_fields.append(self.perform_single_registration((mu, u),
                                                initial_velocity_field=initial_velocity_field,
                                                save_intermediate_results=save_intermediate_results,
                                                registration_params=deepcopy(registration_params)))
        return np.stack(full_velocity_fields, axis=-1)

    def reduce(self, max_basis_size=1, return_all=True, restarts=10, save_intermediate_results=True,
               registration_params={}, trainer_params={}, training_params={}, num_workers=1,
               full_solutions_file=None, full_velocity_fields_file=None, reuse_vector_fields=True):
        assert isinstance(max_basis_size, int) and max_basis_size > 0
        assert isinstance(restarts, int) and restarts > 0

        with self.logger.block("Computing full solutions ..."):
            full_solutions = self.compute_full_solutions(full_solutions_file)

        full_velocity_fields = self.register_full_solutions(full_solutions,
                                                            save_intermediate_results,
                                                            registration_params, num_workers,
                                                            full_velocity_fields_file,
                                                            reuse_vector_fields)

        with self.logger.block("Reducing vector fields using POD ..."):
            reduced_velocity_fields, singular_values = pod(full_velocity_fields,
                                                           modes=max_basis_size,
                                                           return_singular_values=True)

        self.logger.info("Computing reduced coefficients ...")
        reduced_coefficients = full_velocity_fields.T.dot(reduced_velocity_fields)

        self.logger.info("Approximating mapping from parameters to reduced coefficients ...")
        training_data = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for (mu, _), coeff in
                         zip(full_solutions, reduced_coefficients)]
        random.shuffle(training_data)
        validation_data = training_data[:int(0.1 * len(training_data)) + 1]
        training_data = training_data[int(0.1 * len(training_data)) + 2:]

        self.compute_normalization(training_data, validation_data)
        training_data = self.normalize(training_data)
        validation_data = self.normalize(validation_data)

        layers_sizes = [1, 30, 30, reduced_coefficients.shape[1]]

        best_ann, best_loss = self.multiple_restarts_training(training_data, validation_data, layers_sizes,
                                                              restarts, trainer_params, training_params)

        self.logger.info("Building reduced model ...")
        rom = self.build_rom(reduced_velocity_fields, best_ann)

        if return_all:
            return rom, {'reduced_velocity_fields': reduced_velocity_fields,
                         'singular_values': singular_values,
                         'full_velocity_fields': full_velocity_fields,
                         'training_data': training_data,
                         'validation_data': validation_data,
                         'best_loss': best_loss}

        return rom

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
                if best_loss is None or best_loss > loss:
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

    def build_rom(self, velocity_fields, neural_network):
        rom = ReducedSpacetimeModel(self.reference_solution, velocity_fields, neural_network,
                                    self.geodesic_shooter, self.normalize_input,
                                    self.denormalize_output)
        return rom
