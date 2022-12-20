import pickle
import random
import torch
import numpy as np
from functools import partial
import multiprocessing
from copy import deepcopy
import pathlib

import geodesic_shooting
from geodesic_shooting.utils.reduced import pod
from geodesic_shooting.utils.helper_functions import project

from nonlinear_mor.models import ReducedSpacetimeModel
from nonlinear_mor.utils.logger import getLogger
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer
from nonlinear_mor.utils.versioning import get_git_hash, get_version


class NonlinearNeuralNetworkReductor:
    def __init__(self, fom, training_set, reference_parameter,
                 gs_smoothing_params={'alpha': 1000., 'exponent': 3}):
        self.fom = fom
        self.training_set = training_set
        self.reference_parameter = reference_parameter
        self.reference_solution = self.fom.solve(reference_parameter)

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
                                    registration_params={'sigma': 0.1}, filepath_prefix='', interval=10):
        assert len(input_) == 2
        mu, u = input_
        result = self.geodesic_shooter.register(self.reference_solution, u,
                                                initial_vector_field=initial_vector_field,
                                                **registration_params, return_all=True)

        v0 = result['initial_vector_field']
        ts = result['transformed_input']

        if save_intermediate_results:
            filepath = filepath_prefix + '/intermediate_results'
            pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

            u.save(f'{filepath}/full_solution_mu_{str(mu).replace(".", "_")}.png')
            ts.save(f'{filepath}/transformed_solution_mu_{str(mu).replace(".", "_")}.png')
            (u - ts).save(f'{filepath}/difference_mu_{str(mu).replace(".", "_")}.png')
            v0.save(f'{filepath}/full_vector_field_mu_{str(mu).replace(".", "_")}.png',
                    plot_args={'title': '', 'interval': interval, 'color_length': False, 'show_axis': False,
                               'scale': None, 'axis': None, 'figsize': (20, 20)})
            norm = (u - ts).norm / u.norm
            with open(f'{filepath}/relative_mapping_errors.txt', 'a') as errors_file:
                errors_file.write(f"{mu}\t{norm}\t{result['iterations']}\t{result['time']}\n")

        return v0, ts

    def register_full_solutions(self, full_solutions, save_intermediate_results=True,
                                registration_params={'sigma': 0.1, 'iterations': 20},
                                num_workers=1, full_vector_fields_file=None,
                                reuse_vector_fields=True, filepath_prefix='', interval=10):
        if full_vector_fields_file:
            with open(full_vector_fields_file, 'rb') as vector_fields_file:
                return pickle.load(vector_fields_file)
        with self.logger.block("Computing mappings and vector fields ..."):
            if num_workers > 1:
                if reuse_vector_fields:
                    self.logger.warning(f"Reusing vector fields not possible with {num_workers} workers ...")
                with multiprocessing.Pool(num_workers) as pool:
                    perform_registration = partial(self.perform_single_registration,
                                                   initial_vector_field=None,
                                                   save_intermediate_results=save_intermediate_results,
                                                   registration_params=deepcopy(registration_params),
                                                   filepath_prefix=filepath_prefix,
                                                   interval=interval)
                    results = pool.map(perform_registration, full_solutions)
                    full_vector_fields = [r[0] for r in results]
                    transformed_snapshots = [r[1] for r in results]
            else:
                full_vector_fields = []
                transformed_snapshots = []
                for i, (mu, u) in enumerate(full_solutions):
                    if reuse_vector_fields and i > 0:
                        initial_vector_field = full_vector_fields[-1]
                        self.logger.info("Reusing vector field from previous registration ...")
                    else:
                        initial_vector_field = None
                    vf, ts = self.perform_single_registration((mu, u),
                                                              initial_vector_field=initial_vector_field,
                                                              save_intermediate_results=save_intermediate_results,
                                                              registration_params=deepcopy(registration_params),
                                                              filepath_prefix=filepath_prefix,
                                                              interval=interval)
                    full_vector_fields.append(vf)
                    transformed_snapshots.append(ts)
        return full_vector_fields, transformed_snapshots

    def reduce(self, basis_sizes_vector_fields=range(1, 11), l2_prod=False, basis_sizes_snapshots=range(1, 11),
               return_all=True, restarts=10, save_intermediate_results=True, registration_params={}, trainer_params={},
               hidden_layers_vf=[20, 20, 20], hidden_layers_s=[20, 20, 20], training_params={}, validation_ratio_vf=0.1,
               validation_ratio_s=0.1, num_workers=1, full_solutions_file=None, full_vector_fields_file=None,
               reuse_vector_fields=True, filepath_prefix='', interval=10):
        assert isinstance(restarts, int) and restarts > 0

        with self.logger.block("Computing full solutions ..."):
            full_solutions = self.compute_full_solutions(full_solutions_file)

        full_vector_fields, transformed_snapshots = self.register_full_solutions(full_solutions,
                                                                                 save_intermediate_results,
                                                                                 registration_params, num_workers,
                                                                                 full_vector_fields_file,
                                                                                 reuse_vector_fields,
                                                                                 filepath_prefix,
                                                                                 interval)

        with self.logger.block("Reducing vector fields using POD ..."):
            if l2_prod:
                product_operator = None
            else:
                product_operator = self.geodesic_shooter.regularizer.cauchy_navier
            all_reduced_vector_fields, singular_values_vfs = pod(full_vector_fields,
                                                                 num_modes=max(list(basis_sizes_vector_fields)),
                                                                 product_operator=product_operator,
                                                                 return_singular_values='all')

        with self.logger.block("Reducing transformed snapshots using POD ..."):
            ######
            # first apply inverse transformation before POD!!!
            ######
            inverse_transformed_snapshots = []
            for snapshot, vector_field in zip(transformed_snapshots, full_vector_fields):
                time_dependent_vector_field = self.geodesic_shooter.integrate_forward_vector_field(vector_field)
                transformation = self.geodesic_shooter.integrate_forward_flow(time_dependent_vector_field)
                inverse_transformed_snapshots.append(snapshot.push_backward(transformation))
            all_reduced_transformed_snapshots, singular_values_ts = pod(inverse_transformed_snapshots,
                                                                        num_modes=max(list(basis_sizes_snapshots)),
                                                                        product_operator=None,
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
            with open(f'{filepath}/singular_values_vector_fields.txt', 'a') as singular_values_vfs_file:
                for val in singular_values_vfs:
                    singular_values_vfs_file.write(f"{val}\n")
            with open(f'{filepath}/singular_values_transformed_snapshots.txt', 'a') as singular_values_ts_file:
                for val in singular_values_ts:
                    singular_values_ts_file.write(f"{val}\n")

        roms = []

        for basis_size_vf in basis_sizes_vector_fields:
            temp_roms = []
            for basis_size_s in basis_sizes_snapshots:
                reduced_vector_fields = all_reduced_vector_fields[:basis_size_vf]
                reduced_snapshots = all_reduced_transformed_snapshots[:basis_size_s]

                self.logger.info("Computing reduced coefficients for vector fields ...")
                snapshot_matrix = np.stack([a.flatten() for a in reduced_vector_fields])
                if l2_prod:
                    prod_reduced_vector_fields = np.stack([a.flatten() for a in full_vector_fields])
                else:
                    prod_reduced_vector_fields = np.stack([product_operator(a).flatten() for a in full_vector_fields])
                reduced_coefficients_vf = snapshot_matrix.dot(prod_reduced_vector_fields.T).T
                assert reduced_coefficients_vf.shape == (len(self.training_set), len(reduced_vector_fields))

                self.logger.info("Approximating mapping from parameters to reduced coefficients for vector fields ...")
                training_data_vf = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for mu, coeff in
                                    zip(self.training_set, reduced_coefficients_vf)]
                random.shuffle(training_data_vf)
                validation_data_vf = training_data_vf[:int(validation_ratio_vf * len(training_data_vf)) + 1]
                training_data_vf = training_data_vf[int(validation_ratio_vf * len(training_data_vf)) + 2:]

                self.compute_normalization_vector_fields(training_data_vf, validation_data_vf)
                training_data_vf = self.normalize_vector_fields(training_data_vf)
                validation_data_vf = self.normalize_vector_fields(validation_data_vf)

                layers_sizes_vf = [self.fom.parameter_space.dim] + list(hidden_layers_vf) + [basis_size_vf]

                best_ann_vf, best_loss_vf = self.multiple_restarts_training(training_data_vf, validation_data_vf,
                                                                            layers_sizes_vf, restarts, trainer_params,
                                                                            training_params)

                if not l2_prod:
                    for i, v in enumerate(reduced_vector_fields):
                        reduced_vector_fields[i] = v * norms[i]

                self.logger.info("Computing reduced coefficients for snapshots ...")
                reduced_coefficients_s = []
                for s in inverse_transformed_snapshots:
                    reduced_coefficients_s.append(project(reduced_snapshots, s))
                reduced_coefficients_s = np.array(reduced_coefficients_s)

                self.logger.info("Approximating mapping from parameters to reduced coefficients for snapshots ...")
                training_data_s = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for mu, coeff in
                                   zip(self.training_set, reduced_coefficients_s)]
                random.shuffle(training_data_s)
                validation_data_s = training_data_s[:int(validation_ratio_s * len(training_data_s)) + 1]
                training_data_s = training_data_s[int(validation_ratio_s * len(training_data_s)) + 2:]

                self.compute_normalization_snapshots(training_data_s, validation_data_s)
                training_data_s = self.normalize_snapshots(training_data_s)
                validation_data_s = self.normalize_snapshots(validation_data_s)

                layers_sizes_s = [self.fom.parameter_space.dim] + list(hidden_layers_s) + [basis_size_s]

                best_ann_s, best_loss_s = self.multiple_restarts_training(training_data_s, validation_data_s,
                                                                          layers_sizes_s, restarts, trainer_params,
                                                                          training_params)

                self.logger.info("Building reduced model ...")
                rom = self.build_rom(reduced_vector_fields, reduced_snapshots, best_ann_vf, best_ann_s)

                if return_all:
                    temp_roms.append((rom, {'training_data_vf': training_data_vf,
                                            'validation_data_vf': validation_data_vf,
                                            'best_loss_vf': best_loss_vf,
                                            'training_data_s': training_data_s,
                                            'validation_data_s': validation_data_s,
                                            'best_loss_s': best_loss_s}))
                else:
                    temp_roms.append(rom)
            roms.append(temp_roms)

        if return_all:
            return roms, {'all_reduced_vector_fields': all_reduced_vector_fields,
                          'all_reduced_transformed_snapshots': all_reduced_transformed_snapshots,
                          'singular_values_vfs': singular_values_vfs,
                          'singular_values_ts': singular_values_ts,
                          'full_vector_fields': full_vector_fields,
                          'transformed_snapshots': transformed_snapshots}

        return roms

    def compute_normalization_vector_fields(self, training_data, validation_data):
        self.min_input_vf = np.min([elem[0].numpy() for elem in training_data + validation_data])
        self.max_input_vf = np.max([elem[0].numpy() for elem in training_data + validation_data])
        self.min_output_vf = np.min([elem[1].numpy() for elem in training_data + validation_data])
        self.max_output_vf = np.max([elem[1].numpy() for elem in training_data + validation_data])

    def normalize_vector_fields(self, data):
        assert hasattr(self, 'min_input_vf') and hasattr(self, 'max_input_vf')
        assert hasattr(self, 'min_output_vf') and hasattr(self, 'max_output_vf')
        return [(self.normalize_input_vector_fields(elem[0]),
                 self.normalize_output_vector_fields(elem[1])) for elem in data]

    def normalize_input_vector_fields(self, data):
        return (data - self.min_input_vf) / (self.max_input_vf - self.min_input_vf)

    def normalize_output_vector_fields(self, data):
        return (data - self.min_output_vf) / (self.max_output_vf - self.min_output_vf)

    def denormalize_output_vector_fields(self, data):
        return data * (self.max_output_vf - self.min_output_vf) + self.min_output_vf

    def compute_normalization_snapshots(self, training_data, validation_data):
        self.min_input_s = np.min([elem[0].numpy() for elem in training_data + validation_data])
        self.max_input_s = np.max([elem[0].numpy() for elem in training_data + validation_data])
        self.min_output_s = np.min([elem[1].numpy() for elem in training_data + validation_data])
        self.max_output_s = np.max([elem[1].numpy() for elem in training_data + validation_data])

    def normalize_snapshots(self, data):
        assert hasattr(self, 'min_input_s') and hasattr(self, 'max_input_s')
        assert hasattr(self, 'min_output_s') and hasattr(self, 'max_output_s')
        return [(self.normalize_input_snapshots(elem[0]),
                 self.normalize_output_snapshots(elem[1])) for elem in data]

    def normalize_input_snapshots(self, data):
        return (data - self.min_input_s) / (self.max_input_s - self.min_input_s)

    def normalize_output_snapshots(self, data):
        return (data - self.min_output_s) / (self.max_output_s - self.min_output_s)

    def denormalize_output_snapshots(self, data):
        return data * (self.max_output_s - self.min_output_s) + self.min_output_s

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

    def train_neural_network(self, layers_sizes, training_data, validation_data, trainer_params={}, training_params={}):
        neural_network = FullyConnectedNetwork(layers_sizes)
        trainer = Trainer(neural_network, **trainer_params)
        best_loss, _ = trainer.train(training_data, validation_data, **training_params)
        return trainer.network, best_loss

    def build_rom(self, vector_fields, snapshots, neural_network_vector_fields, neural_network_snapshots):
        rom = ReducedSpacetimeModel(vector_fields, neural_network_vector_fields,
                                    snapshots, neural_network_snapshots, self.geodesic_shooter,
                                    self.normalize_input_vector_fields, self.denormalize_output_vector_fields,
                                    self.normalize_input_snapshots, self.denormalize_output_snapshots)
        return rom
