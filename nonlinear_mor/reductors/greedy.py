import numpy as np
import random
import torch
import pathlib

import geodesic_shooting
from geodesic_shooting.utils.helper_functions import lincomb, project
from geodesic_shooting.utils.reduced import pod

from nonlinear_mor.models import ReducedDictionaryModel
from nonlinear_mor.utils.logger import getLogger
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer
from nonlinear_mor.utils.versioning import get_git_hash, get_version


class GreedyDictionaryReductor:
    def __init__(self, fom, training_params, tol, geodesic_shooter):
        self.fom = fom
        self.training_params = training_params
        self.tol = tol
        self.geodesic_shooter = geodesic_shooter

        self.logger = getLogger('nonlinear_mor.GreedyDictionaryReductor')

    def write_summary(self, filepath_prefix='', registration_params={}, additional_text=""):
        pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
        with open(f'{filepath_prefix}/summary.txt', 'a') as summary_file:
            summary_file.write('========================================================\n')
            summary_file.write('Git hash: ' + get_git_hash() + '\n')
            summary_file.write('========================================================\n')
            summary_file.write('FOM: ' + str(self.fom) + '\n')
            summary_file.write('Reductor: ' + str(self.__class__.__name__) + '\n')
            summary_file.write('Geodesic Shooting:\n')
            summary_file.write('------------------\n')
            summary_file.write('Version: ' + get_version(geodesic_shooting) + '\n')
            summary_file.write(str(self.geodesic_shooter) + '\n')
            summary_file.write('------------------\n')
            summary_file.write('Registration parameters: ' + str(registration_params) + '\n')
            summary_file.write(additional_text)

    def reduce(self, max_basis_size=10, l2_prod=False, registration_params={}):
        if l2_prod:
            product_operator = None
        else:
            product_operator = self.geodesic_shooter.regularizer.cauchy_navier

        n = len(self.training_params)
        U = []
        V = []
        pi = [None]*n

        with self.logger.block("Computing full-order solutions for training parameters ..."):
            full_solutions = [self.fom.solve(mu) for mu in self.training_params]

        k = 0
        self.logger.info("Computing worst approximated parameter ...")
        norms = np.array([u.norm for u in full_solutions])
        k_opt = np.argmax(norms)
        U.append(full_solutions[k_opt])
        self.logger.info(f"Worst approximated parameter is mu={self.training_params[k_opt]} ...")

        with self.logger.block("Registering all snapshots ..."):
            self.logger.info(f"Need to register {n} snapshots ...")
            vs = [None]*n
            for i in range(n):
                self.logger.info(f"Running registration number {i} out of {n} ...")
                vs[i] = self.geodesic_shooter.register(U[k], full_solutions[i], **registration_params, return_all=False)

        with self.logger.block("Computing POD of vector fields ..."):
            V_k, singular_values = pod(vs, num_modes=min(max_basis_size, n-1), product_operator=product_operator,
                                       return_singular_values='all')
            V.append(V_k)

        with self.logger.block("Initializing assignment and computing reduced coefficients ..."):
            vs_red = [None]*n
            for i in range(n):
                pi[i] = k
                vs_red[i] = project(V_k, vs[i])

        with self.logger.block("Computing errors and set of solutions that are insufficiently approximated ..."):
            errors = [self._error(full_solutions[i], U[k], lincomb(V_k, vs_red[i])) for i in range(n)]
            Z = [i for i in range(n) if errors[i] > self.tol]

        len_Z = len(Z)
        while len_Z > 0:
            k = k + 1
            with self.logger.block(f"Greedy step number {k} ..."):
                self.logger.info("Computing worst approximated parameter ...")
                k_opt = np.argmax(errors)
                assert k_opt in Z
                U.append(full_solutions[k_opt])
                self.logger.info(f"Worst approximated parameter is mu={self.training_params[k_opt]} ...")

                vs_pod = []
                with self.logger.block("Registering all snapshots ..."):
                    self.logger.info(f"Need to register {len(Z)} snapshots ...")
                    for count, i in enumerate(Z):
                        self.logger.info(f"Running registration number {count} out of {len(Z)} ...")
                        vs[i] = self.geodesic_shooter.register(U[k], full_solutions[i],
                                                               **registration_params, return_all=False)
                        if self._error(full_solutions[i], U[k], vs[i]) <= self.tol:
                            vs_pod.append(vs[i])

                with self.logger.block("Computing POD of vector fields ..."):
                    V_k, singular_values = pod(vs_pod, num_modes=min(max_basis_size, len_Z),
                                               product_operator=product_operator, return_singular_values='all')
                    V.append(V_k)

                with self.logger.block("Computing reduced coefficients, "
                                       "errors and performing updates if necessary ..."):
                    for i in Z:
                        vs_red_tilde = project(V_k, vs[i])
                        new_error = self._error(full_solutions[i], U[k], lincomb(V_k, vs_red_tilde))
                        if new_error < errors[i]:
                            errors[i] = new_error
                            pi[i] = k
                            vs_red[i] = vs_red_tilde

                with self.logger.block("Computing set of solutions that are insufficiently approximated ..."):
                    Z = [i for i in range(n) if errors[i] > self.tol]
                    assert len(Z) < len_Z
                    len_Z = len(Z)

        N = k + 1
        with self.logger.block(f"Solving {N} regression problems ..."):
            Psis = []
            for k in range(N):
                self.logger.info(f"Running fitting process number {k} out of {N} ...")
                T_k = [(self.training_params[i], vs_red[i]) for i in range(n) if pi[i] == k]
                print(f"T_k: {T_k}")
                Psi_k = self._fit_regression_model(T_k)
                Psis.append(Psi_k)

        return self.build_rom(U, V, Psis, pi)

    def _error(self, u_full, u_ref, vf):
        velocity_fields = self.geodesic_shooter.integrate_forward_vector_field(vf)
        flow = self.geodesic_shooter.integrate_forward_flow(velocity_fields)
        u_red = u_ref.push_forward(flow)
        return (u_full - u_red).norm / u_full.norm

    def _fit_regression_model(self, training_data, restarts=10, trainer_params={},
                              hidden_layers=[20, 20, 20], training_params={}):
        self.logger.info("Approximating mapping from parameters to reduced coefficients ...")
        training_data = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for mu, coeff in training_data]
        random.shuffle(training_data)
        validation_ratio = 0.2
        validation_data = training_data[:int(validation_ratio * len(training_data)) + 1]
        training_data = training_data[int(validation_ratio * len(training_data)) + 2:]

        self.compute_normalization(training_data, validation_data)
        training_data = self.normalize(training_data)
        validation_data = self.normalize(validation_data)

        if len(validation_data) == 0:
            raise NotImplementedError
        elif len(training_data) == 0:
            training_data = validation_data

        layers_sizes = [1] + hidden_layers + [int(list(training_data[0][1].shape)[0])]

        print(f"layer sizes: {layers_sizes}")

        best_ann, best_loss = self.multiple_restarts_training(training_data, validation_data, layers_sizes,
                                                              restarts, trainer_params, training_params)

        return best_ann

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

    def build_rom(self, U, V, Psis, pi):
        return ReducedDictionaryModel(self.geodesic_shooter, U, V, Psis, pi, self.training_params,
                                      self.normalize_input, self.denormalize_output)
