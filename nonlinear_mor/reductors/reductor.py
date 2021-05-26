import torch
import numpy as np
import matplotlib.pyplot as plt

import geodesic_shooting
from geodesic_shooting.utils.visualization import plot_vector_field

from nonlinear_mor.models import ReducedSpacetimeModel
from nonlinear_mor.utils import pod
from nonlinear_mor.utils.logger import getLogger
from nonlinear_mor.utils.torch.early_stopping import SimpleEarlyStoppingScheduler
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer


class NonlinearReductor:
    def __init__(self, fom, training_set, reference_parameter,
                 gs_smoothing_params={'alpha': 1000., 'exponent': 3}):
        self.fom = fom
        self.training_set = training_set
        self.reference_parameter = reference_parameter
        self.reference_solution = self.fom.solve(reference_parameter)

        self.geodesic_shooter = geodesic_shooting.GeodesicShooting(**gs_smoothing_params)

    def reduce(self, max_basis_size=1, return_all=True, restarts=10, save_intermediate_results=True,
               registration_params={'sigma': 0.1, 'epsilon': 0.1, 'iterations': 20}):
        assert isinstance(max_basis_size, int) and max_basis_size > 0
        assert isinstance(restarts, int) and restarts > 0

        logger = getLogger('nonlinear_mor.NonlinearReductor.reduce')

        with logger.block("Computing full solutions ..."):
            full_solutions = [(mu, self.fom.solve(mu)) for mu in self.training_set]

        with logger.block("Computing mappings and vector fields ..."):
            full_velocity_fields = []
            for (mu, u) in full_solutions:
#                v0 = self.geodesic_shooter.register(self.reference_solution, u,
#                                                    **registration_params, return_all=False)
                img, v0, _, _, _ = self.geodesic_shooter.register(self.reference_solution, u,
                                                                  **registration_params,
                                                                  return_all=True)

                if save_intermediate_results:
                    plt.matshow(u)
                    plt.savefig(f'intermediate_results/full_solution_mu_'
                                f'{str(mu).replace(".", "_")}.png')
                    plt.close()
                    plt.matshow(img)
                    plt.savefig(f'intermediate_results/mapped_solution_mu_'
                                f'{str(mu).replace(".", "_")}.png')
                    plt.close()
                    plot_vector_field(v0, title="Initial vector field (C2S)", interval=2)
                    plt.savefig('intermediate_results/full_vector_field_mu_'
                                f'{str(mu).replace(".", "_")}.png')
                    plt.close()
                    with open('intermediate_results/'
                              'relative_mapping_errors.txt', 'a') as errors_file:
                        errors_file.write(f"{mu}\t{np.linalg.norm(u - img) / np.linalg.norm(u)}\n")

                full_velocity_fields.append(v0.flatten())

        with logger.block("Reducing vector fields using POD ..."):
            full_velocity_fields = np.stack(full_velocity_fields, axis=-1)
            reduced_velocity_fields = pod(full_velocity_fields, modes=max_basis_size)

        logger.info("Computing reduced coefficients ...")
        reduced_coefficients = full_velocity_fields.T.dot(reduced_velocity_fields)

        logger.info("Approximating mapping from parameters to reduced coefficients ...")
        training_data = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for (mu, _), coeff in
                         zip(full_solutions, reduced_coefficients)]
        validation_data = training_data[:int(0.1 * len(training_data)) + 1]
        training_data = training_data[int(0.1 * len(training_data)) + 2:]
        layers_sizes = [1, 30, 30, reduced_coefficients.shape[1]]

        best_neural_network = None
        best_loss = None

        with logger.block(f"Performing {restarts} restarts of neural network training ..."):
            for _ in range(restarts):
                neural_network, loss = self.train_neural_network(layers_sizes, training_data,
                                                                 validation_data)
                if best_loss is None or best_loss > loss:
                    best_neural_network = neural_network
                    best_loss = loss

            logger.info(f"Trained neural network with loss of {best_loss} ...")

        logger.info("Building reduced model ...")
        rom = self.build_rom(reduced_velocity_fields, best_neural_network)

        if return_all:
            return rom, {'reduced_velocity_fields': reduced_velocity_fields,
                         'training_data': training_data,
                         'validation_data': validation_data}

        return rom

    def train_neural_network(self, layers_sizes, training_data, validation_data):
        neural_network = FullyConnectedNetwork(layers_sizes)
        trainer = Trainer(neural_network, es_scheduler=SimpleEarlyStoppingScheduler)
        best_loss, _ = trainer.train(training_data, validation_data)
        return trainer.network, best_loss

    def build_rom(self, velocity_fields, neural_network):
        rom = ReducedSpacetimeModel(self.reference_solution, velocity_fields, neural_network,
                                    self.geodesic_shooter)
        return rom
