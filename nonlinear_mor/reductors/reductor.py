import torch
import numpy as np

import geodesic_shooting

from nonlinear_mor.models import ReducedSpacetimeModel
from nonlinear_mor.utils import pod
from nonlinear_mor.utils.logger import getLogger
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

    def reduce(self, max_basis_size=1,
               registration_params={'sigma': 0.1, 'epsilon': 0.1, 'iterations': 20}):
        assert isinstance(max_basis_size, int) and max_basis_size > 0

        logger = getLogger('nonlinear_mor.NonlinearReductor.reduce')

        logger.info("Compute full solutions ...")
        full_solutions = [(mu, self.fom.solve(mu)) for mu in self.training_set]

        with logger.block("Compute mappings and vector fields ..."):
            full_velocity_fields = []
            for (mu, u) in full_solutions:
                v0 = self.geodesic_shooter.register(self.reference_solution, u,
                                                    **registration_params, return_all=False)
                full_velocity_fields.append(v0.flatten())

        with logger.block("Reduce vector fields using POD ..."):
            full_velocity_fields = np.stack(full_velocity_fields, axis=-1)
            reduced_velocity_fields = pod(full_velocity_fields, modes=max_basis_size)

        logger.info("Compute reduced coefficients ...")
        reduced_coefficients = full_velocity_fields.T.dot(reduced_velocity_fields)

        logger.info("Approximate mapping from parameters to reduced coefficients ...")
        training_data = [(torch.Tensor([mu, ]), torch.Tensor(coeff)) for (mu, _), coeff in
                         zip(full_solutions, reduced_coefficients)]
        validation_data = training_data[:int(0.1 * len(training_data)) + 1]
        training_data = training_data[int(0.1 * len(training_data)) + 2:]
        layers_sizes = [1, 30, 30, reduced_coefficients.shape[1]]
        neural_network = FullyConnectedNetwork(layers_sizes)
        trainer = Trainer(neural_network)
        trainer.train(training_data, validation_data)

        logger.info("Build reduced model ...")
        rom = self.build_rom(reduced_velocity_fields, trainer.network)

        return rom

    def build_rom(self, velocity_fields, neural_network):
        rom = ReducedSpacetimeModel(self.reference_solution, velocity_fields, neural_network,
                                    self.geodesic_shooter)
        return rom
