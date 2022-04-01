import numpy as np
import torch
import tikzplotlib
import matplotlib.pyplot as plt
import pathlib

from geodesic_shooting.core import VectorField


class ReducedSpacetimeModel:
    def __init__(self, reference_solution, reduced_velocity_fields, neural_network,
                 geodesic_shooter, normalize_input, denormalize_output):
        self.reference_solution = reference_solution
        self.reduced_velocity_fields = reduced_velocity_fields
        self.neural_network = neural_network
        self.geodesic_shooter = geodesic_shooter
        self.normalize_input = normalize_input
        self.denormalize_output = denormalize_output

    def solve(self, mu, save_intermediate_results=True, filepath_prefix=''):
        normalized_mu = self.normalize_input(torch.Tensor([mu, ]))
        normalized_reduced_coefficients = self.neural_network(normalized_mu).data.numpy()
        reduced_coefficients = self.denormalize_output(normalized_reduced_coefficients)
        initial_velocity_field = self.reduced_velocity_fields.T.dot(reduced_coefficients)
        full_shape = (*self.reference_solution.spatial_shape, self.reference_solution.dim)
        initial_velocity_field = VectorField(data=initial_velocity_field.reshape(full_shape))
        if save_intermediate_results:
            filepath_pdf = filepath_prefix + "/figures_pdf"
            filepath_tex = filepath_prefix + "/figures_tex"
            pathlib.Path(filepath_pdf).mkdir(parents=True, exist_ok=True)
            pathlib.Path(filepath_tex).mkdir(parents=True, exist_ok=True)
            initial_velocity_field.plot(title=f"Initial vector field for $\mu={mu}$", scale=1)
            plt.savefig(f"{filepath_pdf}/initial_vector_field_mu_{str(mu).replace('.', '_')}.pdf")
            tikzplotlib.save(f"{filepath_tex}/initial_vector_field_mu_{str(mu).replace('.', '_')}.tex")
        print(f"Reduced coefficients: {reduced_coefficients}")
        product_operator = self.geodesic_shooter.regularizer.cauchy_navier
        print(f"V-norm: {np.sqrt(product_operator(initial_velocity_field).to_numpy().flatten().dot(initial_velocity_field.to_numpy().flatten()))}")
        print(f"l2-norm: {initial_velocity_field.norm}")
        velocity_fields = self.geodesic_shooter.integrate_forward_vector_field(initial_velocity_field)
        flow = self.geodesic_shooter.integrate_forward_flow(velocity_fields)
        mapped_solution = self.geodesic_shooter.push_forward(self.reference_solution, flow)
        return mapped_solution
