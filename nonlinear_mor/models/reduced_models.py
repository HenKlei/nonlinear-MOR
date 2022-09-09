import torch
import pathlib

from geodesic_shooting import ReducedGeodesicShooting
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

    def solve(self, mu, save_intermediate_results=True, filepath_prefix='', interval=3, scale=2):
        normalized_mu = self.normalize_input(torch.Tensor([mu, ]))
        normalized_reduced_coefficients = self.neural_network(normalized_mu).data.numpy()
        reduced_coefficients = self.denormalize_output(normalized_reduced_coefficients)
        if isinstance(self.geodesic_shooter, ReducedGeodesicShooting):
            initial_velocity_field = reduced_coefficients
        else:
            initial_velocity_field = self.reduced_velocity_fields.T.dot(reduced_coefficients)
            full_shape = (*self.reference_solution.spatial_shape, self.reference_solution.dim)
            initial_velocity_field = VectorField(data=initial_velocity_field.reshape(full_shape))
            if save_intermediate_results:
                filepath_tex = filepath_prefix + "/figures_tex"
                pathlib.Path(filepath_tex).mkdir(parents=True, exist_ok=True)
                initial_velocity_field.save_tikz(f"{filepath_tex}/initial_vector_field_mu_{str(mu).replace('.', '_')}.tex",
                                                 title=f"Initial vector field for $\\mu={mu}$",
                                                 interval=interval, scale=scale)
        velocity_fields = self.geodesic_shooter.integrate_forward_vector_field(initial_velocity_field)
        flow = self.geodesic_shooter.integrate_forward_flow(velocity_fields)
        mapped_solution = self.reference_solution.push_forward(flow)
        return mapped_solution
