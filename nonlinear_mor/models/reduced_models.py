import torch

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

    def solve(self, mu):
        normalized_mu = self.normalize_input(torch.Tensor([mu, ]))
        normalized_reduced_coefficients = self.neural_network(normalized_mu).data.numpy()
        reduced_coefficients = self.denormalize_output(normalized_reduced_coefficients)
        initial_velocity_field = self.reduced_velocity_fields.T.dot(reduced_coefficients)
        initial_velocity_field = VectorField(data=initial_velocity_field.reshape((*self.reference_solution.spatial_shape,
            self.reference_solution.dim)))
        velocity_fields = self.geodesic_shooter.integrate_forward_vector_field(initial_velocity_field)
        flow = self.geodesic_shooter.integrate_forward_flow(velocity_fields)
        mapped_solution = self.geodesic_shooter.push_forward(self.reference_solution, flow)
        return mapped_solution
