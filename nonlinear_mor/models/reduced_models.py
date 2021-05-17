import torch


class ReducedSpacetimeModel:
    def __init__(self, reference_solution, reduced_velocity_fields, neural_network,
                 geodesic_shooter):
        self.reference_solution = reference_solution
        self.reduced_velocity_fields = reduced_velocity_fields
        self.neural_network = neural_network
        self.geodesic_shooter = geodesic_shooter

    def solve(self, mu):
        reduced_coefficients = self.neural_network(torch.Tensor([mu, ])).data.numpy()
        initial_velocity_field = self.reduced_velocity_fields.dot(reduced_coefficients)
        initial_velocity_field = initial_velocity_field.reshape((self.reference_solution.ndim,
                                                                 *self.reference_solution.shape))
        velocity_fields = self.geodesic_shooter.integrate_forward_vector_field(
                              initial_velocity_field)
        flow = self.geodesic_shooter.integrate_forward_flow(velocity_fields)
        mapped_solution = self.geodesic_shooter.push_forward(self.reference_solution, flow)
        return mapped_solution
