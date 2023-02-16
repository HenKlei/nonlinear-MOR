import copy
import pathlib
import dill as pickle
import torch

from geodesic_shooting import ReducedGeodesicShooting
from geodesic_shooting.utils.helper_functions import lincomb

from nonlinear_mor.utils.torch.neural_networks import *  # noqa: F401,F403


class ReducedSpacetimeModel:
    def __init__(self, reference_solution, reduced_velocity_fields, neural_network,
                 geodesic_shooter, normalize_input, denormalize_output):
        self.reference_solution = reference_solution
        self.reduced_velocity_fields = reduced_velocity_fields
        self.neural_network = neural_network
        self.geodesic_shooter = geodesic_shooter
        self.normalize_input = normalize_input
        self.denormalize_output = denormalize_output

    @classmethod
    def load_model(cls, model_dictionary):
        neural_network = torch.load(model_dictionary['neural_network'])
        neural_network.eval()
        model_dictionary['neural_network'] = neural_network
        return cls(**model_dictionary)

    def solve(self, mu, save_intermediate_results=True, filepath_prefix='', interval=3, scale=2):
        normalized_mu = self.normalize_input(torch.Tensor([mu, ]))
        normalized_reduced_coefficients = self.neural_network(normalized_mu).data.numpy()
        reduced_coefficients = self.denormalize_output(normalized_reduced_coefficients)
        if isinstance(self.geodesic_shooter, ReducedGeodesicShooting):
            initial_velocity_field = reduced_coefficients
        else:
            initial_velocity_field = lincomb(self.reduced_velocity_fields, reduced_coefficients)
            if save_intermediate_results:
                filepath_tex = filepath_prefix + "/figures_tex"
                pathlib.Path(filepath_tex).mkdir(parents=True, exist_ok=True)
                initial_velocity_field.save_tikz(f"{filepath_tex}/initial_vector_field_mu_"
                                                 f"{str(mu).replace('.', '_')}.tex",
                                                 title=f"Initial vector field for $\\mu={mu}$",
                                                 interval=interval, scale=scale)
        velocity_fields = self.geodesic_shooter.integrate_forward_vector_field(initial_velocity_field)
        flow = velocity_fields.integrate()
        mapped_solution = self.reference_solution.push_forward(flow)
        return mapped_solution

    def save_model(self, filepath_prefix):
        model_dictionary = copy.deepcopy(self.__dict__)
        model_dictionary['neural_network'] = filepath_prefix + '/neural_network.pt'
        torch.save(self.neural_network, model_dictionary['neural_network'])
        with open(filepath_prefix + '/model.pickle', 'wb') as f:
            pickle.dump(model_dictionary, f)
