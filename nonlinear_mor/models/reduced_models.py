import copy
import pathlib
import dill as pickle
import torch

from geodesic_shooting import ReducedGeodesicShooting
from geodesic_shooting.utils.helper_functions import lincomb

from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork


class ReducedSpacetimeModel:
    def __init__(self, reduced_vector_fields, neural_network_vector_fields,
                 snapshots, neural_network_snapshots, geodesic_shooter,
                 normalize_input_vector_fields, denormalize_output_vector_fields,
                 normalize_input_snapshots, denormalize_output_snapshots):
        self.reduced_vector_fields = reduced_vector_fields
        self.neural_network_vector_fields = neural_network_vector_fields
        self.snapshots = snapshots
        self.neural_network_snapshots = neural_network_snapshots
        self.geodesic_shooter = geodesic_shooter
        self.normalize_input_vector_fields = normalize_input_vector_fields
        self.denormalize_output_vector_fields = denormalize_output_vector_fields
        self.normalize_input_snapshots = normalize_input_snapshots
        self.denormalize_output_snapshots = denormalize_output_snapshots

    @classmethod
    def load_model(cls, model_dictionary):
        init_params = model_dictionary['neural_network_vector_fields_init_params']
        neural_network_vector_fields = FullyConnectedNetwork(**init_params)
        neural_network_vector_fields.load_state_dict(torch.load(model_dictionary['neural_network_vector_fields']))
        neural_network_vector_fields.eval()
        model_dictionary['neural_network_vector_fields'] = neural_network_vector_fields

        init_params = model_dictionary['neural_network_snapshots_init_params']
        neural_network_snapshots = FullyConnectedNetwork(**init_params)
        neural_network_snapshots.load_state_dict(torch.load(model_dictionary['neural_network_snapshots']))
        neural_network_snapshots.eval()
        model_dictionary['neural_network_snapshots'] = neural_network_snapshots
        del model_dictionary['neural_network_vector_fields_init_params']
        del model_dictionary['neural_network_snapshots_init_params']
        return cls(**model_dictionary)

    def solve(self, mu, save_intermediate_results=True, filepath_prefix='', interval=3, scale=2):
        normalized_mu_vf = self.normalize_input_vector_fields(torch.Tensor([mu, ]))
        normalized_reduced_coefficients_vf = self.neural_network_vector_fields(normalized_mu_vf).data.numpy()
        reduced_coefficients_vf = self.denormalize_output_vector_fields(normalized_reduced_coefficients_vf)
        if isinstance(self.geodesic_shooter, ReducedGeodesicShooting):
            initial_vector_field = reduced_coefficients_vf
        else:
            initial_vector_field = lincomb(self.reduced_vector_fields, reduced_coefficients_vf)
            if save_intermediate_results:
                filepath_tex = filepath_prefix + "/figures_tex"
                pathlib.Path(filepath_tex).mkdir(parents=True, exist_ok=True)
                initial_vector_field.save_tikz(f"{filepath_tex}/initial_vector_field_mu_"
                                               f"{str(mu).replace('.', '_')}.tex",
                                               title=f"Initial vector field for $\\mu={mu}$",
                                               interval=interval, scale=scale)
        vector_fields = self.geodesic_shooter.integrate_forward_vector_field(initial_vector_field)
        flow = self.geodesic_shooter.integrate_forward_flow(vector_fields)

        normalized_mu_s = self.normalize_input_snapshots(torch.Tensor([mu, ]))
        normalized_reduced_coefficients_s = self.neural_network_snapshots(normalized_mu_s).data.numpy()
        reduced_coefficients_s = self.denormalize_output_snapshots(normalized_reduced_coefficients_s)
        reduced_snapshots = lincomb(self.snapshots, reduced_coefficients_s)

        mapped_solution = reduced_snapshots.push_forward(flow)
        return mapped_solution

    def save_model(self, filepath_prefix):
        model_dictionary = copy.deepcopy(self.__dict__)

        model_dictionary['neural_network_vector_fields'] = filepath_prefix + '/neural_network_vector_fields.pt'
        torch.save(self.neural_network_vector_fields.state_dict(), model_dictionary['neural_network_vector_fields'])

        model_dictionary['neural_network_snapshots'] = filepath_prefix + '/neural_network_snapshots.pt'
        torch.save(self.neural_network_snapshots.state_dict(), model_dictionary['neural_network_snapshots'])

        init_params = self.neural_network_vector_fields.get_init_params()
        model_dictionary['neural_network_vector_fields_init_params'] = init_params
        init_params = self.neural_network_snapshots.get_init_params()
        model_dictionary['neural_network_snapshots_init_params'] = init_params

        with open(filepath_prefix + '/reduced_model.pickle', 'wb') as f:
            pickle.dump(model_dictionary, f)
