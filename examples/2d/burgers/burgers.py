import math

from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import discretize_instationary_fv, RectGrid

from nonlinear_mor.models import WrappedpyMORModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps):
    assert spatial_shape[0] == 2 * spatial_shape[1]
    problem = burgers_problem_2d(vx=1., vy=1., initial_data_type='sin',
                                 parameter_range=(0, 1e42), torus=False)
    model, _ = discretize_instationary_fv(
        problem,
        diameter=math.sqrt(2.) / spatial_shape[1],
        grid_type=RectGrid,
        num_flux='engquist_osher',
        lxf_lambda=1.,
        nt=num_time_steps
    )
    parameter_space = CubicParameterSpace([(0, 10)])
    return WrappedpyMORModel(spatial_shape, num_time_steps, parameter_space, model)
