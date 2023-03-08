import numpy as np
import skimage.transform

from geodesic_shooting.core import ScalarFunction
from geodesic_shooting.utils.create_example_images import make_square

from nonlinear_mor.models import AnalyticalModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extend=[(0., 1.)], temporal_extend=(0., 1.)):
    assert len(spatial_extend) == 1

    def exact_solution(x, *, mu=1.):
        shape = x.shape[:-1]
        square = make_square(shape, np.array(shape) / 2, 40)
        return ScalarFunction(data=skimage.transform.rotate(square.to_numpy(), mu * 90.))

    parameter_space = CubicParameterSpace([(0., 1.)])

    return AnalyticalModel(spatial_shape, num_time_steps, parameter_space, exact_solution,
                           spatial_extend, temporal_extend)
