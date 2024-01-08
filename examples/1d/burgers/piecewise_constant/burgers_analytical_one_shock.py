from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.models import AnalyticalModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extend=[(0., 1.)], temporal_extend=(0., 1.)):
    def exact_solution(x, *, mu=1.):
        assert mu > 0.
        return ScalarFunction(data=(1. * (1./mu * (x[..., 0] - 0.25) - x[..., 1] <= 0.)))

    parameter_space = CubicParameterSpace([(0.25, 1.5)])

    return AnalyticalModel(spatial_shape, num_time_steps, parameter_space, exact_solution,
                           spatial_extend, temporal_extend, name='1dBurgersPiecewiseConstantAnalyticalOneShock')
