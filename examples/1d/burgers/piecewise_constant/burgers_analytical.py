from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.models import AnalyticalModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extend=[(0., 1.)], temporal_extend=(0., 1.)):
    def exact_solution(x, *, mu=1.):
        s_l = 1.5 * mu
        s_m = mu
        s_r = 0.5 * mu
        t_intersection = 0.25 / (s_l - s_r)
        return ScalarFunction(data=(2. * (x[..., 1] <= t_intersection) * (0.25 + s_l * x[..., 1] - x[..., 0] >= 0.)
                                    + (2. * (x[..., 1] > t_intersection)
                                       * (0.25 + (s_l - s_m) * t_intersection + s_m * x[..., 1] - x[..., 0] >= 0.))
                                    + (1. * (0.25 + s_l * x[..., 1] - x[..., 0] < 0.)
                                       * (0.5 + s_r * x[..., 1] - x[..., 0] > 0.))))

    parameter_space = CubicParameterSpace([(0.5, 1.5)])
    default_reference_parameter = 1.

    return AnalyticalModel(spatial_shape, num_time_steps, parameter_space, default_reference_parameter, exact_solution,
                           spatial_extend, temporal_extend, name='1dBurgersPiecewiseConstantAnalytical')
