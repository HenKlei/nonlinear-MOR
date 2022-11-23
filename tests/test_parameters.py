import numpy as np

from nonlinear_mor.utils.parameters import CubicParameterSpace


def test_parameters():
    extends = [(-1., 3.)]
    parameter_space = CubicParameterSpace(extends)
    assert np.array([0.5]) in parameter_space
    assert np.array([-1.1]) not in parameter_space

    samples = parameter_space.sample(num_samples=100, mode='uniform')
    assert len(samples) == 100
    assert all(samples == np.linspace(extends[0][0], extends[0][1], 100))

    extends = [(0., 1.), (-1., 2.)]
    parameter_space = CubicParameterSpace(extends)
    assert np.array([0.5, 0.]) in parameter_space
    assert np.array([-1., 1.]) not in parameter_space

    samples = parameter_space.sample(num_samples=100, mode='uniform')
    assert len(samples) == 100
