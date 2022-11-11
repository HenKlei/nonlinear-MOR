import numpy as np

from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.utils.logger import getLogger


class Model:
    def __init__(self, spatial_shape, num_time_steps, name=''):
        assert isinstance(spatial_shape, tuple)
        assert isinstance(num_time_steps, int) and num_time_steps > 0
        self.spatial_shape = spatial_shape
        self.dim = len(self.spatial_shape)
        self.num_time_steps = num_time_steps
        self.name = name

        self.logger = getLogger(f'nonlinear_mor.{self.name}')

    def __str__(self):
        return self.name

    def solve(self, mu):
        raise NotImplementedError

    def visualize(self, u):
        raise NotImplementedError


class AnalyticalModel(Model):
    def __init__(self, spatial_shape=(100, ), num_time_steps=100, exact_solution=None,
                 spatial_extend=[(0., 1.), ], temporal_extend=(0., 1.), parameter_space=None):
        super().__init__(spatial_shape, num_time_steps, name='AnalyticalModel')
        self.exact_solution = exact_solution
        self.spatial_extend = spatial_extend
        self.temporal_extend = temporal_extend
        self.parameter_space = parameter_space

    def create_summary(self):
        return (str(self) + ':\n' +
                'Spatial extend: ' + str(self.spatial_extend) + '\n' +
                'Spatial shape: ' + str(self.spatial_shape) +
                '\n' + 'Temporal extend: ' + str(self.temporal_extend) + '\n' + 'Temporal shape: ' +
                str(self.spatial_shape))

    def solve(self, mu):
        linspace_list = [np.linspace(s_min, s_max, num) for (s_min, s_max), num in zip(self.spatial_extend,
                                                                                       self.spatial_shape)]
        temp = np.meshgrid(*linspace_list, np.linspace(self.temporal_extend[0], self.temporal_extend[1],
                                                       self.num_time_steps))
        coordinates = np.stack([temp_.T for temp_ in temp], axis=-1)

        with self.logger.block(f"Sampling analytical solution for mu={mu} ..."):
            result = self.exact_solution(coordinates, mu=mu)

        return result


class WrappedpyMORModel(Model):
    def __init__(self, spatial_shape=(100, ), num_time_steps=100, model=None):
        super().__init__(spatial_shape, num_time_steps, name='WrappedpyMORModel')

        self.model = model

    def solve(self, mu):
        with self.logger.block(f"Calling pyMOR to solve for mu={mu} ..."):
            u = self.model.solve(mu).to_numpy()

        u = u.reshape((u.shape[0], *self.spatial_shape))

        return ScalarFunction(data=u)

    def visualize(self, u):
        u = u.to_numpy()
        U = self.model.operator.range.from_numpy(u.reshape(u.shape[0], -1))
        self.model.visualize(U)
