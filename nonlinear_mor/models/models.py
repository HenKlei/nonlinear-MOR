import numpy as np

try:
    from clawpack import pyclaw
    PYCLAW = True
except ImportError:
    PYCLAW = False
    print("PyClaw not available.")

from geodesic_shooting.core import ScalarFunction

from nonlinear_mor.utils.logger import getLogger


class Model:
    def __init__(self, spatial_shape, num_time_steps, parameter_space, name=''):
        assert isinstance(spatial_shape, tuple)
        assert isinstance(num_time_steps, int) and num_time_steps > 0
        self.spatial_shape = spatial_shape
        self.dim = len(self.spatial_shape)
        self.num_time_steps = num_time_steps
        self.parameter_space = parameter_space
        self.name = name

        self.logger = getLogger(f'nonlinear_mor.{self.name}')

    def __str__(self):
        return self.name

    def solve(self, mu):
        raise NotImplementedError

    def visualize(self, u):
        raise NotImplementedError


class AnalyticalModel(Model):
    def __init__(self, spatial_shape=(100, ), num_time_steps=100, parameter_space=None, exact_solution=None,
                 spatial_extend=[(0., 1.), ], temporal_extend=(0., 1.), name='AnalyticalModel'):
        super().__init__(spatial_shape, num_time_steps, parameter_space, name=name)
        self.exact_solution = exact_solution
        self.spatial_extend = spatial_extend
        self.temporal_extend = temporal_extend

    def create_summary(self):
        return (str(self) + ':\n' +
                'Spatial extend: ' + str(self.spatial_extend) + '\n' +
                'Spatial shape: ' + str(self.spatial_shape) + '\n' +
                'Temporal extend: ' + str(self.temporal_extend) + '\n' +
                'Number of time steps: ' + str(self.num_time_steps))

    def solve(self, mu):
        assert mu in self.parameter_space

        linspace_list = [np.linspace(s_min, s_max, num) for (s_min, s_max), num in zip(self.spatial_extend,
                                                                                       self.spatial_shape)]
        temp = np.meshgrid(*linspace_list, np.linspace(self.temporal_extend[0], self.temporal_extend[1],
                                                       self.num_time_steps))
        coordinates = np.stack([temp_.T for temp_ in temp], axis=-1)

        with self.logger.block(f"Sampling analytical solution for mu={mu} ..."):
            result = self.exact_solution(coordinates, mu=mu)

        return result


class WrappedpyMORModel(Model):
    def __init__(self, spatial_shape=(100, ), num_time_steps=100, parameter_space=None, model=None,
                 name='WrappedpyMORModel'):
        super().__init__(spatial_shape, num_time_steps, parameter_space, name=name)
        self.model = model

    def create_summary(self):
        return (str(self) + ':\n' +
                'Spatial shape: ' + str(self.spatial_shape) + '\n' +
                'Number of time steps: ' + str(self.num_time_steps))

    def solve(self, mu):
        assert mu in self.parameter_space

        with self.logger.block(f"Calling pyMOR to solve for mu={mu} ..."):
            u = self.model.solve(mu).to_numpy()

        u = np.moveaxis(u.reshape((u.shape[0], *self.spatial_shape)), 0, -1)

        return ScalarFunction(data=u)

    def visualize(self, u):
        if not isinstance(u, np.ndarray):
            u = u.to_numpy()
        u = np.moveaxis(u, -1, 0)
        U = self.model.operator.range.from_numpy(u.reshape(u.shape[0], -1))
        self.model.visualize(U)


if PYCLAW:
    import logging
    logger = logging.getLogger('pyclaw')
    logger.setLevel(logging.CRITICAL)

    class WrappedPyClawModel(Model):
        def __init__(self, spatial_shape=(100, 100), num_time_steps=100, parameter_space=None,
                     spatial_extend=[(0., 1.), (0., 1.)], t_final=1., call_pyclaw=None, name='WrappedPyClawModel'):
            super().__init__(spatial_shape, num_time_steps, parameter_space, name=name)
            self.call_pyclaw = call_pyclaw
            lower_bounds = [x[0] for x in spatial_extend]
            upper_bounds = [x[1] for x in spatial_extend]
            self.domain = pyclaw.Domain(lower_bounds, upper_bounds, spatial_shape)
            self.claw = pyclaw.Controller()
            self.claw.tfinal = t_final
            self.claw.num_output_times = num_time_steps - 1

        def create_summary(self):
            return (str(self) + ':\n' +
                    'Spatial shape: ' + str(self.spatial_shape) + '\n' +
                    'Number of time steps: ' + str(self.num_time_steps))

        def solve(self, mu):
            assert mu in self.parameter_space

            with self.logger.block(f"Calling PyClaw to solve for mu={mu} ..."):
                self.claw.frames = []
                u = self.call_pyclaw(self.claw, self.domain, mu)

            assert u.shape == (self.num_time_steps, *self.spatial_shape)
            u = np.moveaxis(u, 0, -1)

            return ScalarFunction(data=u)

        def visualize(self, u):
            assert u.dim in (2, 3)
            u = np.moveaxis(u.to_numpy(), -1, 0)

            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            fig, axis = plt.subplots()

            if u.ndim == 2:
                x = self.claw.frames[0].state.grid.c_centers
                vals = axis.plot(x[0], u[0])[0]
            elif u.ndim == 3:
                x, y = self.claw.frames[0].state.grid.c_centers
                vals = axis.imshow(u[0].transpose(), vmin=u.min(), vmax=u.max(),
                                   extent=[x.min(), x.max(), y.min(), y.max()], interpolation='nearest', origin='lower')
                fig.colorbar(vals)

            def init():
                axis.set_title("Frame 0")
                return [vals]

            def update(frame):
                if u.ndim == 2:
                    vals.set_ydata(u[frame])
                elif u.ndim == 3:
                    vals.set_data(u[frame])

                axis.set_title(f"Frame {frame}")
                return [vals]

            ani = FuncAnimation(fig, update, frames=np.arange(0, u.shape[0]), init_func=init)  # noqa: F841
            return ani
