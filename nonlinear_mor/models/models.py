import numpy as np
from functools import partial

from nonlinear_mor.utils.logger import getLogger


class SpacetimeModel:
    def __init__(self, grid_operator, transformation, n_x=100, n_t=100):
        self.grid_operator = grid_operator
        self.transformation = transformation

        assert isinstance(n_x, int) and n_x > 0
        assert isinstance(n_t, int) and n_t > 0
        self.n_x = n_x
        self.n_t = n_t

    def solve(self, mu):
        logger = getLogger('nonlinear_mor.SpacetimeModel')

        logger.info(f"Reparametrizing flux functions for mu={mu} ...")
        self._reparametrize_flux_functions(mu)

        with logger.block("Running `solve`-method of grid operator ..."):
            u = self.grid_operator.solve()

        return u.sample_function_uniformly(self.transformation, n_x=self.n_x, n_t=self.n_t)

    def _reparametrize_flux_functions(self, mu):
        self.grid_operator.time_stepper.discretization.numerical_flux.flux = partial(
            self.grid_operator.time_stepper.discretization.numerical_flux.flux, mu=mu)
        self.grid_operator.time_stepper.discretization.numerical_flux.flux_derivative = partial(
            self.grid_operator.time_stepper.discretization.numerical_flux.flux_derivative, mu=mu)
        self.grid_operator.time_stepper.discretization.inverse_transformation = partial(
            self.grid_operator.time_stepper.discretization.inverse_transformation, mu=mu)

        self.transformation = partial(self.transformation, mu=mu)


class AnalyticalModel:
    def __init__(self, exact_solution, n_x=100, n_t=100, x_min=0., x_max=1., t_min=0., t_max=1.):
        self.exact_solution = exact_solution

        assert isinstance(n_x, int) and n_x > 0
        assert isinstance(n_t, int) and n_t > 0
        self.n_x = n_x
        self.n_t = n_t

        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max

    def solve(self, mu):
        logger = getLogger('nonlinear_mor.SpacetimeModel')

        XX, YY = np.meshgrid(np.linspace(self.x_min, self.x_max, self.n_x),
                             np.linspace(self.t_min, self.t_max, self.n_t))
        XY = np.stack([XX.T, YY.T], axis=-1)

        with logger.block("Sampling analytical solution for mu={mu} ..."):
            result = self.exact_solution(XY, mu=mu)

        return result
