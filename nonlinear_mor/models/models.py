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
