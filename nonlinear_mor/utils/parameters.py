import numpy as np
import numbers

from nonlinear_mor.utils.logger import getLogger


class ParameterSpace:
    def __init__(self):
        pass

    def __contains__(self, mu):
        raise NotImplementedError

    def sample(self, num_samples, mode):
        raise NotImplementedError


class CubicParameterSpace(ParameterSpace):
    def __init__(self, extends):
        super().__init__()
        assert len(extends) > 0
        assert all(lower < upper for (lower, upper) in extends)
        self.extends = np.array(extends)
        self.dim = len(self.extends)

        self.logger = getLogger('nonlinear_mor.CubicParameterSpace')

    def __contains__(self, mu):
        if isinstance(mu, numbers.Number):
            if self.dim != 1:
                return False
            return self.extends[0][0] <= mu <= self.extends[0][1]
        if not hasattr(mu, 'shape'):
            mu = np.array(mu)
        if mu.shape != (self.dim, ):
            return False
        return all(self.extends[i][0] <= mu[i] <= self.extends[i][1] for i in range(self.dim))

    def sample(self, num_samples=1, mode='uniform'):
        assert mode in ['uniform']
        assert isinstance(num_samples, int) and num_samples > 0

        if mode == 'uniform':
            number_of_parameters_per_dimension = int(np.ceil(num_samples**(1/self.dim)))
            self.logger.info(f"Sampling {number_of_parameters_per_dimension**self.dim} parameters uniformly ...")
            linspace_list = [np.linspace(p_min, p_max, number_of_parameters_per_dimension)
                             for p_min, p_max in self.extends]
            temp = np.meshgrid(*linspace_list)
            parameters = np.stack([temp_.T for temp_ in temp], axis=-1).reshape(-1, self.dim)
            assert parameters.shape == (number_of_parameters_per_dimension**self.dim, self.dim)

        return parameters.squeeze()
