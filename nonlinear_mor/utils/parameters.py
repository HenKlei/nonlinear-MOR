import numpy as np


class ParameterSpace:
    def __init__(self):
        pass


class CubicParameterSpace(ParameterSpace):
    def __init__(self, extends):
        super().__init__()
        self.extends = np.array(extends)
        self.dim = len(self.extends)

    def sample(self, num_samples=1, mode='uniform'):
        assert mode in ['uniform']
        assert isinstance(num_samples, int) and num_samples >= 1

        if mode == 'uniform':
            linspace_list = [np.linspace(p_min, p_max, num_samples) for p_min, p_max in self.extends]
            temp = np.meshgrid(*linspace_list)
            parameters = np.stack([temp_.T for temp_ in temp], axis=-1).reshape(-1, self.dim)
            assert parameters.shape == (num_samples**self.dim, self.dim)

        return parameters.squeeze()
