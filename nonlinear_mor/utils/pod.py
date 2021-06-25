import numpy as np


def pod(A, modes=10, return_singular_values=False):
    assert isinstance(modes, int) and modes > 0
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    if return_singular_values:
        return U[:, :min(modes, U.shape[1])], S
    else:
        return U[:, :min(modes, U.shape[1])]
