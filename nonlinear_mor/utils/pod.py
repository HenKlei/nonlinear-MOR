import numpy as np


def pod(A, modes=10):
    assert isinstance(modes, int) and modes > 0
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, :min(modes, U.shape[1])]
