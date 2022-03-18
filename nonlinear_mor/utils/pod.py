import numpy as np


def pod(A, modes=10, product_operator=None, return_singular_values=False):
    assert isinstance(modes, int) and modes > 0

    if product_operator:
        B = np.stack([a.to_numpy().flatten() for a in A], axis=0)
        mean = np.mean(B.reshape((B.shape[0], -1)), axis=0)
        C = np.sum([np.outer(a.to_numpy().flatten()-mean,
                             product_operator(a-mean.reshape(*a.full_shape)).to_numpy().flatten())
                    for a in A],
                   axis=0)
        S, U = np.linalg.eig(C)
        if return_singular_values:
            return np.real(U)[:, :min(modes, U.shape[1])], np.real(S)
        else:
            return np.real(U)[:, :min(modes, U.shape[1])]
    else:  # assuming L2-scalar product as default if no `product_operator` is provided
        assert A.ndim == 2
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        if return_singular_values:
            return U[:, :min(modes, U.shape[1])], S
        else:
            return U[:, :min(modes, U.shape[1])]
