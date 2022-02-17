import numpy as np


def pod(A, modes=10, product_operator=None, return_singular_values=False):
    assert isinstance(modes, int) and modes > 0

    if product_operator:
        mean = np.mean(A.reshape((A.shape[0], -1)), axis=0)
        C = np.sum([np.outer(a.flatten()-mean,
                             product_operator((a.flatten()-mean).reshape(*A.shape[1:])))
                    for a in A],
                   axis=0)
        S, U = np.linalg.eig(C)
        if return_singular_values:
            return U[:, :min(modes, U.shape[1])], S
        else:
            return U[:, :min(modes, U.shape[1])]
    else:  # assuming L2-scalar product as default if no `product_operator` is provided
        assert A.ndim == 2
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        if return_singular_values:
            return U[:, :min(modes, U.shape[1])], S
        else:
            return U[:, :min(modes, U.shape[1])]
