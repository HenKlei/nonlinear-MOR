import numpy as np


def pod(A, modes=10, product_operator=None, return_singular_values=False):
    assert isinstance(modes, int) and modes > 0

    if product_operator:
        B = np.stack([a.to_numpy().flatten() for a in A])
        B_tilde = np.stack([product_operator(a).to_numpy().flatten() for a in A])
        C = B.dot(B_tilde.T)
        S, U = np.linalg.eig(C)
        U = U.T.dot(B)
        if return_singular_values:
            return np.real(U)[:min(modes, U.shape[0])], np.real(S)
        else:
            return np.real(U)[:min(modes, U.shape[0])]
    else:  # assuming L2-scalar product as default if no `product_operator` is provided
        assert A.ndim == 2
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        if return_singular_values:
            return U[:, :min(modes, U.shape[1])], S
        else:
            return U[:, :min(modes, U.shape[1])]
