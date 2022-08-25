import numpy as np


def pod(A, modes=10, product_operator=None, return_singular_values=False):
    assert isinstance(modes, int) and modes > 0

    if product_operator:
        B = np.stack([a.to_numpy().flatten() for a in A])
        B_tilde = np.stack([product_operator(a).to_numpy().flatten() for a in A])
        C = B.dot(B_tilde.T)
        S, V = np.linalg.eig(C)
        selected_modes = min(modes, V.shape[0])
        S = np.sqrt(S[:selected_modes])
        V = V.T
        V = B.T.dot((V[:selected_modes] / S[:, np.newaxis]).T)
        V = V.T
        if return_singular_values:
            return np.real(V), np.real(S)
        else:
            return np.real(V)
    else:  # assuming L2-scalar product as default if no `product_operator` is provided
        A = np.stack([a.to_numpy().flatten() for a in A]).T
        assert A.ndim == 2
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        U = U.T
        if return_singular_values:
            return U[:min(modes, U.shape[1])], S
        else:
            return U[:min(modes, U.shape[1])]
