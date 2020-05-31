import numpy as np

def tolerance_svd(A, delta=10 ** -5):
    U, S, _ = np.linalg.svd(A, full_matrices=False)

    rk = len(S)
    er = 0
    delta *= delta
    for i in range(S.shape[0] - 1, 0, -1):
        er += S[i] ** 2
        if er > delta:
            rk = i + 1
            break
    return U[:, :rk]