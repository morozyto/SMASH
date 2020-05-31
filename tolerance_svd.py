import numpy as np


def tolerance_svd(A, delta=10 ** -5):
    U, S, _ = np.linalg.svd(A, full_matrices=True)

    error = 0
    rank = len(S)
    for i in range(S.shape[0] - 1, 0, -1):
        error += S[i] ** 2
        if error > delta ** 2:
            rank = i + 1
            break
    return U[:, :rank]