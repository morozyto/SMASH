import numpy as np
import log
import tools


def tolerance_svd(A):
    delta = tools.SVD_TOLERANCE
    U, S, _ = np.linalg.svd(A, full_matrices=True)

    error = 0
    rank = len(S)
    for i in range(len(S) - 1, 0, -1):
        error += S[i] ** 2
        if error > delta ** 2:
            rank = i + 1
            break
    log.debug(f'tolerance_svd return {rank} columns from {A.shape[1]} with tolerance {delta}')
    return U[:, :rank]