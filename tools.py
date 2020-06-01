import numpy as np
import log
import random

from scipy.linalg import block_diag, rq, qr

SEPARATION_RATIO = None
APPROXIMATION_RANK = None


def concat_row_wise(a, b):
    assert a.shape[1] == b.shape[1]
    return np.concatenate((a, b), axis=0)


def concat_column_wise(a, b):
    assert a.shape[0] == b.shape[0]
    return np.concatenate((a, b), axis=1)


def get_block(A, i, j):
    return A[np.array(i)[:, np.newaxis], np.array(j)]


def diag(matrices):
    return block_diag(*matrices)


def get_metadata(points):
    assert points
    min_val, max_val = min(points), max(points)
    center = (max_val + min_val) / 2
    radius = max_val - center
    return center, radius


def count_constants(dimensions_count, tolerance):
    assert dimensions_count > 0
    global SEPARATION_RATIO, APPROXIMATION_RANK
    if dimensions_count > 1:
        SEPARATION_RATIO = 0.65
    else:
        SEPARATION_RATIO = 0.6

    if tolerance < 10 ** -8:
        APPROXIMATION_RANK = int(np.log(tolerance) / np.log(SEPARATION_RATIO) - 20)
    elif tolerance < 10 ** -6:
        APPROXIMATION_RANK = int(np.log(tolerance) / np.log(SEPARATION_RATIO) - 15)
    else:
        APPROXIMATION_RANK = max(int(np.log(tolerance) / np.log(SEPARATION_RATIO) - 10), 5)

    log.info(f'Counting SEPARATION_RATIO={SEPARATION_RATIO}, APPROXIMATION_RANK={APPROXIMATION_RANK}')


def get_cauchy_values(n=500):
    x = [k / (n + 1) for k in range(1, n + 1)]
    y = [x_ + (10 ** -7) * random.random() for x_ in x]
    return x, y


def print_matrix(mat):
    if mat is not None:
        return str(mat.shape[0]) + 'x' + str(mat.shape[1])
    else:
        return 'None'


def ql(A):
    R, Q = rq(np.transpose(A))
    return np.transpose(Q), np.transpose(R)  # Q, L


def lq(A):
    Q, R = qr(np.transpose(A))
    return np.transpose(R), np.transpose(Q)  # L, Q


def matmul(A, B):
    A = np.array(A)
    B = np.array(B)

    if (B.shape[0] == 0 or len(B.shape) > 1 and B.shape[1] == 0 or A.shape[0] == 0 or len(A.shape) > 1 and A.shape[1] == 0):
        return np.array([])

    if len(B.shape) == 1:
        B = B.reshape((B.shape[0], 1))

    res = np.zeros((A.shape[0], B.shape[1]))

    if (A.shape[1] != B.shape[0]):
        log.critical(f'Multiplication size missmatch {A.shape[1]} {B.shape[0]}')
    assert A.shape[1] == B.shape[0]

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            res[i][j] = np.sum([A[i][r] * B[r][j] for r in range(A.shape[1])])
    return res


if __name__ == "__main__":
    a = np.array([[1, 2],
                  [3, 4]])
    b = np.array([[5, 6]])
    c = concat_row_wise(a, b)

    assert np.array_equal(c, np.array([[1, 2], [3, 4], [5, 6]]))
    assert np.array_equal(get_block(c, [0], [0]), np.array([[1]]))

    b = np.array([[5], [6]])
    c = concat_column_wise(a, b)
    assert np.array_equal(c, np.array([[1, 2, 5], [3, 4, 6]]))
    assert np.array_equal(get_block(c, [0, 1], [1, 2]), np.array([[2, 5], [4, 6]]))

    center, radius = get_metadata([1, 2, 3, 4])
    assert center == 2.5
    assert radius == 1.5
