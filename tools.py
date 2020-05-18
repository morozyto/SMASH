import numpy as np
import log

from scipy.linalg import block_diag

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


def get_uniform_values(start_value=0, end_value=1, n=10):
    step = (end_value - start_value) / n
    assert step > 0
    assert n > 0
    obj = [i for i in np.arange(start_value, start_value + n * step, step)]
    return obj, obj


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

