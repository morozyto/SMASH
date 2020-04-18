import numpy as np

def concat_row_wise(a, b):
    assert a.shape[1] == b.shape[1]
    return np.concatenate((a, b), axis=0)

def concat_column_wise(a, b):
    assert a.shape[0] == b.shape[0]
    return np.concatenate((a, b), axis=1)

def get_block(A, i, j):
    return A[np.array(i)[:, np.newaxis], np.array(j)]

def get_metadata(points):
    min_val, max_val = min(points), max(points)
    center = (max_val + min_val) / 2
    radius = max_val - center

    return center, radius


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

