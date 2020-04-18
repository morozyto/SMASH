import numpy as np

def concat_row_wise(a, b):
    np.concatenate((a, b))

def concat_column_wise(a, b):
    np.concatenate((a, b), axis=1)

def get_block(A, i, j):
    return A[np.array(i)[:, np.newaxis], np.array(j)]

def get_metadata(points):
    min_val, max_val = min(points), max(points)
    center = (max_val + min_val) / 2
    radius = max_val - center

    return center, radius


if __name__ == "__main__":
    pass

#     x = np.linspace(0, radius, num)