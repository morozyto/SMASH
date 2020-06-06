import numpy as np


def inner_multiply(matrix, w, v):
    w_index = 0
    v_index = 0

    result = matrix.duplicate(deepcopy_leaves=True)

    for obj in result.Partition.level_to_nodes[result.Partition.max_level]:
        t = w[w_index:w_index + len(obj.Indices)].reshape(len(obj.Indices))
        w_i = np.diag(t)
        s = v[v_index:v_index + len(obj.Indices)].reshape(len(obj.Indices))
        v_i = np.diag(s)

        w_index += len(obj.Indices)
        v_index += len(obj.Indices)

        obj.U = w_i @ obj.U
        obj.V = np.transpose(v_i) @ obj.V

        obj.D = w_i @ obj.get_D(result.A) @ v_i

    assert w_index == len(w)
    assert v_index == len(v)
    return result


def build_cauchy_like_matrix(matrix, w1, w2, v1, v2):
    assert len(w1) == len(w2) == len(matrix.X)
    assert len(v1) == len(v2) == len(matrix.Y)

    matrix1 = inner_multiply(matrix, w1, v1)
    matrix2 = inner_multiply(matrix, w2, v2)
    res = matrix1.sum(matrix2)
    return res