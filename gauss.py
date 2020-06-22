import tools
import log
import copy

import numpy as np

def gauss(B, c, use_np = True):
    if use_np:
        return np.linalg.solve(B, c)
    A = copy.copy(B)
    b = copy.copy(c)

    log.debug(f'Gauss started {tools.print_matrix(A)}')
    assert A.shape[0] == A.shape[1] == len(b)
    n = len(b)
    assert n > 0

    current_line = np.array([i for i in range(n)]).reshape(n)
    line_max = np.array([max([abs(A[i][j]) for j in range(n)]) for i in range(n)]).reshape(n)

    for i in range(n - 1):

        index = i + np.argmax([abs(A[current_line[j]][i]) / line_max[current_line[j]] for j in range(i, n)])
        current_line[i], current_line[index] = current_line[index], current_line[i]

        for j in range(i + 1, n):
            factor = A[current_line[j]][i] / A[current_line[i]][i]
            b[current_line[j]] = b[current_line[j]] - factor * b[current_line[i]]
            for k in range(i, n):
                A[current_line[j]][k] = A[current_line[j]][k] - factor * A[current_line[i]][k]

    x = np.zeros(n)
    x[n - 1] = b[current_line[n - 1]] / A[current_line[n - 1]][n - 1]

    for i in range(n - 2, -1, -1):
        summa = sum([A[current_line[i]][j] * x[j] for j in range(i + 1, n)])
        x[i] = (b[current_line[i]] - summa) / A[current_line[i]][i]

    log.debug('Gauss ended')
    return x
