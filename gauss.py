import numpy as np
import tools
import log

def gaussy(A, b):
    log.debug(f'Gauss started {tools.print_matrix(A)}')
    assert A.shape[0] == A.shape[1] == len(b)

    n = len(b)
    l = [0 for _ in range(n)] # [0 for _ in range(n)]
    s = [0 for _ in range(n)] # np.zeros(n)
    for i in range(n):
        l[i] = i
        smax = 0.0
        for j in range(n):
            if abs(A[i][j]) > smax:
                smax = abs(A[i][j])
        s[i] = smax


    for i in range(n - 1):
        rmax = 0.0
        for j in range(i, n):
            ratio = abs(A[l[j]][i]) / s[l[j]]
            if ratio > rmax:
                rmax = ratio
                rindex = j
        temp = l[i]
        l[i] = l[rindex]
        l[rindex] = temp
        for j in range(i + 1, n):
            multiplier = A[l[j]][i] / A[l[i]][i]
            for k in range(i, n):
                A[l[j]][k] = A[l[j]][k] - multiplier * A[l[i]][k]
            b[l[j]] = b[l[j]] - multiplier * b[l[i]]

    x = [0.0 for _ in range(n)]
    x[n - 1] = b[l[n - 1]] / A[l[n - 1]][n - 1]
    for j in range(n - 2, -1, -1):
        summ = 0.0
        for k in range(j + 1, n):
            summ = summ + A[l[j]][k] * x[k]
        x[j] = (b[l[j]] - summ) / A[l[j]][j]

    log.debug('Gauss ended')
    return np.array(x)
