import numpy as np
import tools

def gaussy(A, b):
   # print(f'Gauss started {tools.print_matrix(A)}')
    assert A.shape[0] == A.shape[1] == len(b)
    n = len(b)
    l = [0 for _ in range(n)]
    s = [0.0 for _ in range(n)]
    for i in range(n):
        l[i] = i
        smax = 0.0
        for j in range(n):
            if abs(A[i][j]) > smax:
                smax = abs(A[i][j])
        s[i] = smax

   # print('Gauss step1')


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

    #print('Gauss step2')


    x = [0.0 for _ in range(n)]
    x[n - 1] = b[l[n - 1]] / A[l[n - 1]][n - 1]
    for j in range(n - 2, -1, -1):
        summ = 0.0
        for k in range(j + 1, n):
            summ = summ + A[l[j]][k] * x[k]
        x[j] = (b[l[j]] - summ) / A[l[j]][j]

    #print('Gauss ended')
    return np.array(x)


'''
def gauss(A):
    m = len(A)
    assert all([len(row) == m + 1 for row in A[1:]]), "Matrix rows have non-uniform length"
    n = m + 1

    for k in range(m):
        pivots = [abs(A[i][k]) for i in range(k, m)]
        i_max = pivots.index(max(pivots)) + k

        # Check for singular matrix
        assert A[i_max][k] != 0, "Matrix is singular!"

        # Swap rows
        A[k], A[i_max] = A[i_max], A[k]

        for i in range(k + 1, m):
            f = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] -= A[k][j] * f

            # Fill lower triangular matrix with zeros:
            A[i][k] = 0

    # Solve equation Ax=b for an upper triangular matrix A
    x = []
    for i in range(m - 1, -1, -1):
        x.insert(0, A[i][m] / A[i][i])
        for k in range(i - 1, -1, -1):
            A[k][m] -= A[k][i] * x[0]
    return x
'''