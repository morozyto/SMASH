import tools
import log

import numpy as np
from scipy.linalg import qr #, svd
from numpy.linalg import norm, inv, slogdet, cond

__all__ = ['srrqr']


def srrqr(M, k, f=1., verbose=False):

    m, n = M.shape
    minmn = np.min([m, n])
    assert k <= minmn, "k must be less than min{m,n} k = %d,m = %d, n = %d" % (k, m, n)

    # QR with column pivoting
    Q, R, p = qr(M, mode='economic', pivoting=True) # economic?
    if log.is_debug():
        log.debug('SRRQR p {}'.format(p))
        log.debug(f'SRRQR R.shape = {R.shape}, Q.shape={Q.shape}')

    if k == n:
        return Q, R, p

    increase_found = True
    counter_perm = 0
    iterc = 0

    while (increase_found) and iterc <= 100:
        iterc += 1

        A = R[:k, :k]
        AinvB = np.linalg.solve(A, R[:k, k:])  # Form A^{-1}B

        C = R[k:minmn, k:]

        # Compute column norms of C
        if k < m:
            gamma = np.apply_along_axis(norm, 0, C[:, :n - k])

        # Find row norms of A^{-1}
        omega = np.apply_along_axis(norm, 0, inv(A).T)

        F = AinvB ** 2.
        if k < m:
            F += (np.outer(omega, gamma)) ** 2.
        ind = np.argwhere(F > f ** 2.)
        if ind.size == 0:  # finished
            increase_found = False
        else:  # we can increase |det(A)|
            i, j = ind[0, :]
            counter_perm += 1
            # permute columns i and j
            R[:, [i, j + k]] = R[:, [j + k, i]]
            p[[i, j + k]] = p[[j + k, i]]

            # retriangularize R
            q, R = qr(R, mode='economic')
            Q = np.dot(Q, q)
    # print p

    Rkk = R[:k, :k]
    inv_norm = norm(inv(Rkk), 2)
    res_norm = norm(R[k:minmn, k:], 2) if k < minmn else 0.

    #p = p[:k]
    return Q, R, p



def compr(M, indices):
    if log.is_debug():
        log.debug('M=\n{}'.format(M))
    k = np.min([M.shape[0], M.shape[1]]) # M.shape[0]
    input_matrix = np.transpose(M)

    if log.is_debug():
        log.debug(f'input matrix shape {input_matrix.shape}')

        log.debug(f'srrqr input k={k}, input=\n{input_matrix}')
    Q, R, P = srrqr(input_matrix, k=k, f=2)

    if log.is_debug():
        log.debug(f'srrqr result Q=\n{Q}, R=\n{R}, P=\n{P}')

    r = Q.shape[0]

    rows_count = R.shape[0]
    column_count = R.shape[1]
    R11 = tools.get_block(R, [i for i in range(rows_count)], [i for i in range(rows_count)]) #submatrix_r_r
    R12 = tools.get_block(R, [i for i in range(rows_count)], [i for i in range(rows_count, column_count)]) if column_count > rows_count else None
    if R12 is not None:
        R11 = np.linalg.inv(R11)
        G = np.transpose(np.matmul(R11, R12))

        if log.is_debug():
            log.debug('G=\n{}'.format(G))
            log.debug('indices={}'.format(indices))
            log.debug('P=\n{}'.format(P))

        assert len(indices) == len(P)
        P = np.array([np.array([int(i == item) for i in range(len(indices))]) for item in P])

        if log.is_debug():
            log.debug('Q=\n{}'.format(Q))
            log.debug('R=\n{}'.format(R))
            log.debug('P=\n{}'.format(P))
            log.debug('r={}'.format(r))
            log.debug('input indices  {}'.format(indices))

        new_indices = [i[0] for i in np.matmul(np.transpose(P), np.array([[i] for i in indices]))[:r].tolist()]

        if log.is_debug():
            log.debug('returned indices from compressed {}'.format(new_indices))
        return P, G, new_indices  # i is a list
    else:
        if log.is_debug():
            log.debug('Compr do almost nothing. Return')
        P = np.array([np.array([int(i == item) for i in range(len(indices))]) for item in range(len(indices))])
        G = np.array([np.array([])])
        return P, G, indices  # i is a list


if __name__ == '__main__':
    from scipy.linalg import hadamard

    A = np.random.randn(100, 30)
    k = 30
    Q, R, p = srrqr(A, k, verbose=True)
    print(p)
    print(np.allclose(A[:, p], np.dot(Q, R[:, :k])))

    A = np.array([
         [ 1.000e+00, 1.000e+00, 1.000e+00, 1.000e+00, 1.000e+00],
         [-4.249e-01, -2.124e-01, 0.000e+00, 2.124e-01, 4.249e-01],
         [ 3.611e-01, 9.027e-02, 0.000e+00, 9.027e-02, 3.611e-01],
         [-3.452e-01, -4.315e-02, 0.000e+00, 4.315e-02, 3.452e-01],
         [ 3.476e-01, 2.173e-02, 0.000e+00, 2.173e-02, 3.476e-01],
         [-3.606e-01, -1.127e-02, 0.000e+00, 1.127e-02, 3.606e-01],
         [ 3.813e-01, 5.957e-03, 0.000e+00, 5.957e-03, 3.813e-01],
         [-4.085e-01, -3.191e-03, 0.000e+00, 3.191e-03, 4.085e-01],
         [ 4.420e-01, 1.727e-03, 0.000e+00, 1.727e-03, 4.420e-01],
         [-4.819e-01, -9.411e-04, 0.000e+00, 9.411e-04, 4.819e-01],
         [ 5.285e-01, 5.161e-04, 0.000e+00, 5.161e-04, 5.285e-01],
         [-5.824e-01, -2.844e-04, 0.000e+00, 2.844e-04, 5.824e-01],
         [ 6.444e-01, 1.573e-04, 0.000e+00, 1.573e-04, 6.444e-01],
         [-7.155e-01, -8.734e-05, 0.000e+00, 8.734e-05, 7.155e-01],
         [ 7.967e-01, 4.862e-05, 0.000e+00, 4.862e-05, 7.967e-01],
         [-8.893e-01, -2.714e-05, 0.000e+00, 2.714e-05, 8.893e-01],
         [-2.098e-01, -2.534e-01, -3.216e-01, -4.458e-01, -7.679e-01],
         [-4.290e-01, -4.439e-01, -4.249e-01, -2.759e-01, 6.019e-01],
         [ 5.652e-01, 3.096e-01, -1.205e-01, -7.241e-01, 2.142e-01],
         [-5.576e-01, 2.330e-01, 6.674e-01, -4.327e-01, 4.716e-02],
         [-3.762e-01, 7.672e-01, -5.062e-01, 1.167e-01, -6.174e-03]
         ])
    k = 5
    Q, R, p = srrqr(A, k, f=2, verbose=True)

    P = np.array([np.array([int(i == item) for i in range(len(p))]) for item in p])
    #print(Q)
    #print(R)
    print(p)
    print(P)
    print(A @ P)
    print(np.dot(Q, R))
    print(np.allclose(A @ P, np.dot(Q, R)))
