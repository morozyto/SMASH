import tools
import log

import scipy.linalg
import numpy as np

__all__ = ['compr']

'''
Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for 
 computing a strong rank-revealing QR factorization." SIAM Journal on Scientific Computing 17.4 (1996): 848-869.
'''


def srrqr(M, k, f=2., max_iter_count=np.inf):

    m, n = M.shape
    min_val = np.min([m, n])
    assert k <= min_val, f"SRRQR: k = {k} <= min(n = {n}, m = {m})"

    Q, R, p = scipy.linalg.qr(M, pivoting=True, mode='economic') # economic?

    if k == n:
        return Q, R, p

    count = 0

    while count <= max_iter_count:
        A = R[:k, :k]
        B = R[:k, k:]
        C = R[k:min_val, k:]

        omega = np.apply_along_axis(np.linalg.norm, 1, np.transpose(np.linalg.inv(A)))

        x = np.linalg.solve(A, B) ** 2.
        if k < m:
            gamma = np.apply_along_axis(np.linalg.norm, 0, C[:, :n - k])
            x += (np.outer(omega, gamma)) ** 2.

        indices = np.argwhere(x > f ** 2.)

        if len(indices) == 0:
            break

        i, j = indices[0][:]

        p[[i, j + k]] = p[[j + k, i]]
        R[:, [i, j + k]] = R[:, [j + k, i]]

        q, R = scipy.linalg.qr(R, mode='economic')
        Q = Q @ q

        count += 1

    return Q, R, p


def compr(M, indices):
    if log.is_debug():
        log.debug('M=\n{}'.format(M))
    k = np.min([M.shape[0], M.shape[1]]) # M.shape[0]
    input_matrix = np.transpose(M)

    Q, R, P = srrqr(input_matrix, k=k, f=2)

    if log.is_debug():
        log.debug(f'SRRQR result Q=\n{Q}, R=\n{R}, P=\n{P}')

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
    m, n = 80, 40
    A = np.random.randn(m, n)
    k = min([m, n])
    Q, R, p = srrqr(A, k)
    assert np.allclose(A[:, p], np.dot(Q, R[:, :k]))