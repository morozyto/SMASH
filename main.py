import numpy as np

import hss
import log


def k(x, y):
    """
    Kernel function.
    :param x: first_arg
    :param y: second_arg
    :rtype: int
    """
    dx = 1
    if (x != y):
        return 1 / (x - y)
    else:
        return dx

if __name__ == "__main__":

    def get_uniform_values():
        start_value = 0
        n = 10
        step = 1
        assert step > 0
        assert n > 0
        obj = [i for i in range(start_value, start_value + n * step, step)]
        return obj, obj


    x_values, y_values = get_uniform_values()

    log.info('Starting HSS')

    A = np.array([np.array([k(x, y) for y in y_values]) for x in x_values])

    log.debug('X index values is {}'.format(x_values))
    log.debug('Y index values is {}'.format(y_values))
    log.debug('Not compressed A is {}'.format(A))

    A_ = hss.HSS(x_values, A)

    vec = np.array([1] * A.shape[1])
    log.info('Going to multiply matrices by vec {}'.format(vec))

    not_compr_result = np.matmul(A, vec)
    compr_result = A_.multiply(vec)
    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.info('Not copressed result: {}'.format(not_compr_result))
    log.info('Compressed result: {}'.format(compr_result))
    log.info('Error: {}'.format(error))
    assert error < 3



