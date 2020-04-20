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

    def get_values():
        start_value = 0
        n = 10
        step = 1
        assert step > 0
        assert n > 0
        return [i for i in range(start_value, start_value + n * step, step)]

    values = get_values()

    log.info('Starting HSS')

    A = np.array([np.array([k(x, y) for y in values]) for x in values])

    log.debug('Index values is {}'.format(values))
    log.debug('Not compressed A is {}'.format(A))

    A_ = hss.HSS(values, A)

    vec = np.array([1] * A.shape[1])
    log.info('Going to multiply matrices by vec {}'.format(vec))

    log.info('Not copressed result: {}'.format(np.matmul(A, vec)))
    log.info('Compressed result: {}'.format(A_.multiply(vec)))



