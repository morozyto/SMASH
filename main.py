import numpy as np

import hss
import log

def k(x, y):
    dx = 1
    if (x != y):
        return 1 / (x - y)
    else:
        return dx

if __name__ == "__main__":

    start_value = 0
    n = 10
    step = 1
    assert step > 0
    assert n > 0

    log.info('Starting HSS')

    values = [i for i in range(start_value, start_value + n*step, step)]
    A = np.array([np.array([k(x, y) for y in values]) for x in values])

    log.debug('Index values is {}'.format(values))
    log.debug('Not copressed A is {}'.format(A))

    A_ = hss.HSS(values, A)

    vec = np.array([1] * n)
    log.info('Going to multiply matrices by vec {}'.format(vec))

    log.info('Not copressed result: ', np.matmul(A, vec))
    log.info('Compressed result: ', A_.multiply(vec))



