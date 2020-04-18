import numpy as np

import hss

def k(x, y):
    dx = 1
    if (x != y):
        return 1 / (x - y)
    else:
        return dx

if __name__ == "__main__":
    SEPARATION_RATIO = 0.7
    r = 10

    assert SEPARATION_RATIO > 0
    assert r > 0

    start_value = 0
    n = 10
    step = 1
    assert step > 0
    assert n > 0

    values = [i for i in range(start_value, start_value + n*step, step)]
    A = np.array([np.array([k(x, y) for y in values]) for x in values])

    A_ = hss.HSS(values, A)

    vec = np.array([1] * n)

    print('Not copressed result: ', np.matmul(A, vec))
    print('Compressed result: ', A_.multiply(vec))



