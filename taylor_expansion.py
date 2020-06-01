import tools

import math
import numpy as np

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

def n(radius, l):
    if l == 0:
        return 1
    return ((l / math.e) * ((2 * math.pi * tools.APPROXIMATION_RANK) ** (1 / (2 * tools.APPROXIMATION_RANK))) * (1 / radius)) ** l


def phi(radius, l, x):
    return n(radius, l) * ((x ** l) / math.factorial(l))


def c(k, l, a, b, radius_a, radius_b):
    if l > k:
        return 0
    return -math.factorial(k)*((b - a) ** (-k - 1)) * ((n(radius_a, l)) ** -1) * (n(radius_b, k - l) ** -1) * ((-1) ** (k - l))


def form_well_separated_expansion(X):
    center, radius = tools.get_metadata(X)
    U = np.array([[phi(radius, l, x - center) for l in range(tools.APPROXIMATION_RANK)] for x in X])
    return U


if __name__ == "__main__":
    import log

    tolerance = 10 ** -7
    dimension_count = 1
    tools.count_constants(tolerance, dimension_count)

    x_, _ = tools.get_cauchy_values(100)
    y_, _ = tools.get_cauchy_values(100)
    log.debug(f'X={x_}, Y={y_}')

    center_x, radius_x = tools.get_metadata(x_)
    center_y, radius_y = tools.get_metadata(y_)

    A = np.array([np.array([k(x, y) for y in y_]) for x in x_])
    log.debug(f'input A=\n{A}')
    log.debug(f'max_val is {np.max(np.abs(A))}')

    U = form_well_separated_expansion(x_)
    V = form_well_separated_expansion(y_)
    B = np.array([np.array([c(k, l, center_x, center_y, radius_x, radius_y) for k in range(tools.APPROXIMATION_RANK)])
                  for l in range(tools.APPROXIMATION_RANK)])

    res = U @ B @ np.transpose(V)

    error_mat = A - res

    log.debug(f'compressed A=\n{res}')
    log.info(f'is_farfield={radius_x + radius_y <= tools.SEPARATION_RATIO * abs(center_x - center_y)}')
    log.info(f'predicted max error is'
          f' {np.max(np.abs(A)) * (1 + tools.SEPARATION_RATIO) * (tools.SEPARATION_RATIO ** tools.APPROXIMATION_RANK) / (1 - tools.SEPARATION_RATIO)}')
    log.info(f'actual max_error={np.max(error_mat)}')