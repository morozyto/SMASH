import numpy as np
import random

import hss
import log
import tools
import taylor_expansion

if __name__ == "__main__":
    np.set_printoptions(precision=3)

    tolerance = 10 ** -7
    dimension_count = 1
    tools.count_constants(dimension_count, tolerance)

    def get_cauchy_values():
        n = 51
        x = [k / (n + 1) for k in range(1, n + 1)]
        y = [x_ + (10 ** -7) * random.random() for x_ in x]
        return x, y

    x_values, y_values = tools.get_uniform_values(start_value=0, end_value=1, n=1000) # 50, 51

    log.info('Starting HSS')

    A = np.array([np.array([taylor_expansion.k(x, y) for y in y_values]) for x in x_values])

    log.debug(f'X index values is {x_values}')
    log.debug(f'Y index values is {y_values}')
    log.debug(f'Not compressed A is \n{A}')

    max_values_in_node = 30

    A_ = hss.HSS(x_values, y_values, A, max_values_in_node)
    log.debug(f'Printing result HSS\n{A_}')

    vec = np.array([1] * A.shape[1])
    log.info(f'Going to multiply matrices by vec {vec}')

    not_compr_result = np.matmul(A, vec)
    compr_result = A_.multiply_perfect_binary_tree(vec)
    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.info(f'Not copressed result:\n{not_compr_result}')
    log.info(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    '''
    log.info(f'Going to solve system with b {vec}')

    not_compr_result = np.linalg.solve(A, vec)
    compr_result = A_.solve(vec)
    log.debug(f'Not compressed A is \n{A}')
    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.info(f'Not copressed result:\n{not_compr_result}')
    log.info(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    #assert error < 3
    '''