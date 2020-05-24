import numpy as np
import random
import time
from pympler import asizeof
import copy

import hss
import log
import tools
import taylor_expansion

def build_cauchy_like_matrix(matrix, w1, w2, v1, v2):
    assert len(w1) == len(w2) == len(matrix.X)
    assert len(v1) == len(v2) == len(matrix.Y)

    def inner_multiply(matrix, w, v):
        w_index = 0
        v_index = 0

        result = copy.copy(matrix)

        for obj in result.Partition.level_to_nodes[result.Partition.max_level]:
            w_i = np.diag(w[w_index:w_index + len(obj.Indices)])
            v_i = np.diag(v[v_index:v_index + len(obj.Indices)])

            w_index += len(obj.Indices)
            v_index += len(obj.Indices)

            obj.U = w_i @ obj.U
            obj.V = np.transpose(v_i) @ obj.V

            obj.D = w_i @ obj.get_D(result.A) @ v_i

        assert w_index == len(w)
        assert v_index == len(v)
        return result

    matrix1 = inner_multiply(matrix, w1, v1)
    matrix2 = inner_multiply(matrix, w2, v2)
    return matrix1.sum(matrix2)

if __name__ == "__main__":
    np.set_printoptions(precision=3)

    tolerance = 10 ** -7
    dimension_count = 1
    tools.count_constants(dimension_count, tolerance)

    def get_cauchy_values():
        n = 50
        x = [k / (n + 1) for k in range(1, n + 1)]
        y = [x_ + (10 ** -7) * random.random() for x_ in x]
        return x, y

    x_values, y_values = get_cauchy_values() #tools.get_uniform_values(start_value=0, end_value=1, n=500)

    log.info('Starting HSS')

    A = np.array([np.array([taylor_expansion.k(x, y) for y in y_values]) for x in x_values])

    log.debug(f'X index values is {x_values}')
    log.debug(f'Y index values is {y_values}')
    #log.debug(f'Not compressed A is \n{A}')

    max_values_in_node = 25

    t = time.process_time()
    A_ = hss.HSS(x_values, y_values, A, max_values_in_node)
    log.info(f'HSS construction in seconds: {time.process_time() - t}')

    log.debug(f'Printing result HSS\n{A_}')

    random.seed(1)
    vec = np.array([random.random() for _ in range(A.shape[1])]) # np.array([1] * A.shape[1]) #np.array([[random.random()] for _ in range(A.shape[1])])
    #log.info(f'Going to multiply matrices by vec {vec}')

    t = time.process_time()
    not_compr_result = np.matmul(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.fast_multiply(vec) #A_.multiply_perfect_binary_tree(vec)
    hss_time = time.process_time() - t + norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.debug(f'Not copressed result:\n{not_compr_result}')
    log.debug(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    log.info(f'Usual multiplication performance in seconds: {norm_time}')
    log.info(f'HSS multiplication performance in seconds: {hss_time}')

    log.info(f'Source matrix has memory usage (bytes): {asizeof.asizeof(A)}')
    log.info(f'HSS has memory usage (bytes): {asizeof.asizeof(A_)}')


    #log.info(f'Going to solve system with b {vec}')
    t = time.process_time()

    not_compr_result = np.linalg.solve(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.solve(vec)
    hss_time = time.process_time() - t + norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.debug(f'Not copressed result:\n{not_compr_result}')
    log.debug(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    log.info(f'Usual solver performance in seconds: {norm_time}')
    log.info(f'HSS solver performance in seconds: {hss_time}')

    A_ = hss.HSS(x_values, y_values, A, max_values_in_node)

    log.info(A_)

    B_ = A_.sum(A_)
    B = A * 2

    log.info(B_)

    t = time.process_time()
    not_compr_result = np.matmul(B, vec)
    norm_time = time.process_time() - t

    compr_result = B_.multiply_perfect_binary_tree(vec)
    hss_time = time.process_time() - t + norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.debug(f'Not copressed result:\n{not_compr_result}')
    log.debug(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    log.info(f'Usual multiplication performance in seconds: {norm_time}')
    log.info(f'HSS multiplication performance in seconds: {hss_time}')

    log.info(f'Source matrix has memory usage (bytes): {asizeof.asizeof(A)}')
    log.info(f'HSS has memory usage (bytes): {asizeof.asizeof(A_)}')

    A_ = hss.HSS(x_values, y_values, A, max_values_in_node)
    tmp = build_cauchy_like_matrix(A_, vec, vec, vec, vec)

