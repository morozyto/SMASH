import numpy as np
import random
import time
from pympler import asizeof

import gauss
import hss
import log
import tools
import taylor_expansion


def inner_multiply(matrix, w, v):
    w_index = 0
    v_index = 0

    result = matrix.duplicate(deepcopy_leaves=True)

    for obj in result.Partition.level_to_nodes[result.Partition.max_level]:
        t = w[w_index:w_index + len(obj.Indices)].reshape(len(obj.Indices))
        w_i = np.diag(t)
        s = v[v_index:v_index + len(obj.Indices)].reshape(len(obj.Indices))
        v_i = np.diag(s)

        w_index += len(obj.Indices)
        v_index += len(obj.Indices)

        obj.U = w_i @ obj.U
        obj.V = np.transpose(v_i) @ obj.V

        obj.D = w_i @ obj.get_D(result.A) @ v_i

    assert w_index == len(w)
    assert v_index == len(v)
    return result

def build_cauchy_like_matrix(matrix, w1, w2, v1, v2):
    assert len(w1) == len(w2) == len(matrix.X)
    assert len(v1) == len(v2) == len(matrix.Y)

    matrix1 = inner_multiply(matrix, w1, v1)
    matrix2 = inner_multiply(matrix, w2, v2)
    return matrix1.sum(matrix2)

if __name__ == "__main__":
    np.set_printoptions(precision=3)

    # 1 250  30 solver

    random.seed(3)

    tolerance = 10 ** -7
    dimension_count = 1
    tools.count_constants(dimension_count, tolerance)

    x_values, y_values = tools.get_cauchy_values(n=250)

    log.info('Starting HSS')

    A = np.array([np.array([taylor_expansion.k(x, y) for y in y_values]) for x in x_values])

    log.debug(f'X index values is {x_values}')
    log.debug(f'Y index values is {y_values}')
    #log.debug(f'Not compressed A is \n{A}')

    max_values_in_node = 40

    t = time.process_time()
    A_ = hss.HSS(x_values, y_values, A, max_values_in_node=max_values_in_node)
    log.info(f'HSS construction in seconds: {time.process_time() - t}')

    log.debug(f'Printing result HSS\n{A_}')

    min_val = 0
    max_val = 100
    random_vec = np.array([[random.uniform(min_val, max_val)] for _ in range(A.shape[1])], dtype='float')
    identity_vec = np.array([1] * A.shape[1])
    increasing_vec = np.array([[i] for i in range(A.shape[1])], dtype='float')

    vec = random_vec
    #log.info(f'Going to multiply matrices by vec {vec}')

    t = time.process_time()
    not_compr_result = tools.matmul(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.fast_multiply(vec) #A_.multiply_perfect_binary_tree(vec)
    hss_time = time.process_time() - t - norm_time

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


    t = time.process_time()

    not_compr_result = gauss.gaussy(A, vec) #np.linalg.solve(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.fast_solve(vec)
    hss_time = time.process_time() - t - norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.debug(f'Not copressed result:\n{not_compr_result}')
    log.debug(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    log.info(f'Usual solver performance in seconds: {norm_time}')
    log.info(f'HSS solver performance in seconds: {hss_time}')

    log.info('\n\n\n')
    log.info('Start testing Cauchy-like matrix')

    '''
    vec1 = np.array([random.random() for _ in range(A.shape[1])])
    vec2 = np.array([random.random() for _ in range(A.shape[1])])
    vec3 = np.array([random.random() for _ in range(A.shape[1])])
    vec4 = np.array([random.random() for _ in range(A.shape[1])])
    '''

    B_ = build_cauchy_like_matrix(A_, identity_vec, vec, identity_vec, identity_vec)
    B = np.diag(identity_vec) * A * np.diag(vec) + np.diag(identity_vec) * A * np.diag(identity_vec)

    t = time.process_time()
    not_compr_result = tools.matmul(B, vec)
    norm_time = time.process_time() - t

    compr_result = B_.fast_multiply(vec)
    hss_time = time.process_time() - t - norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.debug(f'Not copressed result:\n{not_compr_result}')
    log.debug(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    log.info(f'Usual Cauchy-like multiplication performance in seconds: {norm_time}')
    log.info(f'HSS Cauchy-like multiplication performance in seconds: {hss_time}')


    t = time.process_time()
    not_compr_result = gauss.gaussy(B, vec)
    norm_time = time.process_time() - t

    compr_result = B_.fast_solve(vec)
    hss_time = time.process_time() - t - norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    log.debug(f'Not copressed result:\n{not_compr_result}')
    log.debug(f'Compressed result:\n{compr_result}')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')
    log.info(f'Usual solver performance in seconds: {norm_time}')
    log.info(f'HSS solver performance in seconds: {hss_time}')




