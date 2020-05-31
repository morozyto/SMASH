import numpy as np
import random
import time
from optparse import OptionParser
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
    res = matrix1.sum(matrix2)
    return res


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-t", "--tolerance", action="store", type="float", default=10 ** -7, dest="tolerance")
    parser.add_option("-p", "--np-precision", action="store", type="int", default=3, dest="precision")
    parser.add_option("-n", action="store", type="int", default=250, dest="edge_size")
    parser.add_option("-m", "--max-node-values", type="int", default=30, dest="max_values_in_node")
    parser.add_option("--max-random-num", type="int", default=100, dest="max_random_num")
    parser.add_option("--min-random-num", type="int", default=0, dest="min_random_num")
    parser.add_option("-c", "--parallel-count", type="int", default=4, dest="parallel_count")
    parser.add_option("--write-logs-stdout", action='store_true', dest="stdout_logs")
    parser.add_option("-l", "--log-file", type="string", default='log', dest="log_file")
    parser.add_option("-d", "--log-dir", type="string", default='logs', dest="log_dir")
    parser.add_option("-v", action='store_true', dest="debug_level")

    (options, _) = parser.parse_args()

    if options.debug_level:
        log.set_debug()

    log.init(save_file_log = not options.stdout_logs, log_name = options.log_file, logs_dir = options.log_dir)

    np.set_printoptions(precision=options.precision)

    dimension_count = 1
    tools.count_constants(dimension_count, options.tolerance)

    x_values, y_values = tools.get_cauchy_values(n=options.edge_size)

    log.info('Starting HSS ')

    A = np.array([np.array([taylor_expansion.k(x, y) for y in y_values]) for x in x_values])

    if log.is_debug():
        log.debug(f'X index values is \n{x_values}')
        log.debug(f'Y index values is \n{y_values}')
        log.debug(f'Not compressed A is \n{A}')


    t = time.process_time()
    A_ = hss.HSS(x_values, y_values, A, max_values_in_node=options.max_values_in_node)
    log.info(f'HSS construction completed in {time.process_time() - t} seconds')

    if log.is_debug():
        log.debug(f'Printing result HSS\n{A_}')

    random_vec = np.array([[random.uniform(options.min_random_num, options.max_random_num)] for _ in range(A.shape[1])], dtype='float')
    identity_vec = np.array([1] * A.shape[1])

    vec = random_vec

    log.info('\n\n\nStart testing Cauchy matrix')
    log.info('Test multiplication')

    if log.is_debug():
        log.debug(f'Going to multiply matrices by vec \n {vec}')

    t = time.process_time()
    not_compr_result = tools.matmul(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.fast_multiply(vec, processes_count=options.parallel_count)
    hss_time = time.process_time() - t - norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    if log.is_debug():
        log.debug(f'Not copressed result:\n{not_compr_result}')
        log.debug(f'Compressed result:\n{compr_result}')

    log.info(f'Usual multiplication performance in {norm_time} seconds')
    log.info(f'HSS multiplication performance in {hss_time} seconds')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')


    log.info('Test solver')
    t = time.process_time()
    not_compr_result = gauss.gaussy(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.fast_solve(vec, processes_count=options.parallel_count)
    hss_time = time.process_time() - t - norm_time

    error_vec = not_compr_result.reshape(len(not_compr_result)) - compr_result.reshape(len(compr_result))
    error = np.linalg.norm(error_vec)

    if log.is_debug():
        log.debug(f'Not copressed result:\n{not_compr_result}')
        log.debug(f'Compressed result:\n{compr_result}')

    log.info(f'Usual solver performance in {norm_time} seconds')
    log.info(f'HSS solver performance in {hss_time} seconds')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')

    log.info('\n\n\nStart testing Cauchy-like matrix')


    B_ = build_cauchy_like_matrix(A_, identity_vec, vec, vec, identity_vec)

    B = np.diag(identity_vec.reshape(len(identity_vec))) * A * np.diag(vec.reshape(len(vec))) \
        + np.diag(vec.reshape(len(vec))) * A * np.diag(identity_vec.reshape(len(identity_vec)))


    t = time.process_time()

    not_compr_result = tools.matmul(B, vec)
    norm_time = time.process_time() - t

    compr_result = B_.fast_multiply(vec, processes_count=options.parallel_count)
    hss_time = time.process_time() - t - norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    if log.is_debug():
        log.debug(f'Not copressed result:\n{not_compr_result}')
        log.debug(f'Compressed result:\n{compr_result}')

    log.info(f'Usual multiplication performance in {norm_time} seconds')
    log.info(f'HSS multiplication performance in {hss_time} seconds')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')

    log.info('Test solver')
    t = time.process_time()
    not_compr_result = gauss.gaussy(B, vec)
    norm_time = time.process_time() - t

    compr_result = B_.fast_solve(vec, processes_count=options.parallel_count)
    hss_time = time.process_time() - t - norm_time

    error_vec = not_compr_result - compr_result
    error = np.linalg.norm(error_vec)

    if log.is_debug():
        log.debug(f'Not copressed result:\n{not_compr_result}')
        log.debug(f'Compressed result:\n{compr_result}')

    log.info(f'Usual solver performance in {norm_time} seconds')
    log.info(f'HSS solver performance in {hss_time} seconds')
    log.info(f'Error vec norm: {error}')
    log.info(f'Relative error: {error / np.linalg.norm(not_compr_result)}')

    log.info("That's all")





