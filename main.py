import gauss
import hss
import log
import tools
import taylor_expansion
import partition

import numpy as np
import random
import time
from optparse import OptionParser


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


def parse_options():
    parser = OptionParser()

    parser.add_option("-f", "--farfield-tolerance", action="store", type="float", default=10 ** -7, dest="farfield_tolerance")
    parser.add_option("-s", "--svd-tolerance", action="store", type="float", default=10 ** -7, dest="svd_tolerance")
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
    return options

def test_adaptive_partition():
    log.info('test_adaptive_partition started')
    max_values_in_node = 2

    points = [(1, 1,), (3, 3,), (9, 9,), (5,5,), (4,4,), (2,2,),  (10, 10,), (7,7,), (8,8,)]
    partition_ = partition.Partition(points, points, points_dimension=2, max_values_in_node=max_values_in_node)
    trueX, data_ = partition_.build_levels(X=points)

    lines = []
    def get_lines(level, node_i, left, right, up, down):
        if level <= partition_.max_level - 1:
            data = data_[(level, node_i,)]
            if data[0] == 0: # вертикальная высота
                mid = data[1]
                lines.append(((left, mid), (right, mid),))
                get_lines(level + 1, 2 * node_i, left, right, up, mid)
                get_lines(level + 1, 2 * node_i + 1, left, right, mid, down)
            else: # горизонтальная высота
                mid = data[1]
                lines.append(((mid, down), (mid, up),))
                get_lines(level + 1, 2 * node_i, left, mid, up, down)
                get_lines(level + 1, 2 * node_i + 1, mid, right, up, down)

    left = min([point[0] for point in points]) - 5
    right = max([point[0] for point in points]) + 5
    up = min([point[1] for point in points]) - 5
    down = max([point[1] for point in points]) + 5

    lines.append(((left, down), (left, up),))
    lines.append(((left, up), (right, up),))
    lines.append(((right, up), (right, down),))
    lines.append(((right, down), (left, down),))
    get_lines(1, 0, left, right, up, down)
    log.info('\n\n\n')


def test_cauchy_matrix(n, max_values_in_node, vec, parallel_count):

    x_values, y_values = tools.get_cauchy_values(n=n)

    A = np.array([np.array([taylor_expansion.k(x, y) for y in y_values]) for x in x_values])

    if log.is_debug():
        log.debug(f'X index values is \n{x_values}')
        log.debug(f'Y index values is \n{y_values}')
        log.debug(f'Not compressed A is \n{A}')

    t = time.process_time()
    A_ = hss.HSS(x_values, y_values, A, max_values_in_node=max_values_in_node)
    log.info(f'HSS construction completed in {time.process_time() - t} seconds')

    if log.is_debug():
        log.debug(f'Printing result HSS\n{A_}')


    log.info('\n\n\nStart testing Cauchy matrix')
    log.info('Test multiplication')

    if log.is_debug():
        log.debug(f'Going to multiply matrices by vec \n {vec}')

    t = time.process_time()
    not_compr_result = tools.matmul(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.fast_multiply(vec, processes_count=parallel_count)
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
    not_compr_result = gauss.gauss(A, vec)
    norm_time = time.process_time() - t

    compr_result = A_.fast_solve(vec, processes_count=parallel_count)
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
    log.info('\n\n\n')

    return A, A_


def test_cauchy_like_matrix(A, A_, vec1, vec2, vec3, vec4, test_vec, parallel_count):

    log.info('Start testing Cauchy-like matrix')

    B_ = build_cauchy_like_matrix(A_, vec1, vec2, vec3, vec4)

    B = np.diag(vec1.reshape(len(vec1))) * A * np.diag(vec2.reshape(len(vec2))) \
        + np.diag(vec3.reshape(len(vec3))) * A * np.diag(vec4.reshape(len(vec4)))

    t = time.process_time()

    not_compr_result = tools.matmul(B, test_vec)
    norm_time = time.process_time() - t

    compr_result = B_.fast_multiply(test_vec, processes_count=parallel_count)
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
    not_compr_result = gauss.gauss(B, test_vec)
    norm_time = time.process_time() - t

    compr_result = B_.fast_solve(test_vec, processes_count=parallel_count)
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
    log.info('\n\n\n')


if __name__ == "__main__":

    log.info('Program started')

    options = parse_options()

    if options.debug_level:
        log.set_debug()

    log.init(save_file_log = not options.stdout_logs, log_name = options.log_file, logs_dir = options.log_dir)

    np.set_printoptions(precision=options.precision)

    dimension_count = 1
    tools.count_constants(dimension_count, options.farfield_tolerance, options.svd_tolerance)

    random_vec = np.array([[random.uniform(options.min_random_num, options.max_random_num)] for _ in range(options.edge_size)],
                          dtype='float')

    test_adaptive_partition()

    A, A_ = test_cauchy_matrix(n=options.edge_size, max_values_in_node=options.max_values_in_node, vec=random_vec, parallel_count=options.parallel_count)

    test_cauchy_like_matrix(A, A_, random_vec, random_vec, random_vec, random_vec, random_vec, parallel_count=options.parallel_count)

    log.info("That's all")





