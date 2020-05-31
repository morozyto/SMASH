import threading
from multiprocessing import SimpleQueue, Process  # , shared_memory
from queue import Queue

import tools
import numpy as np

__all__ = ['fast_multiply']

def get_b(i, level_to_nodes, max_level, b):
    indices = level_to_nodes[max_level][i].Indices
    start, end = indices[0], indices[-1]
    return b[start:end + 1]

def get_G(k, i, level_to_nodes, max_level, b):
    if k == max_level:
        return tools.matmul(np.transpose(level_to_nodes[k][i].V), get_b(i, level_to_nodes, max_level, b))
    else:
        return tools.matmul(np.transpose(level_to_nodes[k][i].Ws[0]), get_G(k + 1, 2*i, level_to_nodes, max_level, b)) + \
        tools.matmul(np.transpose(level_to_nodes[k][i].Ws[1]), get_G(k + 1, 2*i + 1, level_to_nodes, max_level, b))

def get_F(k, i, level_to_nodes, max_level, A, b):
    if k == 1 and i == 0:
        return [0] * level_to_nodes[k][0].R.shape[1]
    if i % 2 == 0:
        return tools.matmul(level_to_nodes[k][i].get_B_subblock(A), get_G(k, i + 1, level_to_nodes, max_level, b)) + \
               tools.matmul(level_to_nodes[k - 1][i // 2].Rs[0], get_F(k - 1, i // 2, level_to_nodes, max_level, A, b))
    else:
        return tools.matmul(level_to_nodes[k][i].get_B_subblock(A), get_G(k, i - 1, level_to_nodes, max_level, b)) + \
               tools.matmul(level_to_nodes[k - 1][i // 2].Rs[1], get_F(k - 1, i // 2, level_to_nodes, max_level, A, b))

def func(i, b, level_to_nodes, max_level, A, q):
    s1 = level_to_nodes[max_level][i].get_D(A)
    s2 = np.array(get_b(i, level_to_nodes, max_level, b))
    s3 = level_to_nodes[max_level][i].U
    s4 = get_F(max_level, i, level_to_nodes, max_level, A, b)
    res = tools.matmul(s1, s2) + tools.matmul(s3, s4)
    q.put((i, res))
    return res

def batch_func(args):
    for arg in args:
        func(*arg)


def fast_multiply(partition, A, b, processes_count=4):

    res = {}

    queue = SimpleQueue()
    processes_count = 1

    args = {}
    tasks_count = len(partition.level_to_nodes[partition.max_level])
    for k in range(0, tasks_count):
        index = k % processes_count
        args[index] = args.get(index, []) + [(k, b, partition.level_to_nodes, partition.max_level, A, queue)]
        #queue.put(k)

    processes = []
    for key in args.keys():
        p = Process(target=batch_func, args=(args[key],))
        p.Daemon = True
        p.start()
        processes.append(p)

    for _ in range(tasks_count):
        pair = queue.get()
        res[pair[0]] = pair[1]

    result = []
    for k in range(0, tasks_count):
        result += list(res[k])

    return np.array(result)

