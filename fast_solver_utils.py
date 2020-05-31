import numpy as np
import functools
import operator
from multiprocessing import SimpleQueue, Process

import gauss
import log
import tools

__all__ = ['solve']

def func(i, obj, b_i, A, q):

    if obj.U.shape[0] > obj.U.shape[1]:

        q_i, tmpU = tools.ql(obj.get_U())

        n_i = tmpU.shape[1]

        new_U = tools.get_block(tmpU, [i for i in range(tmpU.shape[1] - n_i, tmpU.shape[1])],
                                [j for j in range(n_i)])

        s = np.array(b_i).reshape((len(b_i), 1))
        c = np.transpose(q_i)
        t = c @ s

        beta = t[:len(t) - n_i]
        gamma = t[len(t) - n_i:]

        t = np.transpose(q_i) @ obj.get_D(A)

        tmpD, w_i = tools.lq(t)

        new_D = tools.get_block(tmpD, [i for i in range(tmpD.shape[0] - n_i, tmpD.shape[0])],
                                [j for j in range(tmpD.shape[0] - n_i, tmpD.shape[0])])

        tmpV = w_i @ obj.get_V()

        new_V = tools.get_block(tmpV, [i for i in range(tmpV.shape[0] - n_i, tmpD.shape[0])],
                                [j for j in range(tmpV.shape[1])])

        o = tools.get_block(tmpD, [i for i in range(tmpD.shape[0] - n_i)],
                            [j for j in range(tmpD.shape[0] - n_i)])
        z_i = gauss.gaussy(o, beta)  # np.linalg.solve(o, beta)

        z_i = list(z_i)

    else:
        if log.is_debug():
            log.debug('incompressible rows')
        tmpD = obj.get_D(A)
        tmpU = obj.get_U()
        tmpV = obj.get_V()

        new_D = obj.get_D(A)
        new_U = obj.get_U()
        new_V = obj.get_V()

        n_i = obj.get_U().shape[0]
        z_i = []
        q_i = np.identity(obj.get_D(A).shape[0])
        w_i = np.identity(obj.get_D(A).shape[1])

    res = [tmpD, tmpU, tmpV, new_D, new_U, new_V, (z_i, n_i), np.transpose(q_i), np.transpose(w_i)]

    q.put((i, res))


def batch_func(args):
    for arg in args:
        func(*arg)


def solve(hss, b, processes_count=1):
    if hss.Partition.max_level == 1:
        assert len(hss.Partition.level_to_nodes[1]) == 1
        tmp = hss.Partition.level_to_nodes[1][0].get_D(hss.A)
        b = np.array(b).reshape((len(b), 1))
        return gauss.gaussy(tmp, b)

    no_compressible_blocks = all([obj.U.shape[0] <= obj.U.shape[1] for obj in hss.Partition.level_to_nodes[hss.Partition.max_level]])

    if no_compressible_blocks:
        if log.is_debug():
            log.debug('No compressible blocks')
        tmp = hss.duplicate()
        tmp.remove_last_level()
        s = b
        return solve(tmp, s)
    else:
        if log.is_debug():
            log.debug('Compressible blocks')

        res = {}

        queue = SimpleQueue()
        start_index = 0

        args = {}
        tasks_count = len(hss.Partition.level_to_nodes[hss.Partition.max_level])
        for k in range(0, tasks_count):
            index = k % processes_count
            obj = hss.Partition.level_to_nodes[hss.Partition.max_level][k]
            b_i = b[start_index:start_index + obj.U.shape[0]]
            start_index += obj.U.shape[0]
            args[index] = args.get(index, []) + [(k, obj, b_i, hss.A, queue)]

        processes = []
        for key in args.keys():
            p = Process(target=batch_func, args=(args[key],))
            p.Daemon = True
            p.start()
            processes.append(p)

        for _ in range(tasks_count):
            pair = queue.get()
            res[pair[0]] = pair[1]

        tmpDs = [res[i][0] for i in range(0, tasks_count)]
        tmpUs = [res[i][1] for i in range(0, tasks_count)]
        tmpVs = [res[i][2] for i in range(0, tasks_count)]

        newDs = [res[i][3] for i in range(0, tasks_count)]
        newUs = [res[i][4] for i in range(0, tasks_count)]
        newVs = [res[i][5] for i in range(0, tasks_count)]

        z = [res[i][6] for i in range(0, tasks_count)]

        q_is = [res[i][7] for i in range(0, tasks_count)]
        w_is = [res[i][8] for i in range(0, tasks_count)]


        tmp_HSS = hss.duplicate()
        tmp_HSS.set_matrices(tmpUs, tmpVs, tmpDs)

        z__ = functools.reduce(operator.add, [list(z_[0]) + [0] * z_[1] for z_ in z])

        b = np.array(b).reshape((len(b), 1))
        b___ = tools.diag(q_is) @ b
        assert len(z__) == len(b)
        b___2 = tmp_HSS.multiply_perfect_binary_tree(z__)  #, processes_count=processes_count)
        b___2 = b___2.reshape((b___2.shape[0], 1))
        b_ = b___ - b___2
        new_b = []
        i = 0
        for obj in z:
            i += len(obj[0])
            j = 0
            while j < obj[1]:
                new_b.append([b_[i]])
                i += 1
                j += 1

        new_HSS = hss.duplicate()
        new_HSS.set_matrices(newUs, newVs, newDs)

        if log.is_debug():
            log.info(tmp_HSS)
        tmp_x = solve(new_HSS, new_b)

        i = 0
        tmp3 = []
        for z_ in z:
            tmp3 += list(z_[0]) + list(tmp_x[i:i + z_[1]])
            i += z_[1]

        return tools.diag(w_is) @ tmp3