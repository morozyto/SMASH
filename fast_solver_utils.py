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
        if log.is_debug():
            log.info(f'U={tools.print_matrix(obj.get_U())}')
            log.info(f'obj={obj}')
        q_i, tmpU = tools.ql(obj.get_U())
        if log.is_debug():
            log.info(f'q_i={tools.print_matrix(q_i)}')
            log.info(f'tmpU={tools.print_matrix(tmpU)}')

        n_i = tmpU.shape[1]

        if log.is_debug():
            log.info(f'n_i={n_i}, {tmpU.shape[0] - n_i}')

        new_U = tools.get_block(tmpU, [i for i in range(tmpU.shape[1] - n_i, tmpU.shape[1])],
                                [j for j in range(n_i)])

        if log.is_debug():
            log.info(f'new_U={tools.print_matrix(new_U)}')

        if log.is_debug():
            log.info(f'b_i len={len(b_i)}')

        s = np.array(b_i).reshape((len(b_i), 1))
        c = np.transpose(q_i)
        t = c @ s
        if log.is_debug():
            log.info(f't len={len(t)}')

        beta = t[:len(t) - n_i]
        gamma = t[len(t) - n_i:]

        t = np.transpose(q_i) @ obj.get_D(A)
        if log.is_debug():
            log.info(f't ={tools.print_matrix(t)}')

        tmpD, w_i = tools.lq(t)
        if log.is_debug():
            log.info(f'tmpD ={tools.print_matrix(tmpD)}')
            log.info(f'w_i ={tools.print_matrix(w_i)}')

        new_D = tools.get_block(tmpD, [i for i in range(tmpD.shape[0] - n_i, tmpD.shape[0])],
                                [j for j in range(tmpD.shape[0] - n_i, tmpD.shape[0])])

        tmpV = w_i @ obj.get_V()

        if log.is_debug():
            log.info(f'tmpV ={tools.print_matrix(tmpV)}')

        new_V = tools.get_block(tmpV, [i for i in range(tmpV.shape[0] - n_i, tmpD.shape[0])],
                                [j for j in range(tmpV.shape[1])])

        o = tools.get_block(tmpD, [i for i in range(tmpD.shape[0] - n_i)],
                            [j for j in range(tmpD.shape[0] - n_i)])
        z_i = gauss.gaussy(o, beta)  # np.linalg.solve(o, beta)

        if log.is_debug():
            log.info(f'n_i ={n_i}')
            log.info(f'z_i ={len(z_i)}')
        z_i = list(z_i)
        if log.is_debug():
            log.info(f'z_i ={len(z_i)}')
    else:
        if log.is_debug():
            log.info('incompressible rows')
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

def solve(hss, b):
    #log.info('fast solver started')
    if hss.Partition.max_level == 1:
        assert len(hss.Partition.level_to_nodes[1]) == 1
        tmp = hss.Partition.level_to_nodes[1][0].get_D(hss.A)
        b = np.array(b).reshape((len(b), 1))
        return gauss.gaussy(tmp, b)

    no_compressible_blocks = all([obj.U.shape[0] <= obj.U.shape[1] for obj in hss.Partition.level_to_nodes[hss.Partition.max_level]])

    if no_compressible_blocks:
        #log.info('No compressible blocks')
        tmp = hss.duplicate()
        tmp.remove_last_level()
        s = b
        return solve(tmp, s)
    else:
        #log.info('Compressible blocks')

        res = {}

        queue = SimpleQueue()
        processes_count = 3
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



        if log.is_debug():
            log.info(f'check {len(hss.Partition.level_to_nodes[hss.Partition.max_level])}')
        tmp_HSS = hss.duplicate()  # self.duplicate() # copy.deepcopy(self)
        tmp_HSS.set_matrices(tmpUs, tmpVs, tmpDs)

        z__ = functools.reduce(operator.add, [list(z_[0]) + [0] * z_[1] for z_ in z])
        if log.is_debug():
            log.info(f'z__ {len(z__)}')
            log.info(f'b {len(b)}')

        if log.is_debug():
            log.info(tmp_HSS)

        b = np.array(b).reshape((len(b), 1))
        b___ = tools.diag(q_is) @ b
        b___2 = tmp_HSS.multiply_perfect_binary_tree(z__)
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