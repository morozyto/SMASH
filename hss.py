import compression
import log
import partition
import taylor_expansion
import tools
import fast_multiplier_utils
import fast_solver_utils
from tolerance_svd import tolerance_svd

from queue import Queue
import threading
import functools
import operator
import numpy as np


class HSS:

    def __init__(self, X, Y, A, need_to_build=True, max_values_in_node=4):
        self.X = X
        self.Y = Y
        self.A = A
        if need_to_build:
            self.Partition = partition.Partition(X, Y, max_values_in_node=max_values_in_node)
            self.Partition.build_levels()
            self.build()

    def duplicate(self, deepcopy_leaves = False, deepcopy_all = False): #__copy__
        newone = type(self)(self.X, self.Y, self.A, need_to_build=False)
        newone.__dict__.update(self.__dict__)
        newone.Partition = self.Partition.duplicate(deepcopy_leaves, deepcopy_all)
        return newone

    def set_matrices(self, Us, Vs, Ds):
        leaves_len = len(self.Partition.level_to_nodes[self.Partition.max_level])
        assert leaves_len == len(Us) == len(Vs) == len(Ds)
        for i in range(leaves_len):
            obj = self.Partition.level_to_nodes[self.Partition.max_level][i]
            obj.set_D(Ds[i])
            obj.set_U(Us[i])
            obj.set_V(Vs[i])

    def sum(self, rhs):
        assert self.Partition.is_the_same(rhs.Partition)
        res = self.duplicate()
        for k in range(1, self.Partition.max_level + 1):
            for i in range(len(self.Partition.level_to_nodes[k])):
                left_obj = self.Partition.level_to_nodes[k][i]
                right_obj = rhs.Partition.level_to_nodes[k][i]
                res_obj = res.Partition.level_to_nodes[k][i]

                if left_obj.is_leaf:
                    res_obj.D = left_obj.get_D(self.A) + right_obj.get_D(rhs.A)
                    res_obj.U = tools.concat_column_wise(left_obj.get_U(), right_obj.get_U())
                    res_obj.V = tools.concat_column_wise(left_obj.get_V(), right_obj.get_V())
                else:
                    res_obj.Rs = [
                        tools.diag([left_obj.Rs[0], right_obj.Rs[0]]),
                        tools.diag([left_obj.Rs[1], right_obj.Rs[1]]),
                    ]
                    res_obj.Ws = [
                        tools.diag([left_obj.Ws[0], right_obj.Ws[0]]),
                        tools.diag([left_obj.Ws[1], right_obj.Ws[1]]),
                    ]

                if not left_obj.is_root:
                    res_obj.B = tools.diag([left_obj.get_B_subblock(self.A), right_obj.get_B_subblock(rhs.A)])

                res.Partition.level_to_nodes[k][i] = res_obj
        return res

    def build(self):
        log.debug('Start building HSS')
        for l in range(self.Partition.max_level, 0, -1):
            if log.is_debug():
                log.debug('Start building level {}'.format(l))
            for obj in self.Partition.level_to_nodes[l]:
                obj.update_indices()

            def inner_get(A_, row_indices, row_values):
                if A_ is not None:
                    #S_i_, _, _ = svd(A_, check_finite=False)
                    #S_i_, _, _ = randomized_svd(A_,
                     #            n_components=min(A_.shape[0], A_.shape[1]) // 10, #min(A_.shape[0], A_.shape[1]),
                    #             n_iter='auto',
                    #             random_state=1)
                    S_i_ = tolerance_svd(A_)
                    #tol = 10 ** -3
                    #r = TruncatedSVD(algorithm='arpack', tol=tol)
                    #S_i_ = r.transform(A_).dot(np.linalg.inv(np.diag(r.singular_values_)))


                U_i_ = taylor_expansion.form_well_separated_expansion(row_values)
                P, G, new_i = compression.compr(U_i_ if A_ is None else tools.concat_column_wise(U_i_, S_i_), row_indices)
                n = P.shape[0]
                if n - G.shape[0] > 0:
                    if G.shape[0] * G.shape[1] != 0:
                        tmp_ = tools.concat_row_wise(np.identity(n - G.shape[0]), G)
                    else:
                        tmp_ = np.identity(n)
                else:
                    tmp_ = G
                tmp = np.matmul(P, tmp_)
                return new_i, tmp, n

            for obj in self.Partition.level_to_nodes[l]:
                rows_ind = obj.i_row
                columns_ind = [] if obj.is_root else functools.reduce(operator.add, [t.i_col for t in obj.get_N(self.X, self.Y, current_node_is_x=True)])
                A_ = None if obj.is_root else tools.get_block(self.A, rows_ind, columns_ind)

                obj.i_row_cup, tmp, n = inner_get(A_, obj.i_row, [self.X[i] for i in obj.i_row])

                if obj.is_leaf:
                    obj.U = tmp
                else:
                    Rs = [
                        tools.get_block(tmp, [i for i in range(0, len(obj.Children[0].i_row_cup))], [j for j in range(tmp.shape[1])]),
                        tools.get_block(tmp, [i for i in range(len(obj.Children[0].i_row_cup), tmp.shape[0])], [j for j in range(tmp.shape[1])]),
                    ]
                    obj.Rs = Rs


                rows_ind = [] if obj.is_root else functools.reduce(operator.add, [t.i_row for t in obj.get_N(self.X, self.Y, current_node_is_x=False)])
                columns_ind = obj.i_col

                A_t = None if obj.is_root else np.transpose(tools.get_block(self.A, rows_ind, columns_ind))

                obj.i_col_cup, tmp, n = inner_get(A_t, obj.i_col, [self.Y[i] for i in obj.i_col])

                if obj.is_leaf:
                    obj.V = tmp
                else:
                    Ws = [
                        tools.get_block(tmp, [i for i in range(0, len(obj.Children[0].i_col_cup))], [j for j in range(tmp.shape[1])]),
                        tools.get_block(tmp, [i for i in range(len(obj.Children[0].i_col_cup), tmp.shape[0])], [j for j in range(tmp.shape[1])]),
                    ]
                    obj.Ws = Ws

    @property
    def is_perfect_binary_tree(self):
        return self.Partition.is_perfect_binary_tree

    def fast_multiply(self, b, processes_count=1):
        return fast_multiplier_utils.fast_multiply(self.Partition, self.A, b, processes_count=processes_count)

    def multiply_perfect_binary_tree(self, q):

        assert self.is_perfect_binary_tree

        def get_B(l):
            if l == self.Partition.max_level:
                return tools.diag([obj.get_D(self.A) for obj in self.Partition.level_to_nodes[l]])
            else:
                return tools.diag([obj.get_B(self.A) for obj in self.Partition.level_to_nodes[l]])

        def get_U(l):
            if l == self.Partition.max_level:
                return tools.diag([obj.get_U() for obj in self.Partition.level_to_nodes[l]])
            else:
                return tools.diag([obj.get_R() for obj in self.Partition.level_to_nodes[l]])

        def get_V(l):
            if l == self.Partition.max_level:
                return tools.diag([obj.get_V() for obj in self.Partition.level_to_nodes[l]])
            else:
                return tools.diag([obj.get_W() for obj in self.Partition.level_to_nodes[l]])

        Z_index = {}
        Q_index = {}

        V_index = {}

        class WorkerVIndex(threading.Thread):
            def __init__(self, queue):
                threading.Thread.__init__(self)
                self.queue = queue

            def run(self):
                while True:
                    level = self.queue.get()
                    self.do(level)
                    self.queue.task_done()

            def do(self, level):
                V_index[level] = np.transpose(get_V(level))

        queue = Queue()

        for i in range(4):
            worker = WorkerVIndex(queue)
            worker.setDaemon(True)
            worker.start()

        for level in range(self.Partition.max_level, 1, -1):
            queue.put(level)
        queue.join()

        for l in range(self.Partition.max_level, 1, -1):
            if l == self.Partition.max_level:
                Q_index[l] = np.matmul(V_index[l], q)
            else:
                Q_index[l] = np.matmul(V_index[l], Q_index[l + 1])


        class Worker(threading.Thread):
            def __init__(self, queue):
                threading.Thread.__init__(self)
                self.queue = queue

            def run(self):
                while True:
                    level = self.queue.get()
                    self.do(level)
                    self.queue.task_done()

            def do(self, level):
                b = get_B(level - 1)
                Z_index[level] = np.matmul(b, Q_index[level])

        queue = Queue()

        for i in range(4):
            worker = Worker(queue)
            worker.setDaemon(True)
            worker.start()

        for l in range(2, self.Partition.max_level + 1):
            queue.put(l)

        queue.join()

        if log.is_debug():
            log.info(tools.print_matrix(get_B(self.Partition.max_level)))
            log.info(len(q))

        z = np.matmul(get_B(self.Partition.max_level), q)

        tmp = np.matmul(get_U(2), Z_index[2])
        for l in range(3, self.Partition.max_level + 1):
            tmp += Z_index[l]
            tmp = np.matmul(get_U(l), tmp)

        z += tmp

        z = np.array([[z_i] for z_i in z])

        return z

    def fast_solve(self, b, processes_count):
        return fast_solver_utils.solve(self, b, processes_count)

    def remove_last_level(self):
        assert self.Partition.max_level > 1
        for obj in self.Partition.level_to_nodes[self.Partition.max_level - 1]:
            obj.merge_children(self.A)

        del self.Partition.level_to_nodes[self.Partition.max_level]
        self.Partition.max_level -= 1

    def __repr__(self):
        return str(self.Partition)
