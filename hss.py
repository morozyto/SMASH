import compression
import log
import partition
import taylor_expansion
import tools

import time
import copy
import functools
import operator
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd


class HSS:

    def __init__(self, X, Y, A, max_values_in_node=4):
        self.X = X
        self.Y = Y
        self.A = A
        self.Partition = partition.Partition(X, Y, max_values_in_node)
        self.Partition.build_levels()
        self.build()



    def set_matrices(self, Us, Vs, Ds):
        leaves_len = len(self.Partition.level_to_nodes[self.Partition.max_level])
        assert leaves_len == len(Us) == len(Vs) == len(Ds)
        for i in range(leaves_len):
            obj = self.Partition.level_to_nodes[self.Partition.max_level][i]
            obj.set_D(Ds[i])
            obj.set_U(Us[i])
            obj.set_V(Vs[i])


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
                    S_i_, _, _ = randomized_svd(A_,
                                 n_components=50, #min(A_.shape[0], A_.shape[1]),
                                 n_iter='auto',
                                 random_state=None)
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
                    obj.R = tmp


                rows_ind = [] if obj.is_root else functools.reduce(operator.add, [t.i_row for t in obj.get_N(self.X, self.Y, current_node_is_x=False)])
                columns_ind = obj.i_col

                A_t = None if obj.is_root else np.transpose(tools.get_block(self.A, rows_ind, columns_ind))

                obj.i_col_cup, tmp, n = inner_get(A_t, obj.i_col, [self.Y[i] for i in obj.i_col])

                if obj.is_leaf:
                    obj.V = tmp
                else:
                    obj.W = tmp

    @property
    def is_perfect_binary_tree(self):
        for l in range(self.Partition.max_level, 0, -1):
            children_count = None
            for obj in self.Partition.level_to_nodes[l]:
                tmp = len(obj.Children) if obj.Children else 0
                if children_count is None:
                    children_count = tmp
                else:
                    if tmp != children_count:
                        return False
        return True

    def multiply_perfect_binary_tree(self, q):

        t = time.process_time()
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

        for l in range(self.Partition.max_level, 1, -1):
            if l == self.Partition.max_level:
                Q_index[l] = np.matmul(np.transpose(get_V(l)), q)
            else:
                Q_index[l] = np.matmul(np.transpose(get_V(l)), Q_index[l + 1])

        log.info(f'first multiplication stage timing {time.process_time() - t}')
        t = time.process_time()

        for l in range(2, self.Partition.max_level + 1):
            tmp = time.process_time()
            b = get_B(l - 1)
            log.info(f'second multiplication substage 1 timing {time.process_time() - tmp}')
            tmp = time.process_time()
            Z_index[l] = np.matmul(b, Q_index[l])
            log.info(f'second multiplication substage 2 timing {time.process_time() - tmp}')

        log.info(f'second multiplication stage timing {time.process_time() - t}')
        t = time.process_time()

        if log.is_debug():
            log.info(tools.print_matrix(get_B(self.Partition.max_level)))
            log.info(len(q))

        z = np.matmul(get_B(self.Partition.max_level), q)

        tmp = np.matmul(get_U(2), Z_index[2])
        for l in range(3, self.Partition.max_level + 1):
            tmp += Z_index[l]
            tmp = np.matmul(get_U(l), tmp)

        z += tmp

        log.info(f'third multiplication stage timing {time.process_time() - t}')
        t = time.process_time()

        return z

    def solve(self, b):
        if self.Partition.max_level == 1:
            assert len(self.Partition.level_to_nodes[1]) == 1
            tmp = self.Partition.level_to_nodes[1][0].get_D(self.A)
            return np.linalg.solve(tmp, b)

        no_compressible_blocks = all([obj.U.shape[0] <= obj.U.shape[1]
                                         for obj in self.Partition.level_to_nodes[self.Partition.max_level]])
        if no_compressible_blocks:
            log.info('No compressible blocks')
            tmp = copy.copy(self)
            tmp.remove_last_level()
            return tmp.solve(b)
        else:
            log.info('Compressible blocks')

            tmpDs = []
            tmpUs = []
            tmpVs = []

            newDs = []
            newUs = []
            newVs = []

            z = []

            q_is = []
            w_is = []

            start_index = 0
            for obj in self.Partition.level_to_nodes[self.Partition.max_level]:
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

                    new_U = tools.get_block(tmpU, [i for i in range(tmpU.shape[1] - n_i, tmpU.shape[1])], [j for j in range(n_i)])

                    if log.is_debug():
                        log.info(f'new_U={tools.print_matrix(new_U)}')

                    b_i = b[start_index:start_index + obj.U.shape[0]]
                    if log.is_debug():
                        log.info(f'b_i len={len(b_i)}')

                    t = np.transpose(q_i) @ b_i
                    if log.is_debug():
                        log.info(f't len={len(t)}')

                    beta = t[:len(t) - n_i]
                    gamma = t[len(t) - n_i:]

                    t = np.transpose(q_i) @ obj.get_D(self.A)
                    if log.is_debug():
                        log.info(f't ={tools.print_matrix(t)}')

                    tmpD, w_i = tools.lq(t)
                    if log.is_debug():
                        log.info(f'tmpD ={tools.print_matrix(tmpD)}')
                        log.info(f'w_i ={tools.print_matrix(w_i)}')

                    new_D = tools.get_block(tmpD, [i for i in range(tmpD.shape[0] - n_i, tmpD.shape[0])], [j for j in range(tmpD.shape[0] - n_i, tmpD.shape[0])])

                    tmpV = w_i @ obj.get_V()

                    if log.is_debug():
                        log.info(f'tmpV ={tools.print_matrix(tmpV)}')

                    new_V = tools.get_block(tmpV, [i for i in range(tmpV.shape[0] - n_i, tmpD.shape[0])], [j for j in range(tmpV.shape[1])])

                    z_i = np.linalg.solve(tools.get_block(tmpD, [i for i in range(tmpD.shape[0] - n_i)], [j for j in range(tmpD.shape[0] - n_i)]), beta)

                    if log.is_debug():
                        log.info(f'n_i ={n_i}')
                        log.info(f'z_i ={len(z_i)}')
                    z_i = list(z_i)
                    if log.is_debug():
                        log.info(f'z_i ={len(z_i)}')

                else:
                    if log.is_debug():
                        log.info('incompressible rows')
                    tmpD = obj.get_D(self.A)
                    tmpU = obj.get_U()
                    tmpV = obj.get_V()

                    new_D = obj.get_D(self.A)
                    new_U = obj.get_U()
                    new_V = obj.get_V()

                    n_i = obj.get_U().shape[0]
                    z_i = []
                    q_i = np.identity(obj.get_D(self.A).shape[0])
                    w_i = np.identity(obj.get_D(self.A).shape[1])

                tmpDs.append(tmpD)
                tmpUs.append(tmpU)
                tmpVs.append(tmpV)

                newDs.append(new_D)
                newUs.append(new_U)
                newVs.append(new_V)

                z.append((z_i, n_i))
                q_is.append(np.transpose(q_i))
                w_is.append(np.transpose(w_i))

                start_index += obj.U.shape[0]

            if log.is_debug():
                log.info(f'check {len(self.Partition.level_to_nodes[self.Partition.max_level])}')
            tmp_HSS = copy.copy(self) # self.duplicate() # copy.deepcopy(self)
            tmp_HSS.set_matrices(tmpUs, tmpVs, tmpDs)

            z__ = functools.reduce(operator.add, [list(z_[0]) + [0] * z_[1] for z_ in z])
            if log.is_debug():
                log.info(f'z__ {len(z__)}')
                log.info(f'b {len(b)}')

            if log.is_debug():
                log.info(tmp_HSS)
            b_ = tools.diag(q_is) @ b - tmp_HSS.multiply_perfect_binary_tree(z__)
            new_b = []
            i = 0
            for obj in z:
                i += len(obj[0])
                j = 0
                while j < obj[1]:
                    new_b.append(b_[i])
                    i += 1
                    j += 1


            new_HSS = copy.copy(self)
            new_HSS.set_matrices(newUs, newVs, newDs)

            if log.is_debug():
                log.info(tmp_HSS)
            tmp_x = new_HSS.solve(new_b)

            i = 0
            tmp3 = []
            for z_ in z:
                tmp3 += list(z_[0]) + list(tmp_x[i:i + z_[1]])
                i += z_[1]

            return tools.diag(w_is) @ tmp3


    def remove_last_level(self):
        assert self.Partition.max_level > 1
        for obj in self.Partition.level_to_nodes[self.Partition.max_level - 1]:
            obj.merge_children(self.A)

        del self.Partition.level_to_nodes[self.Partition.max_level]
        self.Partition.max_level -= 1

    def __repr__(self):
        return str(self.Partition)