import compression
import log
import partition
import taylor_expansion
import tools

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

    def build(self):
        log.debug('Start building HSS')
        for l in range(self.Partition.max_level, 0, -1):
            log.debug('Start building level {}'.format(l))
            for obj in self.Partition.level_to_nodes[l]:
                obj.update_indices()

            def inner_get(A_, row_indices, row_values):
                if A_ is not None:
                    S_i_, _, _ = svd(A_, check_finite=False)
                    #S_i_, _, _ = randomized_svd(A_,
                                  #n_components=max(A_.shape[0], A_.shape[1]) // 2,
                                  #n_iter='auto',
                                  #random_state=None)
                    #r = TruncatedSVD(algorithm='')


                U_i_ = taylor_expansion.form_well_separated_expansion(row_values)
                P, G, new_i = compression.compr(U_i_ if A_ is None else tools.concat_column_wise(U_i_, S_i_), row_indices)
                n = P.shape[0]
                if n - G.shape[0] > 0:
                    log.debug(f'G shape {G.shape}')
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
            log.debug('check level {}'.format(l))
            for obj in self.Partition.level_to_nodes[l]:
                tmp = len(obj.Children) if obj.Children else 0
                if children_count is None:
                    children_count = tmp
                else:
                    if tmp != children_count:
                        return False
        return True

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

        for l in range(self.Partition.max_level, 1, -1):
            if l == self.Partition.max_level:
                Q_index[l] = np.matmul(np.transpose(get_V(l)), q)
            else:
                Q_index[l] = np.matmul(np.transpose(get_V(l)), Q_index[l + 1])

        for l in range(2, self.Partition.max_level + 1):
            log.debug('counting Z level {}'.format(l))
            b = get_B(l - 1)
            Z_index[l] = np.matmul(b, Q_index[l])

        z = np.matmul(get_B(self.Partition.max_level), q)

        tmp = np.matmul(get_U(2), Z_index[2])
        for l in range(3, self.Partition.max_level + 1):
            tmp += Z_index[l]
            tmp = np.matmul(get_U(l), tmp)

        z += tmp
        return z

    def solve(self, b):
        if self.Partition.max_level == 1:
            assert len(self.Partition.level_to_nodes[1]) == 1
            tmp = self.Partition.level_to_nodes[1][0].get_D(self.A)
            log.debug(f'Compressed A is \n{tmp}')
            return np.linalg.solve(tmp, b)

        has_incompressible_blocks = True
        if has_incompressible_blocks:
            self.remove_last_level()
            return self.solve(b)
        else:
            '''
            tmpDs = []
            tmpUs = []
            tmpVs = []

            newDs = []
            newUs = []
            newVs = []

            z = []

            q_is = []
            w_is = []

            for obj in self.Partition.level_to_nodes[self.Partition.max_level]:
                q_i, tmpU = tools.ql(obj.get_U())
                n_i = tmpU.shape[1]
                new_U = tools.get_block(tmpU, i for i in range(tmpU.shape[0] - n_i, n_i), j for j in range(n_i))


                b_i = get_block(b, obj.Indices, 0)
                t = np.transpose(q_i) @ b_i
                beta = t[:len(t) - n_i]
                gamma = t[len(t) - n_i:]


                t = np.transpose(q_i) @ obj.get_D()
                tmpD, w_i = tools.lq(t)
                new_D = get_block(tmpD, i for i in range(tmpD.shape[0] - n_i, n_i), j for j in range(tmpD.shape[1] - n_i, n_i))


                tmpV = w_i @ obj.get_V()
                new_V = tools.get_block(tmpV, i for i in range(tmpV.shape[0] - n_i, n_i), j for j in range(tmpV.shape[1]))


                z_i = np.linalg.solve(get_block(tmpD, i for i in range(n_i), j for j in range(n_i)), beta)
                z_i += [0] * n_i

                tmpDs.append(tmp_D)
                tmpUs.append(tmpU)
                tmpVs.append(tmpV)

                newDs.append(new_D)
                newUs.append(new_U)
                newVs.append(new_V)

                z.append((z_i, n_i))
                q_is.append(np.transpose(q_i))
                w_is.append(np.transpose(w_i))


            tmp_HSS = build_HSS(tmpDs, tmpUs, tmpVs, ...)
            new_b = diag(q_is) @ b - tmp_HSS.multiply_perfect_binary_tree(tools.concat_row_wise([z_[0] for z_ in z]))


            new_HSS = build_HSS(newDs, newUs, newVs, ...)
            tmp_x = new_HSS.solve(new_b)

            i = 0
            for z_ in z:
                z_[0][-z_[1]:] = tmp_x[i:i + z_[1]]
                i += z_[1]


            return tools.diag(w_is) @ tools.concat_row_wise([z_[0] for z_ in z])

            '''
            pass

    def remove_last_level(self):
        assert self.Partition.max_level > 1
        for obj in self.Partition.level_to_nodes[self.Partition.max_level - 1]:
            obj.merge_children(self.A)

        del self.Partition.level_to_nodes[self.Partition.max_level]
        self.Partition.max_level -= 1

    def __repr__(self):
        return str(self.Partition)