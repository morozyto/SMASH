import compression
import log
import partition
import taylor_expansion
import tools

import functools
import operator
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import block_diag


class HSS:

    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A
        self.Partition = partition.Partition(X, Y)
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
                    S_i_, _, _ = randomized_svd(A_,
                                  n_components=15,
                                  n_iter=5,
                                  random_state=None)
                U_i_ = taylor_expansion.form_well_separated_expansion(row_values)
                P, G, new_i = compression.compr(U_i_ if A_ is None else tools.concat_column_wise(U_i_, S_i_), row_indices)
                n = P.shape[0]
                tmp = np.matmul(P, tools.concat_row_wise(np.identity(n - G.shape[0]), G) if n - G.shape[0] > 0 else G)
                return new_i, tmp, n

            for obj in self.Partition.level_to_nodes[l]:
                rows_ind = obj.i_row
                columns_ind = [] if obj.is_root else functools.reduce(operator.add, [t.i_col for t in obj.N])
                A_ = None if obj.is_root else tools.get_block(self.A, rows_ind, columns_ind)

                obj.i_row_cup, tmp, n = inner_get(A_, obj.i_row, [self.X[i] for i in obj.i_row])

                if obj.is_leaf:
                    obj.U = tmp
                else:
                    obj.R = tmp


                rows_ind = [] if obj.is_root else functools.reduce(operator.add, [t.i_row for t in obj.N])
                columns_ind = obj.i_col

                A_t = None if obj.is_root else np.transpose(tools.get_block(self.A, rows_ind, columns_ind))

                obj.i_col_cup, tmp, n = inner_get(A_t, obj.i_col, [self.X[i] for i in obj.i_col])

                if obj.is_leaf:
                    obj.V = tmp
                else:
                    obj.W = tmp


    def multiply(self, q):

        def diag(matrices):
            return block_diag(*matrices)

        def get_B(l):
            if l == self.Partition.max_level:
                return diag([obj.get_D(self.A) for obj in self.Partition.level_to_nodes[l]])
            else:
                return diag([obj.get_B(self.A) for obj in self.Partition.level_to_nodes[l]])

        def get_U(l):
            if l == self.Partition.max_level:
                return diag([obj.U for obj in self.Partition.level_to_nodes[l]])
            else:
                return diag([obj.R for obj in self.Partition.level_to_nodes[l]])

        def get_V(l):
            if l == self.Partition.max_level:
                return diag([obj.V for obj in self.Partition.level_to_nodes[l]])
            else:
                return diag([obj.W for obj in self.Partition.level_to_nodes[l]])

        Z_index = {}
        Q_index = {}

        for l in range(self.Partition.max_level, 1, -1):
            if l == self.Partition.max_level:
                Q_index[l] = np.matmul(np.transpose(get_V(l)), q)
            else:
                Q_index[l] = np.matmul(np.transpose(get_V(l)), Q_index[l + 1])

        for l in range(2, self.Partition.max_level + 1):
            Z_index[l] = np.matmul(get_B(l - 1), Q_index[l])

        z = np.matmul(get_B(self.Partition.max_level), q)

        tmp = np.matmul(get_U(2), Z_index[2])
        for l in range(3, self.Partition.max_level + 1):
            tmp += Z_index[l]
            tmp = np.matmul(get_U(l), tmp)

        z += tmp
        return z



'''
                A_t = tools.get_block(A, sum([t.i_row for t in obj.N]), obj.i_col).transpose()
                T_i_, _, _ = randomized_svd(A_t,
                              n_components=15,
                              n_iter=5,
                              random_state=None)

                V_i_ = taylor_expansion.form_V(obj.i_col)
                F, H, i = compression.compr(tools.concat_column_wise(T_i_, V_i_))
                if obj.is_leaf:
                    obj.V = [F, H]
                else:
                    obj.W = [W_1, W_2]
'''


'''
                for obj in self.Partition.level_to_nodes[l]:
                log.debug('block has {} blocks in N'.format(len(obj.N)))
                rows_ind = obj.i_row
                columns_ind = functools.reduce(operator.add, [t.i_col for t in obj.N])
                log.debug('rows ind {}, column ind {}'.format(rows_ind, columns_ind))
                A_ = tools.get_block(self.A, rows_ind, columns_ind)
                log.debug('A={}'.format(A_))
                S_i_, _, _ = randomized_svd(A_,
                              n_components=15,
                              n_iter=5,
                              random_state=None)
                U_i_ = taylor_expansion.form_U([self.X[i] for i in obj.i_row])
                P, G, obj.i_row_cup = compression.compr(tools.concat_column_wise(U_i_, S_i_), obj.i_row)
                log.debug('DEBUGGG {}'.format(obj.i_row_cup))
                n = P.shape[0]
                #assert n - G.shape[0] > 0
                tmp = np.matmul(P, tools.concat_row_wise(np.identity(n - G.shape[0]), G) if n - G.shape[0] > 0 else G)

                if obj.is_leaf:
                    obj.U = tmp
                else:
                    R_1 = tools.get_block(tmp, [i for i in range(len(obj.Children[0].i_row))], [i for i in range(tmp.shape[1])])
                    R_2 = tools.get_block(tmp, [i for i in range(len(obj.Children[0].i_row), n)], [i for i in range(tmp.shape[1])])
                    obj.R = [R_1, R_2]
        '''
