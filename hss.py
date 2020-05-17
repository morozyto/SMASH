import compression
import log
import partition
import taylor_expansion
import tools

import functools
import operator
import numpy as np
from sklearn.utils.extmath import randomized_svd


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
            log.debug(f'Compressed A is {tmp}')
            return np.linalg.solve(tmp, b)

        has_incompressible_blocks = True
        if has_incompressible_blocks:
            self.remove_last_level()
            return self.solve(b)
        else:
            pass

    def remove_last_level(self):
        assert self.Partition.max_level > 1
        for obj in self.Partition.level_to_nodes[self.Partition.max_level - 1]:
            obj.merge_children(self.A)

        del self.Partition.level_to_nodes[self.Partition.max_level]
        self.Partition.max_level -= 1

    def __repr__(self):
        return str(self.Partition)