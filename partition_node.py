import tools
import log

import numpy as np
import copy


class Node:

    def __init__(self, indices, parent = None):
        assert indices

        self.Parent = parent
        self.Children = None
        self.Indices = indices
        self.is_leaf = True
        self.U = None
        self.V = None
        self.Rs = None
        self.Ws = None
        self.D = None
        self.B = None
        self.i_row = indices
        self.i_col = indices
        self.i_row_cup = indices
        self.i_col_cup = indices
        self.N_data = {}

    @property
    def R(self):
        if self.Rs and len(self.Rs) == 2:
            return tools.concat_row_wise(self.Rs[0], self.Rs[1])
        return None

    @property
    def W(self):
        if self.Ws and len(self.Ws) == 2:
            return tools.concat_row_wise(self.Ws[0], self.Ws[1])
        return None

    def duplicate(self, deepcopy_leaves = False, deepcopy_all = False):
        newone = type(self)(self.Indices)
        newone.__dict__.update(self.__dict__)

        if not newone.is_leaf:
            children = []
            for child in self.Children:
                children.append(child.duplicate(deepcopy_leaves, deepcopy_all))
                children[-1].Parent = newone

            newone.Children = children

        if (deepcopy_leaves or deepcopy_all) and newone.is_leaf:
            newone.U = copy.deepcopy(self.U)
            newone.V = copy.deepcopy(self.V)
            newone.D = copy.deepcopy(self.D)

        if (deepcopy_all):
            self.Rs = copy.deepcopy(self.Rs)
            self.Ws = copy.deepcopy(self.Ws)
            self.B = copy.deepcopy(self.B)

        return newone

    @property
    def is_root(self):
        return self.Parent is None

    @property
    def sibling(self):
        if self.Parent is None:
            return None
        assert len(self.Parent.Children) == 2
        if self.Parent.Children[0] == self:
            return self.Parent.Children[1]
        else:
            return self.Parent.Children[0]

    @property
    def count_of_points(self):
        return len(self.Indices)

    def set_D(self, D):
        assert self.is_leaf
        self.D = D

    def set_U(self, U):
        assert self.is_leaf
        self.U = U

    def set_V(self, V):
        assert self.is_leaf
        self.V = V

    def update_indices(self):
        if self.Children is None:
            return

        self.i_row = self.Children[0].i_row_cup + self.Children[1].i_row_cup
        self.i_col = self.Children[0].i_col_cup + self.Children[1].i_col_cup

    def get_N(self, X, Y, current_node_is_x):
        if self.Parent is None:
            return []
        if current_node_is_x in self.N_data.keys():
            return self.N_data[current_node_is_x]

        tmp = [self.sibling]

        for p_k in self.Parent.get_N(X, Y, current_node_is_x):
            if p_k.is_leaf:
                if not self.is_farfield(p_k, X, Y, current_node_is_x=current_node_is_x):
                    tmp.append(p_k)
            else:
                for child in p_k.Children:
                    if not self.is_farfield(child, X, Y, current_node_is_x = current_node_is_x):
                        tmp.append(child)

        self.N_data[current_node_is_x] = tmp
        return tmp

    def update_build_indices(self, new_indices=None):
        if new_indices is not None:
            assert self.is_leaf
            self.Indices = new_indices
        else:
            assert not self.is_leaf
            self.Indices = self.Children[0].Indices + self.Children[1].Indices


    def divide_by_half(self, points_dimension=1, current_dimension=0, X=None, saver=None):
        self.dimension = points_dimension
        if X is None:
            sorted_ = [(i, None) for i in self.Indices]
        else:
            sorted_ = sorted([(i, X[i]) for i in self.Indices], key=lambda obj: obj[1] if isinstance(obj[1], float) else (obj[1][current_dimension]))
        mid = len(sorted_) // 2

        future_left = sorted_[:mid]
        future_right = sorted_[mid:]

        left_inds = [obj[0] for obj in future_left]
        right_inds = [obj[0] for obj in future_right]

        if saver is not None:
            if len(sorted_) % 2 == 0:
                saver['data'] = (current_dimension, (sorted_[mid][1][current_dimension] + sorted_[mid - 1][1][current_dimension]) / 2)
            else:
                saver['data'] = (current_dimension, sorted_[mid][1][current_dimension])

        log.debug(f'Partition: left {left_inds}, right {right_inds}')

        left = Node(left_inds, self)
        right = Node(right_inds, self)

        self.Children = [left, right]
        self.is_leaf = False
        self.i_row = None
        self.i_col = None
        self.i_row_cup = None
        self.i_col_cup = None
        return self.Children

    def is_farfield(self, another_node, X, Y, current_node_is_x = True):
        myself_points = X if current_node_is_x else Y
        another_points = Y if current_node_is_x else X  # another_node.Indices
        center, radius = tools.get_metadata([myself_points[t] for t in self.Indices])
        a_center, a_radius = tools.get_metadata([another_points[t] for t in another_node.Indices])

        return radius + a_radius <= tools.SEPARATION_RATIO*abs(center - a_center)

    def get_D(self, A):
        assert self.is_leaf
        if self.D is None:
            return tools.get_block(A, self.Indices, self.Indices)
        else:
            return self.D

    def get_B_subblock(self, A):
        if self.B is None:
            return tools.get_block(A, self.i_row_cup, self.sibling.i_col_cup)
        else:
            return self.B

    def get_R(self):
        return self.R

    def get_W(self):
        return self.W

    def get_U(self):
        return self.U

    def get_V(self):
        return self.V

    def merge_children(self, A):
        self.U = np.matmul(tools.diag([self.Children[0].get_U(), self.Children[1].get_U()]), self.R)

        self.V = np.matmul(tools.diag([self.Children[0].get_V(), self.Children[1].get_V()]), self.W)

        self.D = tools.concat_column_wise(
            tools.concat_row_wise(self.Children[0].get_D(A),
                                  self.Children[1].get_U() @ self.Children[1].get_B_subblock(A) @ np.transpose(self.Children[0].get_V())),
            tools.concat_row_wise(self.Children[0].get_U() @ self.Children[0].get_B_subblock(A) @ np.transpose(self.Children[1].get_V()),
                                  self.Children[1].get_D(A))
            )

        self.Children = None
        self.is_leaf = True
        self.Rs = None
        self.Ws = None


    def __repr__(self):

        return f'Node\nIndices={self.Indices}, is_leaf={self.is_leaf}, U=\n{tools.print_matrix(self.U)},' \
               f' V=\n{tools.print_matrix(self.V)}, R=\n{tools.print_matrix(self.R)}, W=\n{tools.print_matrix(self.W)},' \
               f' D=\n{tools.print_matrix(self.D)}, ' \
               f' i_row={len(self.i_row)}, i_col={len(self.i_col)}, ' \
               f'i_row_cup={len(self.i_row_cup)}, i_col_cup={len(self.i_col_cup)}'