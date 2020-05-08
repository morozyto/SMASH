from tools import *
import log

class Node:

    def __init__(self, indices, parent = None):
        assert indices

        self.Parent = parent
        self.Children = None
        self.Indices = indices
        self.is_leaf = True
        self.U = None
        self.V = None
        self.R = None
        self.W = None
        self.i_row = indices
        self.i_col = indices
        self.i_row_cup = indices
        self.i_col_cup = indices
        self.N_data = {}

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
            for child in p_k.Children:
                if not self.is_farfield(child, X, Y, current_node_is_x = current_node_is_x):
                    tmp.append(child)

        self.N_data[current_node_is_x] = tmp
        return tmp

    def divide_by_half(self):
        mid = len(self.Indices) // 2
        log.debug('Partition: left {}, right {}'.format(self.Indices[:mid], self.Indices[mid:]))
        left = Node(self.Indices[:mid], self)
        right = Node(self.Indices[mid:], self)

        self.Children = [left, right]

        self.is_leaf = False
        self.i_row = None
        self.i_col = None
        self.i_row_cup = None
        self.i_col_cup = None
        return self.Children

    def is_farfield(self, another_node, X, Y, current_node_is_x = True, r = 0.7):
        myself_points = X if current_node_is_x else Y
        another_points = Y if current_node_is_x else X  # another_node.Indices
        center, radius = get_metadata([myself_points[t] for t in self.Indices])
        a_center, a_radius = get_metadata([another_points[t] for t in another_node.Indices])

        return radius + a_radius <= r*abs(center - a_center)

    def get_D(self, A):
        assert self.is_leaf
        return get_block(A, self.Indices, self.Indices)

    def get_B_subblock(self, A):
        return get_block(A, self.i_row, self.sibling.i_col)

    def get_B(self, A):
        assert not self.is_leaf
        B_1 = self.Children[0].get_B_subblock(A)
        B_2 = self.Children[1].get_B_subblock(A)

        first_row = concat_column_wise(np.zeros((B_1.shape[0], B_2.shape[1])), B_1)
        second_row = concat_column_wise(B_2, np.zeros((B_2.shape[0], B_1.shape[1])))
        return concat_row_wise(first_row, second_row)
