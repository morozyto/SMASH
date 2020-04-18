from tools import *

class Node:

    def __init__(self, indices, parent = None):
        assert indices

        self.Parent = parent
        self.Children = None
        self.Indices = indices
        self.is_leaf = True
        self.U = None
        self.V = None
        self.R = None # list
        self.W = None # list
        self.i_row = indices
        self.i_col = indices
        self.i_row_cup = indices
        self.i_col_cup = indices
        self.N_data = None

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

    @property
    def N(self):
        return self.get_N()

    def update_indices(self):
        if self.Children is None:
            return
        self.i_row =
        self.i_col =

    def get_N(self):
        if self.Parent is None:
            return []
        if self.N_data is not None:
            return self.N_data

        self.N_data = [self.sibling]

        for p_k in self.Parent.N:
            for child in p_k.Children:
                if not self.is_farfield(child):
                    self.N_data.append(child)

        return self.N_data

    def divide_by_half(self):
        mid = len(self.Indices) // 2
        left = Node(self.Indices[:mid], self)
        right = Node(self.Indices[mid:], self)

        self.Children = [left, right]
        self.get_N()

        self.is_leaf = False
        self.i_row = None
        self.i_col = None
        return self.Children

    def is_farfield(self, another_node, r = 0.7):

        center, radius = get_metadata(self.Indices)
        a_center, a_radius = get_metadata(another_node.Indices)

        return radius + a_radius <= r*abs(center - a_center)