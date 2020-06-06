import partition_node

class Partition:

    def __init__(self, X, Y, need_to_build=True, max_values_in_node=4, points_dimension=1):
        # X, Y - lists
        self.X = X
        self.Y = Y
        assert isinstance(points_dimension, int) and points_dimension > 0
        self.points_dimension = points_dimension
        assert len(X) == len(Y)
        if need_to_build:
            self.level_to_nodes = { 1: [partition_node.Node(list(range(len(X))))] }
            self.max_points_in_node = max_values_in_node
            self.max_level = 0

    def is_the_same(self, rhs):
        if self.max_level != rhs.max_level:
            return False
        if self.max_points_in_node != rhs.max_points_in_node:
            return False
        if self.X != rhs.X:
            return False
        if self.Y != rhs.Y:
            return False
        for key in self.level_to_nodes.keys():
            if key not in rhs.level_to_nodes.keys() or len(self.level_to_nodes[key]) != len(rhs.level_to_nodes[key]):
                return False
        return True

    def duplicate(self, deepcopy_leaves = False, deepcopy_all = False):
        newone = type(self)(self.X, self.Y, )
        newone.__dict__.update(self.__dict__)

        new_root = self.level_to_nodes[1][0].duplicate(deepcopy_leaves, deepcopy_all)
        newone.level_to_nodes = {}

        current_level_nodes = [new_root]
        current_level = 1

        while True:

            newone.level_to_nodes[current_level] = current_level_nodes
            current_level += 1

            if current_level > self.max_level:
                break

            tmp = []
            for obj in current_level_nodes:
                tmp += obj.Children

            current_level_nodes = tmp


        newone.count_N()
        return newone

    def count_N(self):
        current_level = 1
        while current_level < self.max_level:
            current_nodes = self.level_to_nodes[current_level]
            for obj in current_nodes:
                obj.get_N(self.X, self.Y, current_node_is_x=True)
                obj.get_N(self.X, self.Y, current_node_is_x=False)
            current_level += 1

    def build_levels(self, X=None):
        current_level = 1
        current_dimension = 0
        if self.points_dimension == 1:
            save_mid_lines = None
        else:
            save_mid_lines = {}

        while current_level in self.level_to_nodes:
            current_nodes = self.level_to_nodes[current_level]
            need_next_level = any([obj.count_of_points > self.max_points_in_node for obj in current_nodes])
            if need_next_level:
                for obj in current_nodes:
                    next_level_nodes = self.level_to_nodes.get(current_level + 1, [])
                    if self.points_dimension == 1:
                        saver = None
                    else:
                        saver = {}
                    next_level_nodes += obj.divide_by_half(X=X, points_dimension=self.points_dimension,
                                                           current_dimension=current_dimension, saver=saver)
                    if self.points_dimension != 1:
                        save_mid_lines[(current_level, len(next_level_nodes) // 2 - 1)] = saver['data']
                    self.level_to_nodes[current_level + 1] = next_level_nodes
            current_level += 1
            current_dimension = (current_dimension + 1) % self.points_dimension

        self.max_level = current_level - 1

        if X is not None:
            tmp = []
            for obj in self.level_to_nodes[self.max_level]:
                tmp += obj.Indices

            new_points =  [X[i] for i in tmp]
            X = new_points

            start_ind = 0
            for obj in self.level_to_nodes[self.max_level]:
                obj.update_build_indices([i for i in range(start_ind, start_ind + len(obj.Indices))])
                start_ind += len(obj.Indices)

            for i in range(self.max_level - 1, 0, -1):
                for obj in self.level_to_nodes[i]:
                    obj.update_build_indices()


            return X, save_mid_lines
        else:
            self.count_N()

    @property
    def is_perfect_binary_tree(self):
        return all([len(self.level_to_nodes[l]) == 2 ** (l - 1) for l in range(self.max_level, 0, -1)])

    def __repr__(self):
        res = ''
        for level in sorted(self.level_to_nodes.keys(), reverse=True):
            res += 'Printing level {}\n'.format(level)
            for obj in self.level_to_nodes[level]:
                res += str(obj) + '\n'
        return res
