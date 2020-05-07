import partition_node


class Partition:

    def __init__(self, X, Y = None):
        # X, Y - lists
        self.X = X
        self.Y = Y
        assert len(X) == len(Y)
        self.level_to_nodes = { 1: [partition_node.Node(list(range(len(X))))] }
        self.max_points_in_node = 4
        self.max_level = 0

    def build_levels(self):
        current_level = 1
        while current_level in self.level_to_nodes:
            current_nodes = self.level_to_nodes[current_level]
            for obj in current_nodes:
                if obj.count_of_points > self.max_points_in_node:
                    next_level_nodes = self.level_to_nodes.get(current_level + 1, [])
                    next_level_nodes += obj.divide_by_half()
                    self.level_to_nodes[current_level + 1] = next_level_nodes
            current_level += 1
        self.max_level = current_level - 1

        current_level = 1
        while current_level < self.max_level:
            current_nodes = self.level_to_nodes[current_level]
            for obj in current_nodes:
                obj.get_N(current_node_is_x=True)
                obj.get_N(current_node_is_x=False)
            current_level += 1

