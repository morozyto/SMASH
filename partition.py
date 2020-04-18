from partition_node import *

class Partition:

    def __init__(self, X):
        # X, Y - lists
        self.X = X
        self.level_to_nodes = { 1: [Node(list(range(len(X))))] }
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
                obj.get_N()
            current_level += 1

