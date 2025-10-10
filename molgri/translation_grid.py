from functools import cached_property

import networkx as nx
import numpy as np
from numpy._typing import NDArray
from scipy.spatial import ConvexHull


class TranslationNode:

    """
    This class is a single node, representing one particular quaternion.
    """

    def __init__(self, translation_index: int, coordinate_3d: NDArray, hull = None):
        self.translation_index = translation_index
        self.coordinate_3d = coordinate_3d
        self.hull = hull

    def __str__(self):
        return f'quat={self.translation_index}'

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.translation_index < other.translation_index

    @cached_property
    def volume(self):
        my_convex_hull = ConvexHull(self.hull, qhull_options='QJ')
        return my_convex_hull.volume


class CartesianTranslationObject:

    def __init__(self, x_grid, y_grid, z_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        X, Y, Z = np.meshgrid(self.x_grid, self.y_grid, self.z_grid, indexing='ij')
        self.grid = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        self.delta_x = x_grid[1] - x_grid[0]
        self.delta_y = y_grid[1] - y_grid[0]
        self.delta_z = z_grid[1] - z_grid[0]

    @cached_property
    def hulls(self):
        all_hulls = []
        for point in self.grid:
            hull = get_hull_cuboids(*point, delta_x=self.delta_x, delta_y=self.delta_y, delta_z=self.delta_z)
            all_hulls.append(hull)
        return all_hulls

    @cached_property
    def adjacency(self):
        pass

    def get_translation_network(self) -> nx.Graph:
        G = nx.Graph()

        all_nodes = [TranslationNode(trans_i, coo, self.hulls[trans_i]) for trans_i, coo in enumerate(self.grid)]
        G.add_nodes_from(all_nodes)
        for node_i_1, node_i_2 in zip(self.adjacency.row, self.adjacency.col):
            node1 = all_nodes[node_i_1]
            node2 = all_nodes[node_i_2]
            G.add_edge(node1, node2, edge_type="cartesian")

        # all the properties we want to calculate
        calculate_translation_edge_attributes(G)
        return G

def get_hull_cuboids(x, y, z, delta_x, delta_y, delta_z):
    return np.array([
    [x + delta_x / 2, y + delta_y / 2, z + delta_z / 2],
    [x + delta_x / 2, y + delta_y / 2, z - delta_z / 2],
    [x + delta_x / 2, y - delta_y / 2, z + delta_z / 2],
    [x - delta_x / 2, y + delta_y / 2, z + delta_z / 2],
    [x + delta_x / 2, y - delta_y / 2, z - delta_z / 2],
    [x - delta_x / 2, y - delta_y / 2, z + delta_z / 2],
    [x - delta_x / 2, y + delta_y / 2, z - delta_z / 2],
    [x-delta_x/2, y-delta_y/2, z-delta_z/2]
    ])