from functools import cached_property
from itertools import product

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

class OneDimTranslationNode:

    def __init__(self, direction: str, index: int, coordinate: float, hull: tuple) -> None:
        self.index = index
        self.direction = direction
        self.coordinate = coordinate
        self.hull = hull

    def __str__(self) -> str:
        return f"{self.direction} grid, index: {self.index}, coordinate: {self.coordinate}"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.index < other.index


class TranslationNode:

    """
    This class is a single container node with sub-nodes x, y and z, representing one particular
    translation.
    """

    def __init__(self, x: OneDimTranslationNode, y: OneDimTranslationNode, z: OneDimTranslationNode):
        self.x = x
        self.y = y
        self.z = z
        self.coordinate_3d = np.array([self.x.coordinate, self.y.coordinate, self.z.coordinate])

    def __str__(self):
        return f'({self.x.index}, {self.y.index}, {self.z.index})'

    def get_three_indices(self):
        return [self.x.index, self.y.index, self.z.index]

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting): we compare first x, then y, then z
        """
        return self.get_three_indices() < other.get_three_indices()

    @cached_property
    def hull(self):
        x_hull = self.x.hull
        y_hull = self.y.hull
        z_hull = self.z.hull
        all_vertices = list(product(x_hull, y_hull, z_hull))
        return np.array(all_vertices)

    @cached_property
    def volume(self):
        my_convex_hull = ConvexHull(self.hull, qhull_options='QJ')
        return my_convex_hull.volume

class TranslationNetwork(nx.Graph):

    @cached_property
    def sorted_nodes(self):
        nodes = [node for node in sorted(self.nodes)]
        return nodes

    @cached_property
    def grid(self):
        coordinates = [node.coordinate_3d for node in self.sorted_nodes]
        return np.array(coordinates)

    @cached_property
    def volumes(self):
        volumes = [node.volume for node in self.sorted_nodes]
        return np.array(volumes)

    @cached_property
    def hulls(self):
        hulls = [node.hull for node in self.sorted_nodes]
        return hulls

    @cached_property
    def delta_x(self) -> float:
        first_node = self.sorted_nodes[0]
        return first_node.x.hull[1] - first_node.x.hull[0]

    @cached_property
    def delta_y(self) -> float:
        first_node = self.sorted_nodes[0]
        return first_node.y.hull[1] - first_node.y.hull[0]

    @cached_property
    def delta_z(self) -> float:
        first_node = self.sorted_nodes[0]
        return first_node.z.hull[1] - first_node.z.hull[0]

    def _distances(self) -> dict:
        return {"x": self.delta_x, "y": self.delta_y, "z": self.delta_z}

    def _surfaces(self) -> dict:
        return {"x": self.delta_y*self.delta_z, "y": self.delta_x*self.delta_z, "z": self.delta_x*self.delta_y}

    def _numerical_edge_type(self) -> dict:
        return {"x": 1, "y": 2, "z": 3}

    def calculate_all_edge_properties(self):
        df_edges = nx.to_pandas_edgelist(self)
        # now list all properties to be calculated
        df_edges["numerical_edge_type"] = df_edges.apply(
            lambda row: self._numerical_edge_type()[row["edge_type"]], axis=1)
        df_edges["distance"] = df_edges.apply(
            lambda row: self._distances()[row["edge_type"]], axis=1)
        df_edges["surface"] = df_edges.apply(
            lambda row: self._surfaces()[row["edge_type"]], axis=1)
        for attribute in ["distance", "surface", "numerical_edge_type"]:
            nx.set_edge_attributes(self, df_edges.set_index(["source", "target"])[attribute].to_dict(), name=attribute)


def create_translation_network(algorithm_keyword: str = "cartesian_nonperiodic", *args, **kwargs) -> TranslationNetwork:
    match algorithm_keyword:
        case "cartesian_nonperiodic":
            return _create_cartesian_network("none", *args, **kwargs)
        case "cartesian_periodic":
            return _create_cartesian_network("xyz", *args, **kwargs)
        case "cartesian_xy_periodic":
            return _create_cartesian_network("xy", *args, **kwargs)
        case "radial":
            # todo
            pass
        case _:
            raise KeyError(f"{algorithm_keyword} is not a valid rotation algorithm keyword")

def _create_cartesian_network(periodic_in_dimensions,
                              x_linspace_params, y_linspace_params, z_linspace_params):
    x_grid = np.linspace(*x_linspace_params)
    y_grid = np.linspace(*y_linspace_params)
    z_grid = np.linspace(*z_linspace_params)

    sub_networks = []
    labels = ("x", "y", "z")
    subgrids = (x_grid, y_grid, z_grid)
    for label, sub_grid in zip(labels, subgrids):
        delta_coo = sub_grid[1] - sub_grid[0]
        nodes = []
        for coo_i, coo in enumerate(sub_grid):
            hull = (coo - delta_coo / 2, coo + delta_coo / 2)
            nodes.append(OneDimTranslationNode(label, coo_i, coo, hull))
        G = nx.Graph()
        G.add_nodes_from(nodes)
        # now add edges to these sub-graphs - this is without periodicity
        for node_1, node_2 in zip(nodes[:-1], nodes[1:]):
            G.add_edge(node_1, node_2, edge_type=label)
        # possible periodicity - add edge between first and last element
        if label in periodic_in_dimensions:
            G.add_edge(nodes[0], nodes[-1], edge_type=label)
        sub_networks.append(G)

    # now combine the sub-networks
    xy_network = nx.cartesian_product(sub_networks[0], sub_networks[1])
    full_network = nx.cartesian_product(xy_network, sub_networks[2])

    mapping = {((a, b), c): TranslationNode(a, b, c) for ((a, b), c) in full_network.nodes}
    full_network = nx.relabel_nodes(full_network, mapping)
    full_network = TranslationNetwork(full_network)
    # # all the properties we want to calculate
    full_network.calculate_all_edge_properties()
    return full_network

