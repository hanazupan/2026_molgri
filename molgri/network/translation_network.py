from abc import ABC
from functools import cached_property
from itertools import product

import networkx as nx
import numpy as np
from numpy._typing import NDArray

from molgri.network.utils import AbstractNetwork, AbstractNode, ReducedSphericalVoronoi, find_shared_vertices, circular_sector_area
from molgri.network.polytope import IcosahedronPolytope
from molgri.utils import exact_area_of_spherical_polygon


class OneDimTranslationNode:

    def __init__(self, direction: str, index: int, coordinate: float, hull: tuple) -> None:
        self.index = index
        self.name = direction
        self.coordinate = coordinate
        self.hull = hull

    def __str__(self) -> str:
        return f"{self.name} grid, index: {self.index}, coordinate: {self.coordinate}"

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

class SphericalNode:

    def __init__(self, spherical_index: int, unit_vector: NDArray, unit_hull = None):
        self.index = spherical_index
        self.coordinate = unit_vector
        self.hull = unit_hull

    def __str__(self):
        return f'Sph node {self.index}'

    def __repr__(self) -> str:
        return self.__str__()

    @cached_property
    def unit_voronoi_area(self):
        return exact_area_of_spherical_polygon(self.hull)

class SphericalTranslationNode(AbstractNode):

    def __init__(self, r: OneDimTranslationNode, sphere: SphericalNode):
        self.r = r
        self.sphere = sphere
        self.coordinate = self.r.coordinate * self.sphere.coordinate

    def __str__(self):
        return f'Sph. tr. node {self.get_two_indices()}'

    def __repr__(self):
        return self.__str__()

    def get_two_indices(self):
        return [self.r.index, self.sphere.index]

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.get_two_indices() < other.get_two_indices()

    @cached_property
    def hull(self):
        spherical_hull = self.sphere.hull
        radial_hull = self.r.hull

        vertices = []
        # add bottom vertices
        if np.isclose(radial_hull[0], 0.0):
            vertices.append(np.zeros((1, 3)))
        else:
            vertices.append(spherical_hull * np.linalg.norm(radial_hull[0]))
        # add upper vertices
        vertices.append(spherical_hull * np.linalg.norm(radial_hull[1]))
        return vertices

    def volume(self):
        radius_smaller = self.r.hull[0]
        radius_larger = self.r.hull[1]
        # how much of the unit surface is this spherical surface
        percentage = self.sphere.unit_voronoi_area / (4 * np.pi)
        # the same percentage of the volume is this cell
        position_volume = 4 / 3 * np.pi * (radius_larger ** 3 - radius_smaller ** 3) * percentage
        return position_volume

    def apply_transform_on(self, molecular_coordinates: NDArray) -> NDArray:
        return molecular_coordinates + self.coordinate


class TranslationNode(AbstractNode):

    """
    This class is a single container node with sub-nodes x, y and z, representing one particular
    translation.
    """

    def __init__(self, x: OneDimTranslationNode, y: OneDimTranslationNode, z: OneDimTranslationNode):
        self.x = x
        self.y = y
        self.z = z
        self.coordinate = np.array([self.x.coordinate, self.y.coordinate, self.z.coordinate])

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


    def volume(self):
        side_1 = self.x.hull[1] - self.x.hull[0]
        side_2 = self.y.hull[1] - self.y.hull[0]
        side_3 = self.z.hull[1] - self.z.hull[0]
        return side_1 * side_2 * side_3

    def apply_transform_on(self, molecular_coordinates: NDArray) -> NDArray:
        return molecular_coordinates + self.coordinate

class TranslationNetwork(AbstractNetwork, ABC):

    @cached_property
    def grid(self):
        coordinates = [node.coordinate for node in self.sorted_nodes]
        return np.array(coordinates)

class SphericalTranslationNetwork(TranslationNetwork):

    def _radial_distance(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode):
        return np.abs(node1.r.coordinate - node2.r.coordinate)

    def _spherical_distance(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode):
        # looking for shared vertices to span our circle slice
        shared_upper = find_shared_vertices(node1.hull[1], node2.hull[1])
        shared_lower = find_shared_vertices(node1.hull[0], node2.hull[0])
        return circular_sector_area(shared_upper, shared_lower)

    def _distances(self, edge_dict) -> dict:
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        if edge_dict["edge_type"] == "r":
            return {"r": self._radial_distance(node1, node2)}
        elif edge_dict["edge_type"] == "spherical":
            return {"spherical": self._spherical_distance(node1, node2)}


    def _radial_surface(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode):
        unit_area = node1.sphere.unit_voronoi_area
        # scale to a radius between both layers
        in_between_radius = np.min([node1.r.hull[1], node2.r.hull[1]])
        return in_between_radius ** 2 * unit_area

    def _spherical_surface(self, node1: SphericalTranslationNode, node2: SphericalTranslationNode):
        # looking for shared vertices to span our circle slice
        shared_upper = find_shared_vertices(node1.hull[1], node2.hull[1])
        shared_lower = find_shared_vertices(node1.hull[0], node2.hull[0])
        return circular_sector_area(shared_upper, shared_lower)

    def _surfaces(self, edge_dict) -> dict:
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        if edge_dict["edge_type"] == "r":
            return {"r": self._radial_surface(node1, node2)}
        elif edge_dict["edge_type"] == "spherical":
            return {"spherical": self._spherical_surface(node1, node2)}

    def _numerical_edge_type(self, edge_dict) -> dict:
        return {"r": 1, "spherical": 2}


class CartesianTranslationNetwork(TranslationNetwork):

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

    def _distances(self, edge_dict) -> dict:
        return {"x": self.delta_x, "y": self.delta_y, "z": self.delta_z}

    def _surfaces(self, edge_dict) -> dict:
        return {"x": self.delta_y*self.delta_z, "y": self.delta_x*self.delta_z, "z": self.delta_x*self.delta_y}

    def _numerical_edge_type(self, edge_dict) -> dict:
        return {"x": 1, "y": 2, "z": 3}

def create_translation_network(algorithm_keyword: str = "cartesian_nonperiodic", *args, **kwargs) -> TranslationNetwork:
    match algorithm_keyword:
        case "cartesian_nonperiodic":
            return _create_cartesian_network("none", *args, **kwargs)
        case "cartesian_periodic":
            return _create_cartesian_network("xyz", *args, **kwargs)
        case "cartesian_xy_periodic":
            return _create_cartesian_network("xy", *args, **kwargs)
        case "spherical":
            return _create_spherical_coordinate_network(*args, **kwargs)
        case _:
            raise KeyError(f"{algorithm_keyword} is not a valid translation algorithm keyword")

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
    full_network = CartesianTranslationNetwork(full_network)
    return full_network

def _create_spherical_coordinate_network(spherical_N_points, radial_parameters):
    radial_network = _create_radial_network(radial_parameters)
    spherical_network = _create_spherical_network(spherical_N_points)
    full_network = nx.cartesian_product(radial_network,spherical_network)
    mapping = {(a, b): SphericalTranslationNode(a, b) for (a, b) in full_network.nodes}
    full_network = nx.relabel_nodes(full_network, mapping)
    full_network = SphericalTranslationNetwork(full_network)
    return full_network

def _create_radial_network(radial_parameters) -> nx.Graph:
    r_grid = np.linspace(*radial_parameters)
    nodes  = []
    delta_r = r_grid[1] - r_grid[0]
    for coo_i, coo in enumerate(r_grid):
        hull = (coo - delta_r / 2, coo + delta_r / 2)
        nodes.append(OneDimTranslationNode("r", coo_i, coo, hull))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    # now add edges to these sub-graphs - this is without periodicity
    for node_1, node_2 in zip(nodes[:-1], nodes[1:]):
        G.add_edge(node_1, node_2, edge_type="r")
    return G

def _create_spherical_network(spherical_N_points):
    ico = IcosahedronPolytope()
    ico.create_exactly_N_points(spherical_N_points)
    spherical_points = ico.get_nodes(projection=True)
    unit_spherical_voronoi = ReducedSphericalVoronoi(spherical_points)
    layer_adjacency = unit_spherical_voronoi.get_adjacency_matrix()
    hulls = unit_spherical_voronoi.get_hulls()

    G = nx.Graph()
    all_layer_nodes = [SphericalNode(direction_i, coo_3d, hulls[direction_i]) for direction_i, coo_3d in enumerate(spherical_points)]
    G.add_nodes_from(all_layer_nodes)
    for node_i_1, node_i_2 in zip(layer_adjacency.row, layer_adjacency.col):
        node1 = all_layer_nodes[node_i_1]
        node2 = all_layer_nodes[node_i_2]
        G.add_edge(node1, node2, edge_type="spherical")
    return G
