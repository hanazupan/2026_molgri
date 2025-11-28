from __future__ import annotations

from abc import abstractmethod, ABC
from copy import copy
from functools import cached_property
from itertools import combinations

import networkx as nx
import numpy as np
from numpy._typing import NDArray
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.spatial import SphericalVoronoi

from molgri.constants import UNIQUE_TOL
from molgri.utils.arrays import all_rows_unique, which_row_is_k, angle_between_vectors


class AbstractNode(ABC):

    @abstractmethod
    def __lt__(self, other: "AbstractNode") -> bool:
        pass

    @property
    @abstractmethod
    def hull(self) -> NDArray:
        pass


    @abstractmethod
    def apply_transform_on(self, molecular_coordinates: NDArray, weights: NDArray = None) -> NDArray:
        pass

class AbstractNetwork(nx.Graph, ABC):

    """
    Just a bit enhanced networkx Graph that I use for my RotationNetwork, TranslationNetwork and FullNetwork.
    The assumptions are:
        - all nodes are objects that can be sorted (have implemented __lt__)
        - all nodes have the properties hull and volume
        - edges have properties edge_type and methods are implemented to further calculate numeric_edge_type, distance
         and surface
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if there is only one node there are no edges and therefore no edge properties
        if self.number_of_nodes() > 1:
            self.calculate_all_edge_properties()

    def create_pseudotrajectory_coordinates_from(self, moving_coordinates: NDArray, weights: NDArray = None) -> list:
        nodes = [node.apply_transform_on(moving_coordinates, weights=weights) for node in sorted(self.nodes)]
        return nodes

    def add_node_property(self, sorted_values_list: list, property_name: str):
        for node_i, node in enumerate(self.sorted_nodes):
            node.__dict__[property_name] = sorted_values_list[node_i]

    def get_node_property(self, property_name: str) -> NDArray:
        chosen_property = [node.__dict__[property_name] for node in self.sorted_nodes]
        chosen_property = np.array(chosen_property, dtype=float)
        return chosen_property

    @cached_property
    def sorted_nodes(self):
        nodes = [node for node in sorted(self.nodes)]
        return nodes

    @cached_property
    def volumes(self) -> NDArray:
        volumes = [node.volume for node in self.sorted_nodes]
        volumes = np.array(volumes, dtype=float)
        return volumes

    @cached_property
    def hulls(self):
        hulls = [node.hull for node in self.sorted_nodes]
        return hulls

    @abstractmethod
    def grid(self) -> NDArray:
        pass

    @abstractmethod
    def _distances(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a method to calculate distance is returned. Edge properties
        dictionary can be given as an argument.
        """
        pass

    @abstractmethod
    def _surfaces(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a method to calculate surface is returned. Edge properties
        dictionary can be given as an argument.
        """
        pass

    @abstractmethod
    def _numerical_edge_type(self, *edge_dict) -> dict:
        """
        Must return a dict in which for every edge type a number is returned.
        """
        pass

    def calculate_all_edge_properties(self):
        df_edges = nx.to_pandas_edgelist(self)
        #print(df_edges)
        # now list all properties to be calculated
        df_edges["numerical_edge_type"] = df_edges.apply(
            lambda row: self._numerical_edge_type(row.to_dict())[row["edge_type"]], axis=1)
        df_edges["distance"] = df_edges.apply(
            lambda row: self._distances(row.to_dict())[row["edge_type"]], axis=1)
        # there is some problem with surfaces, investigate
        df_edges["surface"] = df_edges.apply(
            lambda row: self._surfaces(row.to_dict())[row["edge_type"]], axis=1)
        for attribute in ["distance", "surface", "numerical_edge_type"]:
            nx.set_edge_attributes(self, df_edges.set_index(["source", "target"])[attribute].to_dict(), name=attribute)

    @cached_property
    def adjacency_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=bool)

    @cached_property
    def adjacency_type_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=bool, weight="numerical_edge_type")

    @cached_property
    def distance_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=float, weight="distance")

    @cached_property
    def surface_matrix(self):
        return nx.adjacency_matrix(self, nodelist=self.sorted_nodes, dtype=float, weight="surface")


class ReducedSphericalVoronoi(SphericalVoronoi):
    """
    This layer on top of SphericalVoronoi has two purposes:
    - removing vertices that repeat
    - allowing the choice of exactly one spherical point
    """

    def __init__(self, points, radius=1.0, threshold=10 ** -UNIQUE_TOL):
        assert len(points.shape) == 2, "Must provide a 2D array of points"
        self.num_dimensions = points.shape[1]
        num_points = len(points)

        if num_points <= 4:
            raise ValueError(f"You are using ReducedSphericalVoronoi for < 5 points, where you should be using MikroVoronoi.")

        super().__init__(points, radius=radius, threshold=threshold)
        if self.num_dimensions == 3:
            self.areas = super().calculate_areas()
        self._purge_redundant_voronoi_vertices()
        # make sure no repeated vertices now
        all_rows_unique(self.vertices)

    def calculate_areas(self) -> NDArray:
        """
        This is overwritten with previous values so that the regions are not messed up again after calculating areas.

        Returns:
            an array of areas the same length as the number of points
        """
        return self.areas

    def get_adjacency_matrix(self) -> coo_array:
        """
        Adjacent points share at least dimension-1 vertices.
        """
        if len(self.points) == 1:
            return coo_array(np.zeros((1,1)))

        num_points = len(self.points)
        rows = []
        columns = []
        elements = []

        # neighbours have at least two spherical Voronoi vertices in common
        for index_tuple in combinations(list(range(num_points)), 2):
            set_1 = set(self.regions[index_tuple[0]])
            set_2 = set(self.regions[index_tuple[1]])

            if len(set_1.intersection(set_2)) >= self.num_dimensions - 1:
                rows.extend([index_tuple[0], index_tuple[1]])
                columns.extend([index_tuple[1], index_tuple[0]])
                elements.extend([True, True])

        adj_matrix = coo_array((elements, (rows, columns)), shape=(num_points, num_points))
        return adj_matrix

    def get_hulls(self):
        return [self.vertices[region] for region in self.regions]

    def _purge_redundant_voronoi_vertices(self):
        original_vertices = copy(self.vertices)
        # correctly determines which lines to use
        indexes = np.unique(original_vertices, axis=0, return_index=True)[1]
        new_vertices = np.array([original_vertices[index] for index in sorted(indexes)])

        # regions
        # correctly assigns
        old2new = {old_i: which_row_is_k(new_vertices, old)[0] for old_i, old in enumerate(original_vertices)}
        old_regions = self.regions
        new_regions = []
        for i, region in enumerate(old_regions):
            fresh_region = []
            for j, el in enumerate(region):
                fresh_region.append(old2new[el])
            new_regions.append(list(set(fresh_region)))

        # now we can overwrite
        self.vertices = new_vertices
        self.regions = new_regions

class MikroVoronoi(ReducedSphericalVoronoi):
    """
    This is a class mocking spherical voronoi cells for very small numbers of points (when it's impossible to
    actually get voronoi cells).
    """

    def __init__(self, points, **kwargs):
        assert len(points.shape) == 2, "Must provide a 2D array of points"
        self.num_dimensions = points.shape[1]
        assert self.num_dimensions in [3, 4]
        self.N_points = len(points)
        self.points = points

        if self.num_dimensions  == 3:
            self.vertices = None #np.array([[0, 0, 1]])
            self.regions = None #[[0]]
            # area of unit sphere divided into  N parts
            self.areas = np.array([4 * np.pi / self.N_points] * self.N_points)
        else:
            self.points = points
            self.vertices = None #np.array([[1, 0, 0, 0], [-1, 0, 0, 0]])
            self.regions = None #[[0], [1]]
            # hyperarea of half unit hypersphere  divided into  N parts
            self.areas = np.array([np.pi ** 2 / self.N_points] * self.N_points)

    def get_adjacency_matrix(self) -> coo_array:
        """
        For such a small number of points you are neighbour with everybody except yourself.
        """
        result = np.eye(self.N_points)
        # invert 0s and 1s
        result =1 - result
        return coo_array(result)


    def get_hulls(self):
        return [None] * self.N_points
        #eturn [self.vertices[region] for region in self.regions]




def get_spherical_voronoi(points, **kwargs):
    if len(points) <= 4 and points.shape[1] == 3:
        return MikroVoronoi(points, **kwargs)
    elif len(points) <= 8 and points.shape[1] == 4:
        return MikroVoronoi(points, **kwargs)
    else:
        return ReducedSphericalVoronoi(points, **kwargs)


def find_shared_vertices(vertices1: NDArray, vertices2: NDArray) -> NDArray:
    """
    Given coordinates of vertices around point 1 (vertices1) and coordinates of vertices around point 2 (vertices2),
    find the intersection (vertices that belong to both sets).

    Args:
        vertices1 (NDArray): array of coordinates, shape (N_1, 3)
        vertices2 (NDArray): array of coordinates, shape (N_2, 3)

    Returns:
        array of coordinates, shape (N_3, 3) where N_3 <= N_1 and N_3 <= N_2
    """

    # coordinates of points must be converted to tuples so that intersection of sets can be used
    vertices_point1 = (tuple(i) for i in vertices1)
    vertices_point2 = (tuple(i) for i in vertices2)
    border_vertices = np.array(list(set(vertices_point1).intersection(set(vertices_point2))))
    return border_vertices


def circular_sector_area(shared_upper_vertices: NDArray, shared_lower_vertices: NDArray) -> float:
    """
    Find the area of either circular sector or a difference between a smaller and bigger circular sector (same angle,
    two different radii).

    Args:
        shared_upper_vertices (NDArray): coordinates of the upper arch, should be exactly two
        shared_lower_vertices (NDArray): coordinates of the lower arch, should be exactly two or one if it is (0,0,0)

    Returns:
        area in Angstrom^2
    """
    assert shared_upper_vertices.shape == (2, 3)
    assert shared_lower_vertices.shape == (2, 3) or np.allclose(shared_lower_vertices, 0.0)

    radius_smaller = np.linalg.norm(shared_lower_vertices[0])
    radius_larger = np.linalg.norm(shared_upper_vertices[0])
    angle = angle_between_vectors(shared_upper_vertices[0], shared_upper_vertices[1])

    return (radius_larger ** 2 - radius_smaller ** 2) * angle / 2
