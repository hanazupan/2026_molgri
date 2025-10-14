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
from molgri.utils import all_rows_unique, which_row_is_k, angle_between_vectors


class AbstractNode(ABC):

    @abstractmethod
    def __lt__(self, other: "AbstractNode") -> bool:
        pass

    @abstractmethod
    def hull(self) -> NDArray:
        pass

    @abstractmethod
    def volume(self) -> float:
        pass

    @abstractmethod
    def apply_transform_on(self, molecular_coordinates: NDArray) -> NDArray:
        pass

    def get_transformed_bimolecular_structure(self, static_coordinates: NDArray, moving_coordinates: NDArray) -> NDArray:
        transformed_moving_molecule = self.apply_transform_on(moving_coordinates)
        merged_coordinates = np.vstack([static_coordinates, transformed_moving_molecule])
        return merged_coordinates

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
        self.calculate_all_edge_properties()

    def create_pseudotrajectory_coordinates_from(self, static_coordinates: NDArray, moving_coordinates: NDArray):
        nodes = [node.get_transformed_bimolecular_structure(static_coordinates, moving_coordinates) for node in sorted(self.nodes)]
        return nodes

    @cached_property
    def sorted_nodes(self):
        nodes = [node for node in sorted(self.nodes)]
        return nodes

    @cached_property
    def volumes(self) -> NDArray:
        volumes = [node.volume() for node in self.sorted_nodes]
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
        print(df_edges)
        # now list all properties to be calculated
        df_edges["numerical_edge_type"] = df_edges.apply(
            lambda row: self._numerical_edge_type(row.to_dict())[row["edge_type"]], axis=1)
        df_edges["distance"] = df_edges.apply(
            lambda row: self._distances(row.to_dict())[row["edge_type"]], axis=1)
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
        if num_points == 1:
            # this is mocked
            self.points = points
            self.vertices = np.array([[0, 0, 1]])
            self.regions = [[0]]
            # each point is assigned proportional areas
            if self.num_dimensions == 3:
                self.areas = np.array([4*np.pi])
        elif 1 < num_points <= 4:
            raise ValueError(f"For technical reasons, the number of name can be either 1 or >4, your choice of "
                             f"{num_points} is not supported.")
        else:
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
