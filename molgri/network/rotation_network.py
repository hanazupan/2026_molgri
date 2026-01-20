from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from molgri.network.abstract import AbstractNetwork, AbstractNode
from molgri.utils.spheres import exact_area_of_spherical_polygon
from molgri.utils.quaternions import (find_shared_quaternions,
                                      distance_between_quaternions,
                                      cut_off_constant_dimension_quat)


class RotationNode(AbstractNode):

    """
    This class is a single container node with sub-nodes x, y and z, representing one particular
    translation.
    """

    def __init__(self, rotation_index: int, quaternion: NDArray, hypersphere_hull=None, hull_volume=None):
        self.index = rotation_index
        self.coordinate = quaternion
        self.hull = hypersphere_hull
        self.volume = hull_volume

    def hull(self) -> NDArray:
        return self.hull

    def __str__(self):
        return f'quat={self.index}'

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.index < other.index


    def apply_transform_on(self, molecular_coordinates: NDArray, weights=None) -> NDArray:
        """
        Appy the rotation of this node onto the rigid body defined by given molecular coordinates. If weights are not
        provided,the rotation is done around geometrical center, otherwise around the center of mass.

        Args:
            molecular_coordinates (NDArray): an array of molecular coordinates of shape (N_atoms, 3)
            weights (NDArray or None): usually given as a list of atomic weights of length N_atoms

        Returns:
            an array of molecular coordinates of shape (N_atoms, 3) after the rotation is applied
        """
        center_of_geometry = np.average(molecular_coordinates, axis=0, weights=weights)
        shifted_points = molecular_coordinates - center_of_geometry
        rot = Rotation.from_quat(self.coordinate, scalar_first=True)
        rotated_points = rot.apply(shifted_points)
        rotated_points += center_of_geometry
        return rotated_points


class RotationNetwork(AbstractNetwork):

    @cached_property
    def grid(self):
        coordinates = [node.coordinate for node in self.sorted_nodes]
        return np.array(coordinates)

    def _distances(self, edge_dict) -> dict:
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        return {"rotational": distance_between_quaternions(node1.coordinate, node2.coordinate)}

    def _surfaces(self, edge_dict) -> dict:
        node1 = edge_dict["source"]
        node2 = edge_dict["target"]
        shared_vertices = find_shared_quaternions(node1.hull, node2.hull)
        if np.linalg.matrix_rank(shared_vertices) < 4:
            lower_dim_points = cut_off_constant_dimension_quat(shared_vertices)
            area = exact_area_of_spherical_polygon(lower_dim_points)
        else:
            area = 0.0
        return  {"rotational": area}

    def _numerical_edge_type(self, edge_dict) -> dict:
        return  {"rotational": 4}