from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from itertools import groupby

from molgri.network.rotation_network import RotationNode
from molgri.network.translation_network import TranslationNode
from molgri.network.abstract import AbstractNetwork, AbstractNode


class FullNode(AbstractNode):

    def __init__(self, translation_node: TranslationNode, rotation_node: RotationNode):
        self.translation_node = translation_node
        self.rotation_node = rotation_node
        self.universe = None

    def __str__(self):
        return f'({str(self.translation_node)}, {str(self.rotation_node)})'

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.get_indices() < other.get_indices()

    def get_7d_coordinate(self):
        return np.concatenate((self.translation_node.coordinate, self.rotation_node.coordinate))

    def get_indices(self):
        return [self.translation_node, self.rotation_node]


    @cached_property
    def volume(self):
        return self.translation_node.volume * self.rotation_node.volume

    def hull(self) -> NDArray:
        return (self.translation_node.hull, self.rotation_node.hull)

    def apply_transform_on(self, molecular_coordinates: NDArray, weights=None) -> NDArray:
        # first the rotation
        rotated_points = self.rotation_node.apply_transform_on(molecular_coordinates)
        # afterwards the translation
        translated_points = self.translation_node.apply_transform_on(rotated_points)
        return translated_points



class FullNetwork(AbstractNetwork):

    @cached_property
    def grid(self):
        coordinates = [node.get_7d_coordinate() for node in self.sorted_nodes]
        return np.array(coordinates)

    def _distances(self, edge_dict) -> dict:
        return {edge_dict["edge_type"]: edge_dict["distance"]}

    def _surfaces(self, edge_dict) -> dict:
        return {edge_dict["edge_type"]: edge_dict["surface"]}

    def _numerical_edge_type(self, edge_dict) -> dict:
        return {edge_dict["edge_type"]: edge_dict["numerical_edge_type"]}

    def get_translation_indices(self) -> NDArray:
        N_translations, N_rotations = self.list_of_position_nodes.shape
        indices = np.array([np.repeat(i, N_rotations) for i in range(N_translations)], dtype=int)
        indices = indices.reshape(-1)
        return indices

    def get_rotation_indices(self) -> NDArray:
        indices = np.array([node.rotation_node.index for node in self.sorted_nodes], dtype=int)
        return indices

    @cached_property
    def list_of_position_nodes(self):
        """
        The result is an array, first line are all nodes at the first position, the second all nodes at the second
        position and so on.
        """
        groups = np.array([list(g) for k, g in groupby(self.sorted_nodes, key=lambda o: o.translation_node)])
        return groups


    def get_property_per_position(self, property_name):
        """
        The result is an array, first line are all nodes at the first position, the second all nodes at the second
        position and so on. But instead of returning the nodes directly we return some property of that node. To do
        this efficiently we use the numpy vectorize function.
        """
        groups = self.list_of_position_nodes
        func = np.vectorize(lambda o: o.get_node_property(property_name))
        return func(groups)


