from functools import cached_property

import numpy as np
from numpy._typing import NDArray

from molgri.network.rotation_network import RotationNode, RotationNetwork
from molgri.network.translation_network import TranslationNode, TranslationNetwork
from molgri.network.utils import AbstractNetwork, AbstractNode


class FullNode(AbstractNode):

    def __init__(self, translation_node: TranslationNode, rotation_node: RotationNode):
        self.translation_node = translation_node
        self.rotation_node = rotation_node

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
        return np.concat((self.translation_node.coordinate_3d, self.rotation_node.quaternion))

    def get_indices(self):
        return [self.translation_node, self.rotation_node]

    def volume(self):
        return self.translation_node.volume * self.rotation_node.volume

    def hull(self) -> NDArray:
        return (self.translation_node.hull, self.rotation_node.hull)


class FullNetwork(AbstractNetwork):

    def __init__(self, translation_network: TranslationNetwork, rotation_network: RotationNetwork, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_network = translation_network
        self.rotation_network = rotation_network

    @cached_property
    def grid(self):
        coordinates = [node.get_3d_coordinate() for node in self.sorted_nodes]
        return np.array(coordinates)

    def _distances(self, edge_dict) -> dict:
        rot_dict = self.rotation_network._distances(edge_dict)
        trans_dict = self.translation_network._distances(edge_dict)
        return rot_dict | trans_dict

    def _surfaces(self, edge_dict) -> dict:
        rot_dict = self.rotation_network._surfaces(edge_dict)
        trans_dict = self.translation_network._surfaces(edge_dict)
        return rot_dict | trans_dict

    def _numerical_edge_type(self) -> dict:
        rot_dict = self.rotation_network._numerical_edge_type()
        trans_dict = self.translation_network._numerical_edge_type()
        return rot_dict | trans_dict