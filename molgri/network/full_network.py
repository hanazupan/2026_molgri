from functools import cached_property

import networkx as nx
import numpy as np
import MDAnalysis as md
from MDAnalysis import Merge
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from molgri.network.rotation_network import RotationNode, RotationNetwork
from molgri.network.translation_network import TranslationNode, TranslationNetwork
from molgri.network.utils import AbstractNetwork, AbstractNode


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
        return np.concat((self.translation_node.coordinate, self.rotation_node.coordinate))

    def get_indices(self):
        return [self.translation_node, self.rotation_node]

    def volume(self):
        return self.translation_node.volume * self.rotation_node.volume

    def hull(self) -> NDArray:
        return (self.translation_node.hull, self.rotation_node.hull)

    def _apply_to_universe(self, moving_molecule: md.Universe) -> md.Universe:
        copy_moving = moving_molecule.copy()
        full_coord = self.get_7d_coordinate()
        position = full_coord[:3]
        orientation = full_coord[3:]
        rotation_body = Rotation.from_quat(orientation, scalar_first=True)
        copy_moving.atoms.rotate(rotation_body.as_matrix(), point=copy_moving.atoms.center_of_mass())
        copy_moving.atoms.translate(position)
        return copy_moving

    def get_transformed_bimolecular_structure(self, static_molecule: md.Universe, moving_molecule: md.Universe) -> (md.Universe):
        transformed_moving_molecule = self._apply_to_universe(moving_molecule)
        merged_universe = Merge(static_molecule.atoms, transformed_moving_molecule.atoms)
        return merged_universe


class FullNetwork(AbstractNetwork):

    def get_psudotrajectory(self, static_molecule: md.Universe, moving_molecule: md.Universe):
        return [node.get_transformed_bimolecular_structure(static_molecule, moving_molecule) for node in self.sorted_nodes]

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


def create_full_network(translation_network: TranslationNetwork, rotation_network: RotationNetwork) -> FullNetwork:
    full_network = nx.cartesian_product(translation_network, rotation_network)
    mapping = {(trans, rot): FullNode(trans, rot) for (trans, rot) in full_network.nodes}
    full_network = nx.relabel_nodes(full_network, mapping)
    return FullNetwork(full_network)