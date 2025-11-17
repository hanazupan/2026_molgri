from functools import cached_property
from typing import Tuple

import networkx as nx
import numpy as np
from numpy._typing import NDArray
from scipy.sparse import coo_array
from scipy.spatial.transform import Rotation

from molgri.network.abstract import AbstractNetwork, AbstractNode, get_spherical_voronoi
from molgri.network.polytope import Cube4DPolytope

from molgri.utils.arrays import (all_rows_unique)
from molgri.utils.spheres import exact_area_of_spherical_polygon
from molgri.utils.quaternions import (double_coverage_from_upper_quaternions, find_shared_quaternions,
                                      hypersphere_voronoi_cell_volumes, random_quaternions,
                                      distance_between_quaternions, cut_off_constant_dimension_quat)


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
        lower_dim_points = cut_off_constant_dimension_quat(shared_vertices)
        return  {"rotational": exact_area_of_spherical_polygon(lower_dim_points)}

    def _numerical_edge_type(self, edge_dict) -> dict:
        return  {"rotational": 4}

def create_rotation_network(algorithm_keyword: str, N_rotations: int, rotation_random_seed, **kwargs) -> (
        RotationNetwork):
    """
    This is a high-level method that is given the algorithm name and the number of points (optionally other
    arguments) and returns a RotationNetwork which consists of Nodes and Edges between them - so as a minimum,
    the function must assign quaternions and their neighbours.

    Args:
        algorithm_keyword (str): defines the way how quaternions are selected, currently: 'random' and 'hypercube'
        N_rotations (int): number of rotations to generate

    Returns:
        a network of rotational nodes connected with edges
    """

    # people may interpret "no rotations" as zero, but actually that means we are using exactly one rotation quaternion
    if N_rotations == 0:
        N_rotations = 1
    # for exactly one rotation just use identity
    if N_rotations == 1:
        quaternions = np.array([[1, 0, 0, 0]])
    else:
        match algorithm_keyword:
            case "random":
                quaternions = random_quaternions(N_rotations, only_upper=True,
                                                 rotation_random_seed=rotation_random_seed, **kwargs)
            case "hypercube":
                polytope = Cube4DPolytope()
                quaternions = polytope.create_exactly_N_points(N_rotations, rotation_random_seed=rotation_random_seed, **kwargs)
            case _:
                raise KeyError(f"{algorithm_keyword} is not a valid rotation algorithm keyword")
        # test that we have the right number of unique quaternions
    assert len(quaternions) == N_rotations
    all_rows_unique(quaternions)
    return _create_network_from_upper_quaternions(quaternions)


def _adjacency_hulls_from_upper_quaternions(upper_quaternions: NDArray) -> Tuple[coo_array, NDArray, NDArray]:
    """
    In this function we deal with two properties of quaternion networks that are affected by double coverage: the
    hulls and the neighbourhood. Both must be first determined for a hypersphere that contains not only the
    quaternions +q but also -q. Afterwards, we only use the hulls of the first half of quaternions (+q), but we also
    consider the neighbours of -q to determine the neighbourhood.

    Args:
        upper_quaternions (NDArray): an array of shape (N_quaternions, 4) that saves all quaternions of the upper
            hemisphere (+q) that we are interested in

    Returns:
        a tuple of adjacency matrix of shape (N_quaternions, N_quaternions) and a list of hull vectors of length
        N_quaternions and a list of volumes of length N_quaternions
    """
    N_upper_points = upper_quaternions.shape[0]
    double_coverage_points = double_coverage_from_upper_quaternions(upper_quaternions)
    unit_spherical_voronoi = get_spherical_voronoi(double_coverage_points)
    hulls = unit_spherical_voronoi.get_hulls()
    volumes = hypersphere_voronoi_cell_volumes(double_coverage_points)


    # why can we just use the hulls of half the points? We only use the hulls to calculate lengths/areas/volumes, so
    # we need a set of points that are in the proximity of q or in the proximity of -q, but we don't really care if
    # these points are in the upper or lower hemisphere. If we combine vertices from both q and -q then we no longer
    # have only one sregion but two and so calculating its volume just gets more complex.
    single_coverage_hulls = hulls[:N_upper_points]
    single_coverage_volumes = volumes[:N_upper_points]

    # this is the matrix double the size of what we need
    adjacency_double_coverage = unit_spherical_voronoi.get_adjacency_matrix().toarray()

    # include the adjacency of opposing neighbours

    # what is happening here: the adjacency_double_coverage matrix has four quadrants. The upper left and the bottom
    # right are the same, so the upper points that are neighbours to other upper points or the lower points that are
    # neighbours to other lower points. What is of interest now are the other two sub-matrices that show when an
    # upper point is a neighbour to a lower point or vice versa.
    # Therefore, here we are copying the positions of the True values from the upper right matrix to the upper left
    # matrix to account for additional neighbourhood relations.
    upper_left = adjacency_double_coverage[:N_upper_points, :N_upper_points]
    upper_right = adjacency_double_coverage[:N_upper_points, N_upper_points:]
    upper_left += upper_right

    # now return only upper left quadrant
    return coo_array(upper_left), single_coverage_hulls, single_coverage_volumes

def _create_network_from_upper_quaternions(upper_quaternions: NDArray) -> RotationNetwork:
    G = nx.Graph()
    adj_matrix, all_hulls, all_volumes = _adjacency_hulls_from_upper_quaternions(upper_quaternions)

    # first creating nodes with their corresponding hulls
    all_layer_nodes = [RotationNode(rot_i, quat, all_hulls[rot_i], all_volumes[rot_i]) for rot_i, quat in enumerate(
        upper_quaternions)]
    G.add_nodes_from(all_layer_nodes)
    # then adding edges from the adjacency matrx
    for node_i_1, node_i_2 in zip(adj_matrix.row, adj_matrix.col):
        node1 = all_layer_nodes[node_i_1]
        node2 = all_layer_nodes[node_i_2]
        G.add_edge(node1, node2, edge_type="rotational")
    my_network = RotationNetwork(G)
    return my_network

