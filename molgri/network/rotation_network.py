from functools import cached_property
from typing import Tuple

import networkx as nx
import numpy as np
from numpy._typing import NDArray
from scipy.sparse import coo_array
from scipy.spatial import ConvexHull, geometric_slerp
from scipy.spatial.transform import Rotation

from molgri.network.utils import AbstractNetwork, AbstractNode, ReducedSphericalVoronoi
from molgri.network.polytope import Cube4DPolytope
from molgri.utils import all_rows_unique, hemisphere_quaternion_set, q_in_upper_sphere, quaternion_in_array, \
    random_quaternions, \
    distance_between_quaternions, \
    exact_area_of_spherical_polygon, find_shared_quaternions, cut_off_constant_dimension, \
    double_coverage_from_upper_quaternions, two_sets_of_quaternions_equal


class RotationNode(AbstractNode):

    """
    This class is a single container node with sub-nodes x, y and z, representing one particular
    translation.
    """

    def __init__(self, rotation_index: int, quaternion: NDArray, hypersphere_hull = None):
        self.index = rotation_index
        self.coordinate = quaternion
        self.hull = hypersphere_hull

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


    def volume(self):
        print(self.coordinate, self.index)
        # numerically estimate the volume
        level_of_detail = 20 # higher = more detail (interpolation points)
        # additional points are slerps between hull points
        additional_points = []
        for index_1, point1 in enumerate(self.hull):
            for point2 in self.hull[index_1+1:]:
                points = geometric_slerp(point1, point2, t=np.linspace(0, 1, level_of_detail))
                additional_points.append(points)
        all_hull_points = np.vstack([np.vstack(additional_points), self.hull])
        my_convex_hull = ConvexHull(all_hull_points, qhull_options='QJ')
        return my_convex_hull.area / 2.0

    def apply_transform_on(self, molecular_coordinates: NDArray) -> NDArray:
        # todo: important to consider center of mass?
        center_of_geometry = molecular_coordinates.mean(axis=0)
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
        print(node1.coordinate, node2.coordinate, "\n", np.round(shared_vertices, 3))
        lower_dim_points = cut_off_constant_dimension(shared_vertices)
        return  {"rotational": exact_area_of_spherical_polygon(lower_dim_points)}

    def _numerical_edge_type(self, edge_dict) -> dict:
        return  {"rotational": 4}

def create_rotation_network(algorithm_keyword: str = "hypercube", *args, **kwargs) -> RotationNetwork:
    match algorithm_keyword:
        case "random":
            quaternions = random_quaternions(*args, only_upper=True, **kwargs)
        case "hypercube":
            polytope = Cube4DPolytope()
            quaternions = polytope.create_exactly_N_points(*args, **kwargs)
            print("generated quat are", quaternions)
        case _:
            raise KeyError(f"{algorithm_keyword} is not a valid rotation algorithm keyword")
    assert len(quaternions) == args[0]
    all_rows_unique(quaternions)
    return _create_network_from_upper_quaternions(quaternions)


def _adjacency_hulls_from_upper_quaternions(upper_quaternions: NDArray) -> Tuple[coo_array, list]:
    N_upper_points = upper_quaternions.shape[0]
    double_coverage_points = double_coverage_from_upper_quaternions(upper_quaternions)
    unit_spherical_voronoi = ReducedSphericalVoronoi(double_coverage_points)
    hulls = unit_spherical_voronoi.get_hulls()

    # why can we just use the hulls of half the points? We only use the hulls to calculate lengths/areas/volumes, so
    # we need a set of points that are in the proximity of q or in the proximity of -q, but we don't really care if
    # these points are in the upper or lower hemisphere. If we combine vertices from both q and -q then we no longer
    # have only one sregion but two and so calculating its volume just gets more complex.
    single_coverage_hulls = hulls[:N_upper_points]

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
    return coo_array(upper_left), single_coverage_hulls

def _create_network_from_upper_quaternions(upper_quaternions: NDArray) -> RotationNetwork:
    G = nx.Graph()
    adj_matrix, all_hulls = _adjacency_hulls_from_upper_quaternions(upper_quaternions)
    print(adj_matrix.shape)

    all_layer_nodes = [RotationNode(rot_i, quat, all_hulls[rot_i]) for rot_i, quat in enumerate(upper_quaternions)]
    G.add_nodes_from(all_layer_nodes)
    for node_i_1, node_i_2 in zip(adj_matrix.row, adj_matrix.col):
        node1 = all_layer_nodes[node_i_1]
        node2 = all_layer_nodes[node_i_2]
        G.add_edge(node1, node2, edge_type="rotational")
    my_network = RotationNetwork(G)
    return my_network

