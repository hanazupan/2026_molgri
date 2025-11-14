from functools import cached_property
from typing import Tuple

import networkx as nx
import numpy as np
from numpy._typing import NDArray
from scipy.sparse import coo_array
from scipy.spatial import ConvexHull, geometric_slerp
from scipy.spatial.transform import Rotation

from molgri.network.utils import AbstractNetwork, AbstractNode, ReducedSphericalVoronoi, get_spherical_voronoi
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


    def volume(self, level_of_detail: int = 10):
        """
        We are numerically estimating the volume of the Voronoi cell.

        Args:
            level_of_detail (int, optional): The number of interpolation points for volume calculation. I have tested
            that after 10, the accuracy doesn't increase, just the calculation time does

        """
        # additional points are slerps between hull points
        def spherical_tetra_volume(u):
            """
            Compute volume of spherical tetrahedron on unit S^3.
            u: (4,4) array of 4 unit 4D vertices
            Returns: 3-volume on unit S^3
            """
            G = u @ u.T  # Gram matrix (dot products)
            det = np.linalg.det(u)
            s = 0.0
            for i in range(4):
                for j in range(i + 1, 4):
                    s += G[i, j]
            vol = 2 * np.arctan2(abs(det), 1 + s)
            return vol

        my_hull = ConvexHull(self.hull, qhull_options='QJ')
        total_vol = []
        for simplex in my_hull.simplices:
            tetra = self.hull[simplex]
            piece_volume = spherical_tetra_volume(tetra)
            sub_hull = ConvexHull(tetra, qhull_options='QJ')
            print(np.round(sub_hull.area/2, 4))
            total_vol.append(piece_volume)
        print(np.sort(np.round(np.array(total_vol), 4)))
        print()
        return np.sum(total_vol)

        additional_points = []
        for index_1, point1 in enumerate(self.hull):


            for point2 in self.hull[index_1+1:]:
                points = geometric_slerp(point1, point2, t=np.linspace(0, 1, level_of_detail))
                additional_points.append(points)
        if additional_points:
            all_hull_points = np.vstack([np.vstack(additional_points), self.hull])
            my_convex_hull = ConvexHull(all_hull_points, qhull_options='QJ')
            return my_convex_hull.area / 2.0
        else:
            all_hull_points = self.hull
            return 0.0


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
        lower_dim_points = cut_off_constant_dimension(shared_vertices)
        return  {"rotational": exact_area_of_spherical_polygon(lower_dim_points)}

    def _numerical_edge_type(self, edge_dict) -> dict:
        return  {"rotational": 4}

def create_rotation_network(algorithm_keyword: str, N_rotations: int, **kwargs) -> RotationNetwork:
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
                quaternions = random_quaternions(N_rotations, only_upper=True, **kwargs)
            case "hypercube":
                polytope = Cube4DPolytope()
                quaternions = polytope.create_exactly_N_points(N_rotations, **kwargs)
            case _:
                raise KeyError(f"{algorithm_keyword} is not a valid rotation algorithm keyword")
        # test that we have the right number of unique quaternions
    assert len(quaternions) == N_rotations
    all_rows_unique(quaternions)
    return _create_network_from_upper_quaternions(quaternions)


def _adjacency_hulls_from_upper_quaternions(upper_quaternions: NDArray) -> Tuple[coo_array, list]:
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
        N_quaternions
    """
    N_upper_points = upper_quaternions.shape[0]
    double_coverage_points = double_coverage_from_upper_quaternions(upper_quaternions)
    unit_spherical_voronoi = get_spherical_voronoi(double_coverage_points)
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

    # first creating nodes with their corresponding hulls
    all_layer_nodes = [RotationNode(rot_i, quat, all_hulls[rot_i]) for rot_i, quat in enumerate(upper_quaternions)]
    G.add_nodes_from(all_layer_nodes)
    # then adding edges from the adjacency matrx
    for node_i_1, node_i_2 in zip(adj_matrix.row, adj_matrix.col):
        node1 = all_layer_nodes[node_i_1]
        node2 = all_layer_nodes[node_i_2]
        G.add_edge(node1, node2, edge_type="rotational")
    my_network = RotationNetwork(G)
    return my_network

