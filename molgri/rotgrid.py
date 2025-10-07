"""
Here we create a full rotational (sub-network). The different classes just have different ways of creating the
initial set of points on the hypersphere.
"""
from functools import cached_property

from numpy.typing import NDArray
import numpy as np
import networkx as nx
from scipy.linalg import svd
from scipy.sparse import coo_array
from scipy.spatial import ConvexHull, geometric_slerp

from molgri.polytope import Cube4DPolytope
from molgri.transgrid import ReducedSphericalVoronoi
from molgri.utils import all_row_norms_equal_k, distance_between_quaternions, exact_area_of_spherical_polygon, \
    find_inverse_quaternion, \
    find_shared_quaternions, find_shared_rows, is_array_with_d_dim_r_rows_c_columns, \
    random_quaternions


class RotationNode:

    """
    This class is a single node, representing one particular quaternion.
    """

    def __init__(self, rotation_index: int, quaternion: NDArray, hypersphere_hull = None):
        self.rotation_index = rotation_index
        self.quaternion = quaternion
        self.hull = hypersphere_hull

    def __str__(self):
        return f'quat={self.rotation_index}'

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.rotation_index < other.rotation_index

    @cached_property
    def volume(self):
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


class RotationObject:

    """
    This is used to create an entire quaternion sub-networks with its specific edge distances, surfaces etc.
    """

    def __init__(self, points_on_upper_hypersphere: NDArray):
        # all assertions - 2D array with any number of rows but always four columns and each row has norm 1
        is_array_with_d_dim_r_rows_c_columns(points_on_upper_hypersphere, d=2, c=4)
        all_row_norms_equal_k(points_on_upper_hypersphere, k=1.0)

        self.upper_points = points_on_upper_hypersphere
        self.N_upper_points = len(self.upper_points)
        self.double_coverage_points = self._copy_upper_points_to_bottom()

    @cached_property
    def grid(self):
        return self.upper_points

    def get_double_coverage_points(self):
        return self.double_coverage_points

    def _copy_upper_points_to_bottom(self) -> NDArray:
        """
        For each q in self.upper points also get -q so that the resulting list is twice as long.
        """
        all_points = np.zeros((2 * self.N_upper_points, 4))
        all_points[:self.N_upper_points] = self.upper_points
        for i in range(self.N_upper_points):
            inverse_q = find_inverse_quaternion(self.upper_points[i])
            all_points[self.N_upper_points + i] = inverse_q
        return all_points

    @cached_property
    def adjacency(self):
        """
        Obtain the adjacency matrix of the size (self.upper_points, self.upper_points) but which is able to consider
        that adjacency can result from the bottom points too.
        """
        unit_spherical_voronoi = ReducedSphericalVoronoi(self.double_coverage_points)
        # this is the matrix double the size of what we need
        adjacency_double_coverage = unit_spherical_voronoi.get_adjacency_matrix().toarray()

        # include the adjacency of opposing neighbours
        upper_index2lower_index = {i:self.N_upper_points+i for i in range(self.N_upper_points)}
        for i, line in enumerate(adjacency_double_coverage):
            for j, el in enumerate(line):
                if el and j in upper_index2lower_index .keys():
                    adjacency_double_coverage[i][upper_index2lower_index [j]] = adjacency_double_coverage[i][j]

        # now return only upper left quadrant
        adj_matrix = adjacency_double_coverage[:self.N_upper_points, :self.N_upper_points]
        return coo_array(adj_matrix)

    @cached_property
    def hulls(self):
        double_coverage_voronoi = ReducedSphericalVoronoi(self.double_coverage_points)
        hulls = double_coverage_voronoi.get_hulls()
        single_coverage_hulls = hulls[:self.N_upper_points]
        return single_coverage_hulls


    def get_rotation_network(self) -> nx.Graph:
        """
        The most important getter of this class, from simply a set of points you are getting a full network.
        """
        G = nx.Graph()

        all_layer_nodes = [RotationNode(rot_i, quat, self.hulls[rot_i]) for rot_i, quat in enumerate(self.grid)]
        G.add_nodes_from(all_layer_nodes)
        for node_i_1, node_i_2 in zip(self.adjacency.row, self.adjacency.col):
            node1 = all_layer_nodes[node_i_1]
            node2 = all_layer_nodes[node_i_2]
            G.add_edge(node1, node2, edge_type="rotational")

        # all the properties we want to calculate
        calculate_rotation_edge_attributes(G)
        return G


class RandomRotationObject(RotationObject):

    def __init__(self, N_rot: int):
        upper_quaternions = random_quaternions(N_rot, only_upper=True)
        super().__init__(upper_quaternions)


class HypercubeRotationObject(RotationObject):

    def __init__(self, N_rot: int):
        self.polytope = Cube4DPolytope()
        upper_nodes = self.polytope.create_exactly_N_points(N_rot)
        super().__init__(upper_nodes)


def create_rotation_object(N_rot: int, algorithm_keyword: str = "hypercube") -> RotationObject:
    match algorithm_keyword:
        case "hypercube":
            rotation_object = HypercubeRotationObject(N_rot)
        case "random":
            rotation_object = RandomRotationObject(N_rot)
        case _:
            raise KeyError(f"{algorithm_keyword} is not a valid rotation algorithm keyword")

    return rotation_object


def calculate_rotation_edge_attributes(G):
    df_edges = nx.to_pandas_edgelist(G)

    attributes = ["distance", "surface", "numerical_edge_type"]
    functions = [_rotation_distance_between_nodes, _rotation_surface_between_nodes, _numerical_edge_type]

    for attribute, function in zip(attributes, functions):
        df_edges[attribute] = df_edges.apply(lambda row: function(row.to_dict()),axis=1)
        nx.set_edge_attributes(G, df_edges.set_index(["source", "target"])[attribute].to_dict(), name=attribute)


def _numerical_edge_type(edge_attributes: dict):
    type_to_num = {"radial": 3, "spherical": 2, "rotational": 1}
    return type_to_num[edge_attributes["edge_type"]]


def _rotation_distance_between_nodes(edge_attributes: dict):
    node1 = edge_attributes["source"]
    node2 = edge_attributes["target"]
    return distance_between_quaternions(node1.quaternion, node2.quaternion)

def _rotation_surface_between_nodes(edge_attributes: dict):
    node1 = edge_attributes["source"]
    node2 = edge_attributes["target"]

    shared_vertices = find_shared_quaternions(node1.hull, node2.hull)

    # matrix rank of shared_vertices should be one less than the dimensionality of space, because it's a
    # normal sphere (well, some points on a sphere) hidden inside 4D coordinates, or a part of a planar circle
    # expressed with 3D coordinates
    u, s, vh = svd(shared_vertices)
    # rotate till last dimension is only zeros, then cut off the redundant dimension. Now we can correctly
    # calculate borders using lower-dimensional tools
    border_full_rank_points = np.dot(shared_vertices, vh.T)[:, :-1]

    return exact_area_of_spherical_polygon(border_full_rank_points)


if __name__ == "__main__":
    from molgri.plotting import draw_points, show_array
    import plotly.graph_objects as go


    rand_rot = HypercubeRotationObject(20)

    rot_network = rand_rot.get_rotation_network()




