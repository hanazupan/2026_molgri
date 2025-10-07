from functools import cached_property

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
import networkx as nx
from scipy.spatial import ConvexHull, geometric_slerp

from molgri.transgrid import ReducedSphericalVoronoi
from molgri.grid_geometry import calculate_new_edge_attribute

class MolgriGraph(nx.Graph):

    def get_7d_coordinates(self):
        return np.array([node.get_7d_coordinate() for node in sorted(self.nodes)])

    def get_three_indices(self):
        return np.array([node.get_three_indices() for node in sorted(self.nodes)])


    def volumes(self):
        return np.array([node.volume() for node in sorted(self.nodes)])

    @cached_property
    def adjacency_matrix(self):
        return nx.adjacency_matrix(self, nodelist=sorted(self.nodes), dtype=bool)

    @cached_property
    def adjacency_type_matrix(self):
        return nx.adjacency_matrix(self, nodelist=sorted(self.nodes), dtype=int, weight="edge_type")

    @cached_property
    def distance_matrix(self):
        calculate_new_edge_attribute(self, "distance")
        return nx.adjacency_matrix(self, nodelist=sorted(self.nodes), dtype=float, weight="distance")

    @cached_property
    def surface_matrix(self):
        calculate_new_edge_attribute(self, "surface")
        return nx.adjacency_matrix(self, nodelist=sorted(self.nodes), dtype=float, weight="surface")

    # def show_graph(self, node_property: str = "total_index", edge_property: str = "edge_type"):
    #     labels = {node: node_i for node_i, node in enumerate(sorted(self.nodes))}
    #
    #     if edge_property == "edge_type":
    #         edge_labels = {(u,v): edge_data[edge_property] for u, v, edge_data in self.edges(data=True)}
    #     else:
    #         calculate_new_edge_attribute(self, edge_property)
    #         edge_labels = {(u, v): np.round(edge_data[edge_property], 2) for u, v, edge_data in self.edges(data=True)}
    #
    #
    #     type_to_color = {"radial": "red", "spherical": "blue", "rotational": "yellow"}
    #
    #     calculate_new_edge_attribute(self, "numerical_edge_type")
    #
    #     edge_colors = [type_to_color[edge_data["edge_type"]] for u, v, edge_data in self.edges(data=True)]
    #
    #
    #     pos = nx.kamada_kawai_layout(self, weight="numerical_edge_type")
    #     nx.draw(self, pos, labels=labels)
    #     nx.draw_networkx_edges(self, pos, edge_color=edge_colors)
    #     nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)
    #     plt.show()

class FullNode:

    def __init__(self, spherical_node, radial_node, rotation_node = None):
        self.spherical_node = spherical_node
        self.radial_node = radial_node
        self.rotation_node = rotation_node

    def __getattr__(self, name):
        for obj in (self.spherical_node, self.radial_node, self.rotation_node):
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __str__(self):
        return f'({str(self.spherical_node)}, {str(self.radial_node)}, {str(self.rotation_node)})'

    def __lt__(self, other):
        """
        How do we know a node is "larger" (should come later in sorting)
        - first we compare the radial index
        - if both are the same, we compare the spherical index
        - if both are the same, we compare the rotation index
        """
        return self.get_three_indices() < other.get_three_indices()

    def get_7d_coordinate(self):
        coo_3d = normalise_vectors(self.spherical_node.unit_vector, length=self.radial_node.radius)
        return np.concat((coo_3d, self.rotation_node.quaternion))

    def get_three_indices(self):
        return [self.radial_node.radial_index, self.spherical_node.spherical_index, self.rotation_node.rotation_index]

    def volume(self):
        radius_smaller = self.radial_node.hull[0]
        radius_larger = self.radial_node.hull[1]
        # how much of the unit surface is this spherical surface
        percentage = self.spherical_node.unit_voronoi_area / (4 * np.pi)
        # the same percentage of the volume is this cell
        position_volume = 4 / 3 * np.pi * (radius_larger ** 3 - radius_smaller ** 3) * percentage
        return position_volume * self.rotation_node.volume

    @cached_property
    def position_grid_hull(self):
        """
        Returns a list with two elements: first is an array of all hull vertices at the radius smaller than position,
        second an array of all hull vertices at the radius larger than position.
        Returns:

        """
        spherical_hull = self.spherical_node.hull
        radial_hull = self.radial_node.hull

        vertices = []
        # add bottom vertices
        if np.isclose(radial_hull[0], 0.0):
            vertices.append(np.zeros((1, 3)))
        else:
            vertices.append(spherical_hull * np.linalg.norm(radial_hull[0]))
        # add upper vertices
        vertices.append(spherical_hull * np.linalg.norm(radial_hull[1]))
        return vertices


class SphericalNode:

    def __init__(self, spherical_index: int, unit_vector: NDArray, unit_hull = None):
        self.spherical_index = spherical_index
        self.unit_vector = unit_vector
        self.hull = unit_hull

    def __str__(self):
        return f'dir={self.spherical_index}'

    @cached_property
    def unit_voronoi_area(self):
        return exact_area_of_spherical_polygon(self.hull)


class RadialNode:

    def __init__(self, radial_index: int, radius: float, radial_hull = None):
        self.radial_index = radial_index
        self.radius = radius
        self.hull = radial_hull

    def __str__(self):
        return f'rad={self.radial_index}'





def create_unit_radius_network(spherical_points):
    unit_spherical_voronoi = ReducedSphericalVoronoi(spherical_points)
    layer_adjacency = unit_spherical_voronoi.get_adjacency_matrix()
    hulls = unit_spherical_voronoi.get_hulls()

    G = MolgriGraph()
    all_layer_nodes = [SphericalNode(direction_i, coo_3d, hulls[direction_i]) for direction_i, coo_3d in enumerate(spherical_points)]
    G.add_nodes_from(all_layer_nodes)
    for node_i_1, node_i_2 in zip(layer_adjacency.row, layer_adjacency.col):
        node1 = all_layer_nodes[node_i_1]
        node2 = all_layer_nodes[node_i_2]
        G.add_edge(node1, node2, edge_type="spherical")
    return G


def _get_between_radii(radial_grid) -> NDArray:
    """
    If your radial points are [R_1, R_1+dR, R_1+2dR ... R_1+NdR], the in-between radii are
    [0, R_1+1/2 dR, R_1+3/2 dR ... R_1+(N+1/2)dR]

    Returns:
        a 1D array of in-berween radii (in Angstrom)
    """
    if len(radial_grid) == 1:
        increment = radial_grid[0]
    else:
        increment = radial_grid[1] - radial_grid[0]

    between_radii = radial_grid + increment / 2
    between_radii = np.concatenate([[0, ], between_radii])
    return between_radii


def create_radial_network(radial_points):
    G = MolgriGraph()

    in_between_points = _get_between_radii(radial_points)
    radial_hull = [np.array([in_between_points[i], in_between_points[i+1]]) for i, _ in enumerate(radial_points)]
    all_layer_nodes = [RadialNode(radial_i, radius, radial_hull[radial_i]) for radial_i, radius in enumerate(radial_points)]

    # first create just the first layer
    for node in all_layer_nodes:
        G.add_node(node)
    # add edges for all neighbouring levels
    for node_1, node_2 in zip(all_layer_nodes[:-1], all_layer_nodes[1:]):
        G.add_edge(node_1, node_2, edge_type="radial")
    return G

# def create_rotational_network(quaternions):
#     # todo: need adjacency matrix for quaternions (check double coverage!)
#     unit_spherical_voronoi = ReducedSphericalVoronoi(quaternions)
#     layer_adjacency = unit_spherical_voronoi.get_adjacency_matrix()
#     hulls = unit_spherical_voronoi.get_hulls()
#
#     G = MolgriGraph()
#
#     all_layer_nodes = [RotationNode(rot_i, quat, hulls[rot_i]) for rot_i, quat in enumerate(quaternions)]
#     G.add_nodes_from(all_layer_nodes)
#     for node_i_1, node_i_2 in zip(layer_adjacency.row, layer_adjacency.col):
#         node1 = all_layer_nodes[node_i_1]
#         node2 = all_layer_nodes[node_i_2]
#         G.add_edge(node1, node2, edge_type="rotational")
#     return G


def create_full_network(spherical_points, radial_distances, quaternions):
    base_layer_network = create_unit_radius_network(spherical_points)
    rad_network = create_radial_network(radial_distances)
    rotation_network = create_rotational_network(quaternions)

    position_network = nx.cartesian_product(base_layer_network, rad_network)
    full_network = nx.cartesian_product(position_network, rotation_network)

    mapping = { ((a, b), c): FullNode(a, b, c) for ((a, b), c) in full_network.nodes }
    full_network = nx.relabel_nodes(full_network, mapping)
    full_network = MolgriGraph(full_network)

    return full_network


if __name__ == '__main__':
    from molgri.plotting import show_array
    from molgri.polytope import IcosahedronPolytope, Cube4DPolytope
    from molgri.utils import all_row_norms_equal_k, dist_on_sphere, distance_between_quaternions, \
    exact_area_of_spherical_polygon, normalise_vectors, random_sphere_points, random_quaternions


    radial_points = np.linspace(1.5, 4.5, num=3)
    print("radial_points", radial_points)

    # TWO OPTIONS to create spherical points

    spherical_points = random_sphere_points(1)
    # ico = IcosahedronPolytope()
    # ico.divide_edges()
    # spherical_points = ico.get_nodes(projection=True)


    # TWO OPTIONS to create quaternions

    #quaternions = random_quaternions(15)

    cube = Cube4DPolytope()
    cube.create_exactly_N_points(N=16)
    quaternions = cube.get_nodes(projection=True)
    print("quaternions", quaternions)

    full_network = create_full_network(spherical_points, radial_points, quaternions)
    #full_network.show_graph(edge_property="distance")



    # show_array(full_network.adjacency_matrix.toarray(), "Adjacency")
    # show_array(full_network.distance_matrix.toarray(), "Distance")
    # show_array(full_network.surface_matrix.toarray(), "Surface")
    #print(full_network.volumes())