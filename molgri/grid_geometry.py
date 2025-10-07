from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from molgri.full_network import FullNode

import numpy as np
import networkx as nx
from scipy.linalg import svd


from molgri.transgrid import circular_sector_area, find_shared_vertices
from molgri.utils import distance_between_quaternions, exact_area_of_spherical_polygon, find_shared_rows


def calculate_new_edge_attribute(G, attribute: str):
    df_edges = nx.to_pandas_edgelist(G)

    # decide which function to use; function must depend on three parameters: node_1, node_2, dict_of_existing_edge_attributes
    if attribute == "distance":
        function = distance_between_nodes
    elif attribute == "surface":
        function = surface_between_nodes
    elif attribute == "numerical_edge_type":
        function = numerical_edge_type
    else:
        raise ValueError(f"Attribute {attribute} unknown; must be 'distance' or 'surface'.")


    df_edges[attribute] = df_edges.apply(
        lambda row: function(row["source"], row["target"], row.to_dict()),
        axis=1
    )

    nx.set_edge_attributes(G, df_edges.set_index(["source", "target"])[attribute].to_dict(), name=attribute)


def numerical_edge_type(node1: FullNode, node2: FullNode, edge_attributes: dict):
    type_to_num = {"radial": 3, "spherical": 2, "rotational": 1}
    return type_to_num[edge_attributes["edge_type"]]

def distance_between_nodes(node1: FullNode, node2: FullNode, edge_attributes: dict):
    if "edge_type" not in edge_attributes.keys():
        raise ValueError("Cannot calculate edge length if edge_type not given.")

    edge_type = edge_attributes["edge_type"]
    if edge_type == "radial":
        return _radial_distance_between_nodes(node1, node2)
    elif edge_type == "spherical":
        return _spherical_distance_between_nodes(node1, node2)
    elif edge_type == "rotational":
        return _rotation_distance_between_nodes(node1, node2)
    else:
        raise ValueError(f"Unknown edge type: {edge_type}, possible types: 'radial', 'spherical', 'rotational'")


def _radial_distance_between_nodes(node1: FullNode, node2: FullNode):
    return np.abs(node1.radial_node.radius - node2.radial_node.radius)

def _spherical_distance_between_nodes(node1: FullNode, node2: FullNode):
    position_hull_node_1 = node1.position_grid_hull
    position_hull_node_2 = node2.position_grid_hull
    # looking for shared vertices to span our circle slice
    shared_upper = find_shared_vertices(position_hull_node_1[1], position_hull_node_2[1])
    shared_lower = find_shared_vertices(position_hull_node_1[0], position_hull_node_2[0])
    return circular_sector_area(shared_upper, shared_lower)

def _rotation_distance_between_nodes(node1: FullNode, node2: FullNode):
    return distance_between_quaternions(node1.rotation_node.quaternion, node2.rotation_node.quaternion)


def surface_between_nodes(node1: FullNode, node2: FullNode, edge_attributes: dict):
    if "edge_type" not in edge_attributes.keys():
        raise ValueError("Cannot calculate edge surface if edge_type not given.")

    edge_type = edge_attributes["edge_type"]

    if edge_type == "radial":
        return _radial_surface_between_nodes(node1, node2)
    elif edge_type == "spherical":
        return _spherical_surface_between_nodes(node1, node2)
    elif edge_type == "rotational":
        return _rotation_surface_between_nodes(node1, node2)
    else:
        raise ValueError(f"Unknown edge type: {edge_type}, possible types: 'radial', 'spherical', 'rotational'")


def _radial_surface_between_nodes(node1: FullNode, node2: FullNode):
    unit_area = node1.spherical_node.unit_voronoi_area
    # scale to a radius between both layers
    in_between_radius = np.min([node1.radial_node.hull[1], node2.radial_node.hull[1]])
    return in_between_radius ** 2 * unit_area



def _spherical_surface_between_nodes(node1: FullNode, node2: FullNode):
    position_hull_node_1 = node1.position_grid_hull
    position_hull_node_2 = node2.position_grid_hull
    # looking for shared vertices to span our circle slice
    shared_upper = find_shared_vertices(position_hull_node_1[1], position_hull_node_2[1])
    shared_lower = find_shared_vertices(position_hull_node_1[0], position_hull_node_2[0])

    return circular_sector_area(shared_upper, shared_lower)


def _rotation_surface_between_nodes(node1: FullNode, node2: FullNode):
    shared_vertices = find_shared_rows(node1.rotation_node.hull, node2.rotation_node.hull)

    # matrix rank of shared_vertices should be one less than the dimensionality of space, because it's a
    # normal sphere (well, some points on a sphere) hidden inside 4D coordinates, or a part of a planar circle
    # expressed with 3D coordinates
    u, s, vh = svd(shared_vertices)
    # rotate till last dimension is only zeros, then cut off the redundant dimension. Now we can correctly
    # calculate borders using lower-dimensional tools
    border_full_rank_points = np.dot(shared_vertices, vh.T)[:, :-1]
    return exact_area_of_spherical_polygon(border_full_rank_points)